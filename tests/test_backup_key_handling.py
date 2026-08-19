"""Backups and the encryption key.

The default archive includes ``.crypto.key`` so it restores standalone; that
also makes the archive as sensitive as the key. These tests pin what happens
around that: the archive is owner-only from creation, the caller is warned
when key material is inside, ``include_key=False`` really leaves the key
files out, and restoring a keyless archive over a store that has a key is
refused unless the caller says ``use_existing_key=True`` — in which case the
key is written through the project's key writer and the restore is proven by
reading the records back.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import stat
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import backup as backup_mod
from iai_mcp.types import MemoryRecord


@pytest.fixture(autouse=True)
def _crypto(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


def _make_rec(store, text: str) -> MemoryRecord:
    rng = np.random.default_rng(len(text))
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(UTC)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text,
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=[], language="en",
    )


@pytest.fixture()
def store_dir(tmp_path, monkeypatch):
    from iai_mcp.crypto import KEY_BYTES, write_key_material
    from iai_mcp.store import MemoryStore

    d = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(d))
    # A file key, the way `iai-mcp crypto init` lays one down; it takes
    # precedence over the passphrase env so the store really runs on it.
    write_key_material(d / ".crypto.key", secrets.token_bytes(KEY_BYTES))
    store = MemoryStore(path=d)
    _KEEP.append(store)
    try:
        store.insert(_make_rec(store, "a memory worth keeping"))
    finally:
        store.close()
    assert (d / ".crypto.key").exists(), "fixture precondition: store has a key"
    return d


# Stores opened during a test are kept referenced until the process ends:
# upstream keys some per-process state by id(store), and a collected store's
# id can be handed to the next MemoryStore in the same process (see the
# in-process repro in the PR thread). Not this PR's concern; the tests just
# stay out of its way.
_KEEP: list = []


def _surfaces(store_path):
    """Read every literal_surface back from ``store_path`` in a FRESH process,
    so the read proves the on-disk store + key, not this process's caches."""
    code = (
        "import json, sys\n"
        "from iai_mcp.store import MemoryStore\n"
        "s = MemoryStore(path=sys.argv[1])\n"
        "print(json.dumps(sorted(r.literal_surface for r in s.all_records())))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(store_path)],
        capture_output=True, text=True, check=False,
        env={**os.environ, "IAI_MCP_STORE": str(store_path)},
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "read failed")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _members(archive) -> set[str]:
    with tarfile.open(str(archive), "r:gz") as tar:
        return {m.name for m in tar.getmembers()}


def test_default_backup_includes_key_and_warns_loudly(store_dir, tmp_path, caplog):
    out = tmp_path / "with-key.tar.gz"
    old_umask = os.umask(0o000)  # the archive must be 0600 even under a permissive umask
    try:
        with caplog.at_level(logging.WARNING, logger="iai_mcp.backup"):
            backup_mod.backup(out)
    finally:
        os.umask(old_umask)
    assert ".crypto.key" in _members(out)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "encryption key" in r.getMessage()]
    assert warnings, "a backup that carries the key must say so at WARNING"
    assert ".crypto.key" in warnings[0].getMessage()
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp.*")), "no temp archive left behind"


def test_include_key_false_leaves_key_material_out(store_dir, tmp_path, caplog):
    # Also plant a pre-rotation generation: it is key material too.
    (store_dir / ".crypto.key.pre-rotate").write_bytes(b"old-key-bytes")
    out = tmp_path / "no-key.tar.gz"
    with caplog.at_level(logging.INFO, logger="iai_mcp.backup"):
        backup_mod.backup(out, include_key=False)
    names = _members(out)
    assert not any(n.startswith(".crypto.key") for n in names), names
    assert any(n.startswith("hippo/brain.") for n in names), "the store itself still travels"
    assert not [r for r in caplog.records if r.levelno == logging.WARNING and "encryption key" in r.getMessage()]
    assert any("no key files" in r.getMessage() for r in caplog.records)


def test_keyless_restore_over_keyed_store_is_refused_by_default(store_dir, tmp_path):
    out = tmp_path / "no-key.tar.gz"
    backup_mod.backup(out, include_key=False)
    key_before = (store_dir / ".crypto.key").read_bytes()
    with pytest.raises(backup_mod.RestoreKeyError, match="use_existing_key"):
        backup_mod.restore(out, store_dir)
    # Nothing moved: the live store and its key are exactly where they were.
    assert (store_dir / ".crypto.key").read_bytes() == key_before
    assert not list(store_dir.parent.glob(".pre-restore-*"))
    assert _surfaces(store_dir) == ["a memory worth keeping"]


def test_keyless_restore_with_use_existing_key_is_proven_by_reading_back(store_dir, tmp_path):
    out = tmp_path / "no-key.tar.gz"
    backup_mod.backup(out, include_key=False)
    original_key = (store_dir / ".crypto.key").read_bytes()

    backup_mod.restore(out, store_dir, use_existing_key=True)

    key = store_dir / ".crypto.key"
    assert key.read_bytes() == original_key
    if os.name != "nt":
        assert stat.S_IMODE(os.stat(key).st_mode) == 0o600
    assert _surfaces(store_dir) == ["a memory worth keeping"], "rows read back with the carried key"
    assert list(store_dir.parent.glob(".pre-restore-*")), "previous store kept aside, not deleted"


def test_keyless_archive_from_another_store_fails_loudly_and_keeps_the_old_store(store_dir, tmp_path, monkeypatch):
    from iai_mcp.crypto import KEY_BYTES, write_key_material
    from iai_mcp.store import MemoryStore

    # Store B: a different key, a different memory.
    other = tmp_path / "other-store"
    write_key_material(other / ".crypto.key", secrets.token_bytes(KEY_BYTES))
    monkeypatch.setenv("IAI_MCP_STORE", str(other))
    st = MemoryStore(path=other)
    _KEEP.append(st)
    try:
        st.insert(_make_rec(st, "somebody else's memory"))
    finally:
        st.close()
    keyless_b = tmp_path / "b-no-key.tar.gz"
    backup_mod.backup(keyless_b, include_key=False)

    # Restore B's keyless rows over store A, insisting on A's key: must not
    # silently succeed with unreadable rows.
    monkeypatch.setenv("IAI_MCP_STORE", str(store_dir))
    with pytest.raises(backup_mod.RestoreKeyError, match="does not decrypt"):
        backup_mod.restore(keyless_b, store_dir, use_existing_key=True)
    pre = list(store_dir.parent.glob(".pre-restore-*"))
    assert pre, "the previous store must still exist for recovery"
    assert _surfaces(pre[0]) == ["a memory worth keeping"]


def test_keyless_restore_into_empty_target_warns(store_dir, tmp_path, caplog):
    out = tmp_path / "no-key.tar.gz"
    backup_mod.backup(out, include_key=False)
    target = tmp_path / "fresh-machine"
    with caplog.at_level(logging.WARNING, logger="iai_mcp.backup"):
        backup_mod.restore(out, target)
    assert not (target / ".crypto.key").exists()
    # The rows themselves were still restored — only the key is missing.
    assert any(p.name.startswith("brain.") for p in (target / "hippo").iterdir())
    assert any("cannot be decrypted" in r.getMessage() for r in caplog.records)


def test_default_restore_round_trips_key_and_rows(store_dir, tmp_path):
    out = tmp_path / "with-key.tar.gz"
    backup_mod.backup(out)
    target = tmp_path / "fresh-machine"
    backup_mod.restore(out, target)
    assert (target / ".crypto.key").read_bytes() == (store_dir / ".crypto.key").read_bytes()
    assert _surfaces(target) == ["a memory worth keeping"]


def test_lookalike_nested_member_is_not_key_material(store_dir, tmp_path):
    keyless = tmp_path / "no-key.tar.gz"
    backup_mod.backup(keyless, include_key=False)
    # Rebuild the archive with a nested look-alike added; it must not make
    # the archive count as carrying the key.
    decoy_src = tmp_path / "decoy-bytes"
    decoy_src.write_bytes(b"not a key")
    decoy_archive = tmp_path / "decoy.tar.gz"
    with tarfile.open(str(keyless), "r:gz") as src, tarfile.open(str(decoy_archive), "w:gz") as dst:
        for m in src.getmembers():
            dst.addfile(m, src.extractfile(m) if m.isfile() else None)
        dst.add(str(decoy_src), arcname="hippo/.crypto.key.decoy")
    with pytest.raises(backup_mod.RestoreKeyError):
        backup_mod.restore(decoy_archive, store_dir)


def test_corrupt_archive_is_rejected_before_anything_moves(store_dir, tmp_path):
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"this is not a tarball")
    with pytest.raises(tarfile.TarError):
        backup_mod.restore(bad, store_dir)
    assert not list(store_dir.parent.glob(".pre-restore-*"))
    assert _surfaces(store_dir) == ["a memory worth keeping"]


def test_dot_prefixed_member_names_are_normalised():
    assert backup_mod._root_member_name("./.crypto.key") == ".crypto.key"
    assert backup_mod._root_member_name("././.crypto.key") == ".crypto.key"
    assert backup_mod._is_key_member("././.crypto.key")
    assert not backup_mod._is_key_member("hippo/.crypto.key")
    assert not backup_mod._is_key_member("./hippo/.crypto.key.decoy")
