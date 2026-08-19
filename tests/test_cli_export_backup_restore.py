"""`iai export` / `iai backup` / `iai restore` — the CLI door out of the store.

The store is encrypted at rest and lived, until now, behind `cp -a ~/.iai-mcp/`
in the README. These tests pin three things:

* `iai export` writes plain JSONL and the export carries the fields that make a
  memory *yours* — the verbatim ``literal_surface`` plus ``provenance`` and
  ``tags`` (previously dropped by ``export_jsonl``).
* `iai backup` / `iai restore` are reachable from the CLI and round-trip a store.
* The export file is created 0600 (it is plaintext).
"""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import iai_cli
from iai_mcp.types import MemoryRecord


@pytest.fixture(autouse=True)
def _crypto(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


def _make_rec(store, text: str, tags: list[str], provenance: list[dict]) -> MemoryRecord:
    rng = np.random.default_rng(len(text))
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(UTC)
    return MemoryRecord(
        id=uuid4(), tier="episodic",
        literal_surface=text,
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=provenance, created_at=now, updated_at=now, tags=tags,
        language="en",
    )


@pytest.fixture()
def populated_store(tmp_path, monkeypatch):
    from iai_mcp.store import MemoryStore

    store_dir = tmp_path / "store"
    monkeypatch.setenv("IAI_MCP_STORE", str(store_dir))
    store = MemoryStore(path=store_dir)
    try:
        store.insert(_make_rec(store, "the surface stays verbatim, even with odd  spacing",
                               tags=["role:user", "t:export"],
                               provenance=[{"ts": "2026-08-18T22:00:00+00:00", "cue": "test", "session_id": "s-1", "role": "user"}]))
        store.insert(_make_rec(store, "second record", tags=["t:export"], provenance=[]))
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
    return store_dir


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_export_writes_jsonl_with_verbatim_provenance_and_tags(populated_store, tmp_path, capsys):
    out = tmp_path / "mem.jsonl"
    rc = iai_cli.main(["export", "--output", str(out)])
    assert rc == 0
    assert out.exists()
    rows = _lines(out)
    assert len(rows) == 2
    by_text = {r["literal_surface"]: r for r in rows}
    rec = by_text["the surface stays verbatim, even with odd  spacing"]  # double space preserved
    assert rec["tags"] == ["role:user", "t:export"]
    assert rec["provenance"] and rec["provenance"][0]["session_id"] == "s-1"
    for key in ("id", "tier", "aaak_index", "created_at", "never_decay", "never_merge", "language"):
        assert key in rec
    assert f"exported → {out}" in capsys.readouterr().out


def test_export_default_path_is_in_store_and_0600(populated_store):
    rc = iai_cli.main(["export"])
    assert rc == 0
    exports = sorted(populated_store.glob("export-*.jsonl"))
    assert exports, "default export lands in the store dir"
    mode = stat.S_IMODE(os.stat(exports[-1]).st_mode)
    if os.name != "nt":
        assert mode == 0o600, f"export must be owner-only, got {oct(mode)}"


def test_backup_and_restore_round_trip_from_cli(populated_store, tmp_path, capsys):
    archive = tmp_path / "brain.tar.gz"
    assert iai_cli.main(["backup", "--output", str(archive)]) == 0
    assert archive.exists() and archive.stat().st_size > 0
    target = tmp_path / "restored"
    assert iai_cli.main(["restore", str(archive), "--target", str(target)]) == 0
    assert (target / "hippo").exists()
    assert any(p.name.startswith("brain.") for p in (target / "hippo").iterdir())
    out = capsys.readouterr().out
    assert "backup →" in out and "restored →" in out


def test_export_help_says_the_file_is_plaintext(capsys):
    with pytest.raises(SystemExit) as exc:
        iai_cli.main(["export", "--help"])
    assert exc.value.code == 0
    text = " ".join(capsys.readouterr().out.split())  # argparse re-wraps; compare on words
    assert "this file is plaintext" in text


def _surfaces_in_fresh_process(store_path):
    """Read every literal_surface back from ``store_path`` in a fresh process:
    proves the on-disk store + key, not this process's caches."""
    import json as _json
    import subprocess
    import sys

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
    assert proc.returncode == 0, proc.stderr[-400:]
    return _json.loads(proc.stdout.strip().splitlines()[-1])


def test_restored_store_reads_back(populated_store, tmp_path):
    archive = tmp_path / "brain.tar.gz"
    assert iai_cli.main(["backup", "--output", str(archive)]) == 0
    target = tmp_path / "restored"
    assert iai_cli.main(["restore", str(archive), "--target", str(target)]) == 0
    assert _surfaces_in_fresh_process(target) == [
        "second record",
        "the surface stays verbatim, even with odd  spacing",
    ]


def test_restore_rejects_a_corrupt_archive_before_touching_the_store(populated_store, tmp_path, capsys):
    bad = tmp_path / "bad.tar.gz"
    bad.write_bytes(b"not a tarball")
    key_before = (populated_store / ".crypto.key").read_bytes() if (populated_store / ".crypto.key").exists() else None
    rc = iai_cli.main(["restore", str(bad), "--target", str(populated_store)])
    assert rc == 2
    assert "cannot open" in capsys.readouterr().err
    assert not list(populated_store.parent.glob(".pre-restore-*"))
    if key_before is not None:
        assert (populated_store / ".crypto.key").read_bytes() == key_before


def test_restore_into_live_store_refuses_while_daemon_is_up(populated_store, tmp_path, monkeypatch, capsys):
    archive = tmp_path / "brain.tar.gz"
    assert iai_cli.main(["backup", "--output", str(archive)]) == 0
    monkeypatch.setattr(iai_cli, "_daemon_is_up", lambda: True)
    rc = iai_cli.main(["restore", str(archive)])  # no --target → the live store
    assert rc == 2
    assert "daemon is running" in capsys.readouterr().err
    assert not list(populated_store.parent.glob(".pre-restore-*"))
    # Naming the live store explicitly via --target is still the live store.
    rc = iai_cli.main(["restore", str(archive), "--target", str(populated_store)])
    assert rc == 2
    assert not list(populated_store.parent.glob(".pre-restore-*"))
    # --force goes through.
    assert iai_cli.main(["restore", str(archive), "--force"]) == 0
    assert list(populated_store.parent.glob(".pre-restore-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_backup_and_export_are_owner_only_even_under_permissive_umask(populated_store, tmp_path):
    # The archive carries .crypto.key by design; the export is plaintext.
    # Neither may ever sit at the umask default, even for the write window.
    old = os.umask(0o000)
    try:
        archive = tmp_path / "loose-umask.tar.gz"
        assert iai_cli.main(["backup", "--output", str(archive)]) == 0
        assert stat.S_IMODE(os.stat(archive).st_mode) == 0o600
        out = tmp_path / "loose-umask.jsonl"
        assert iai_cli.main(["export", "--output", str(out)]) == 0
        assert stat.S_IMODE(os.stat(out).st_mode) == 0o600
    finally:
        os.umask(old)
