from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = os.path.expanduser("~/.iai-mcp")


def _store_path() -> Path:
    return Path(os.environ.get("IAI_MCP_STORE", DEFAULT_STORE_PATH))


def export_jsonl(output: Path | None = None) -> Path:
    from iai_mcp.store import MemoryStore

    store_dir = _store_path()
    store = MemoryStore(str(store_dir))

    if output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = store_dir / f"export-{ts}.jsonl"

    records = store.all_records()
    count = 0
    with open(output, "w", encoding="utf-8") as f:
        for rec in records:
            entry = {
                "id": str(rec.id),
                "tier": rec.tier,
                "literal_surface": rec.literal_surface,
                "aaak_index": rec.aaak_index,
                "community_id": rec.community_id,
                "centrality": rec.centrality,
                "detail_level": rec.detail_level,
                "pinned": rec.pinned,
                "stability": rec.stability,
                "difficulty": rec.difficulty,
                "created_at": rec.created_at.isoformat() if rec.created_at else None,
                "updated_at": rec.updated_at.isoformat() if rec.updated_at else None,
                "last_reviewed": rec.last_reviewed.isoformat() if rec.last_reviewed else None,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d records to %s", count, output)
    return output


PRIMARY_KEY_FILE = ".crypto.key"
KEY_FILES = (PRIMARY_KEY_FILE, ".crypto.key.pre-rotate")


def _root_member_name(name: str) -> str:
    """Archive member name normalised to its root-level form: "./x", "././x"
    and "x" all become "x"; anything nested keeps its directory."""
    parts = [seg for seg in name.split("/") if seg not in ("", ".")]
    return "/".join(parts)


def _is_key_member(name: str) -> bool:
    # Exact root-level names only. A nested or look-alike path
    # ("hippo/.crypto.key.decoy") is not key material and must not
    # suppress the warning or the carry-forward logic.
    return _root_member_name(name) in KEY_FILES


def backup(output: Path | None = None, *, include_key: bool = True) -> Path:
    """Write a tar.gz of the store (DB + WAL/SHM, config, side stores).

    By default the archive includes ``.crypto.key`` (and any pre-rotation
    generation), so it restores standalone on a fresh machine — which also
    means the archive is as sensitive as the key itself. Pass
    ``include_key=False`` to leave the key files out; such an archive can only
    be restored where the matching key already exists (see ``restore()``,
    ``use_existing_key``). Note that "without key files" is not "ciphertext
    only": side stores such as ``.memory-bank`` hold plaintext snippets and
    travel in the archive either way.
    """
    store_dir = _store_path()

    if output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = store_dir / f"brain-backup-{ts}.tar.gz"

    items_to_include: "list[tuple[Path, str]]" = []

    # Retained pre-rotation generations travel with the key: restoring a
    # backup taken during an outstanding partial rotation without that file
    # strands exactly the rows it exists to save.
    key_names = list(KEY_FILES) if include_key else []
    for name in [
        *key_names,
        "config.json",
        "lifecycle_state.json",
        ".daemon-state.json",
    ]:
        p = store_dir / name
        if p.exists():
            items_to_include.append((p, name))

    # The memory database itself: everything under hippo/ except what is
    # rebuilt from the DB at boot (RO snapshot, ANN index, column index),
    # transient locks, and temp litter. The DB travels together with its
    # WAL/SHM so a live-store snapshot stays crash-consistent — recovery
    # replays it exactly like a power loss. Globbing the directory instead
    # of hardcoding file names means an engine file rename can never again
    # silently drop the database from the backup set.
    hippo_dir = store_dir / "hippo"
    if hippo_dir.exists():
        for p in sorted(hippo_dir.iterdir()):
            name = p.name
            if name == ".lock" or name.startswith("tmp"):
                continue
            if name.endswith((".ro", ".hnsw", ".colindex")):
                continue
            items_to_include.append((p, f"hippo/{name}"))

    for bank_name in ("bank", ".memory-bank"):
        bank_dir = store_dir / bank_name
        if bank_dir.exists():
            items_to_include.append((bank_dir, bank_name))

    if not any(arc.startswith("hippo/brain.") for _, arc in items_to_include):
        logger.warning(
            "backup: no database file found under %s — the archive holds "
            "config/key material only",
            hippo_dir,
        )

    # The archive may hold key material and always holds the store: owner-only
    # from the first byte (same posture as .crypto.key itself). Written through
    # a 0600 descriptor to a temp name, fsynced, then atomically renamed, so no
    # partial or umask-readable archive is ever visible at the final path.
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.parent / f"{output.name}.tmp.{os.getpid()}"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    fd = os.open(str(tmp), flags, 0o600)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as fobj:
            with tarfile.open(fileobj=fobj, mode="w:gz") as tar:
                for full_path, arcname in items_to_include:
                    tar.add(str(full_path), arcname=arcname)
            fobj.flush()
            os.fsync(fobj.fileno())
        os.replace(str(tmp), str(output))
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    if os.name == "nt":
        from iai_mcp._ipc import restrict_file_to_current_user

        restrict_file_to_current_user(output)

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info("Backup created: %s (%.1f MB, %d items)", output, size_mb, len(items_to_include))
    key_included = [arc for _, arc in items_to_include if _is_key_member(arc)]
    if key_included:
        logger.warning(
            "backup: %s contains the encryption key (%s) — anyone holding this "
            "archive can decrypt every memory in it. Store it like the key, "
            "not like a database dump; use include_key=False for an archive "
            "without key files, which restores only next to your existing key.",
            output,
            ", ".join(key_included),
        )
    else:
        logger.info(
            "backup: %s carries no key files; it can only be restored where the "
            "matching key already exists (restore(..., use_existing_key=True)).",
            output,
        )
    return output


class RestoreKeyError(RuntimeError):
    """The archive holds no primary key and the caller has not said which key
    the restored rows should be read with."""


def _validate_archive(archive: Path, target: Path) -> tuple[list[tarfile.TarInfo], bool]:
    """Open the archive and check every member BEFORE anything on disk moves.

    Returns (members, archive_has_primary_key). Raises on a missing/corrupt
    archive or a member that would escape ``target``.
    """
    with tarfile.open(str(archive), "r:gz") as tar:
        members = tar.getmembers()
    resolved_target = target.resolve()
    for member in members:
        member_path = (resolved_target / member.name).resolve()
        # Containment check, not a string prefix: "/a/b-evil" shares the
        # prefix of target "/a/b" but is a sibling escape.
        if not member_path.is_relative_to(resolved_target):
            raise ValueError(f"Path traversal detected in archive: {member.name}")
    has_primary = any(_root_member_name(m.name) == PRIMARY_KEY_FILE for m in members)
    return members, has_primary


def _prove_readable(target: Path) -> None:
    """Open the restored store and decrypt its records once. A wrong key
    fails here, loudly, instead of at the first recall days later."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=target)
    try:
        for rec in store.all_records():
            _ = rec.literal_surface
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()


def restore(archive: Path, target: Path | None = None, *, use_existing_key: bool = False) -> Path:
    """Unpack ``archive`` into ``target`` (default: the store dir).

    Existing data at ``target`` is moved to a sibling ``.pre-restore-<ts>``
    directory, never deleted. The archive is fully validated before that move.

    If the archive carries no ``.crypto.key`` (``backup(include_key=False)``)
    and ``target`` already has one, the call refuses unless
    ``use_existing_key=True`` — a keyless archive from *another* store would
    otherwise be silently paired with the wrong key. With ``use_existing_key``
    the current key is written into the restored store via the project's key
    writer and the restore is proven by decrypting the records once.
    """
    if target is None:
        target = _store_path()

    members, archive_has_key = _validate_archive(archive, target)

    existing_key_bytes: dict[str, bytes] = {}
    if target.exists():
        for name in KEY_FILES:
            k = target / name
            if k.exists():
                existing_key_bytes[name] = k.read_bytes()

    if not archive_has_key and PRIMARY_KEY_FILE in existing_key_bytes and not use_existing_key:
        raise RestoreKeyError(
            f"{archive} holds no {PRIMARY_KEY_FILE} and {target} already has one. "
            "If this archive was taken from THIS store (backup(include_key=False)), "
            "pass use_existing_key=True to restore next to the existing key; "
            "if it came from another store, restore into an empty target and "
            "place that store's key there."
        )

    pre_restore: Path | None = None
    if target.exists() and any(target.iterdir()):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        pre_restore = target.parent / f".pre-restore-{ts}"
        shutil.move(str(target), str(pre_restore))
        logger.info("Existing data moved to %s", pre_restore)

    target.mkdir(parents=True, exist_ok=True)

    with tarfile.open(str(archive), "r:gz") as tar:
        for member in members:
            member_path = (target / member.name).resolve()
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                member_path.parent.mkdir(parents=True, exist_ok=True)
                if _is_key_member(member.name):
                    # Key bytes reach disk only through the project's key
                    # writer (0600 fd, fsync, atomic replace, Windows ACL).
                    from iai_mcp.crypto import write_key_material

                    with tar.extractfile(member) as src:
                        write_key_material(member_path, src.read())
                    continue
                with tar.extractfile(member) as src, open(member_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    if not archive_has_key:
        if use_existing_key and PRIMARY_KEY_FILE in existing_key_bytes:
            from iai_mcp.crypto import write_key_material

            for name, data in existing_key_bytes.items():
                write_key_material(target / name, data)
            logger.info(
                "Archive had no key; wrote the existing %s into the restored store",
                ", ".join(existing_key_bytes),
            )
            try:
                _prove_readable(target)
            except Exception as exc:  # any decrypt failure must surface
                raise RestoreKeyError(
                    f"restored {target} does not decrypt with the existing key "
                    f"({exc.__class__.__name__}: {exc}). The previous store is intact at "
                    f"{pre_restore}; the archive was likely taken from a different store."
                ) from exc
        elif not (target / PRIMARY_KEY_FILE).exists():
            logger.warning(
                "restore: %s holds no %s and none exists at %s — the restored "
                "records cannot be decrypted until the matching key is placed "
                "there (or IAI_MCP_CRYPTO_PASSPHRASE derives it).",
                archive,
                PRIMARY_KEY_FILE,
                target,
            )

    logger.info("Restored brain from %s to %s", archive, target)
    return target
