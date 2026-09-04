"""Content hashing helpers.

Every artifact in this project carries a content hash (CLAUDE.md §9). Prompts are
hashed twice, per `execution_plan.md` §A.1 rule 3: the UTF-8 bytes of every visible
message *before* template rendering, and the rendered token sequence *after*.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of raw bytes, lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of the UTF-8 encoding of ``text`` (§A.1 rule 3, pre-render hash)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's contents, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_token_ids(token_ids: Sequence[int]) -> str:
    """SHA-256 of a token-ID sequence (§A.1 rule 3, post-render hash).

    Token IDs are serialized as a comma-joined decimal string so the hash is
    stable across storage formats (JSON, Parquet, NPZ) and across platforms.
    """
    if any(not isinstance(t, int) or isinstance(t, bool) for t in token_ids):
        raise TypeError("token_ids must be a sequence of plain ints")
    if any(t < 0 for t in token_ids):
        raise ValueError("token_ids must be non-negative")
    return sha256_text(",".join(str(t) for t in token_ids))


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace, UTF-8 preserved."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_json(obj: Any) -> str:
    """SHA-256 of the canonical JSON encoding of ``obj``."""
    return sha256_text(canonical_json(obj))


def sha256_tree(root: str | Path, exclude_names: Iterable[str] = (".git", "__pycache__")) -> str:
    """Content hash of a directory tree: relative path + mode bit + file hash, sorted.

    Used for `repository tree hash` in the §5.8 environment-variant record and for the
    §G.5 `workspace_tree_hash` column. Mode is reduced to the executable bit, which is the
    only permission difference that changes task semantics (a pre-commit hook must be +x).
    """
    root_path = Path(root)
    if not root_path.is_dir():
        raise NotADirectoryError(f"not a directory: {root_path}")
    skip = set(exclude_names)
    entries: list[str] = []
    for path in sorted(root_path.rglob("*")):
        if any(part in skip for part in path.relative_to(root_path).parts):
            continue
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(root_path).as_posix()
        executable = "1" if path.stat().st_mode & 0o111 else "0"
        entries.append(f"{rel}\x00{executable}\x00{sha256_file(path)}")
    return sha256_text("\n".join(entries))
