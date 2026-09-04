"""Seed derivation (`execution_plan.md` §G.14, §E.3).

One private master seed is committed as a hash before confirmatory assignment and revealed
with the release. Every sub-seed is derived by SHA-256 over::

    master_seed | study_id | source_prefix_id | assignment | seed_block | tail_index

Two properties this module exists to guarantee:

- **Worker scheduling must not change a derived seed** (§G.14). The derivation depends only
  on the job's identity, never on hostname, GPU index, wall-clock time, or dispatch order.
- **The same seed-block index is shared across matched treatment cells** (§G.14), so paired
  cells differ by treatment rather than by noise draw.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .hashing import sha256_text

# The derivation is a pure function of these fields, joined by a byte that cannot occur in
# any of them, so no two distinct field tuples can collide by concatenation.
_SEP = "\x00"

MAX_SEED = 1 << 64


@dataclass(frozen=True, slots=True)
class SeedKey:
    """The complete identity of one sampling job (§G.14 derivation tuple)."""

    study_id: str
    source_prefix_id: str
    assignment: str
    seed_block: int
    tail_index: int

    def __post_init__(self) -> None:
        for field in ("study_id", "source_prefix_id", "assignment"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"SeedKey.{field} must be a non-empty string, got {value!r}")
            if _SEP in value:
                raise ValueError(f"SeedKey.{field} must not contain a NUL byte")
        if not 0 <= self.seed_block <= 999:
            raise ValueError(f"seed_block {self.seed_block} outside [0, 999]")
        if not 0 <= self.tail_index <= 9999:
            raise ValueError(f"tail_index {self.tail_index} outside [0, 9999]")

    def key_string(self) -> str:
        """The exact string hashed, excluding the master seed."""
        return _SEP.join(
            (
                self.study_id,
                self.source_prefix_id,
                self.assignment,
                str(self.seed_block),
                str(self.tail_index),
            )
        )


def master_seed_commitment(master_seed: str) -> str:
    """SHA-256 of the master seed, committed publicly before confirmatory assignment (§G.14)."""
    if not master_seed:
        raise ValueError("master_seed must be non-empty")
    return sha256_text(master_seed)


def derive_seed(master_seed: str, key: SeedKey) -> int:
    """Derive the 64-bit sampling seed for one job.

    The full SHA-256 digest is computed over the §G.14 tuple and truncated to its leading
    64 bits, interpreted big-endian.
    """
    if not master_seed:
        raise ValueError("master_seed must be non-empty")
    if _SEP in master_seed:
        raise ValueError("master_seed must not contain a NUL byte")
    payload = f"{master_seed}{_SEP}{key.key_string()}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def derive_priority(master_seed: str, key: SeedKey) -> float:
    """Derive a job-order priority in [0, 1) from the same key (§G.15 step 2).

    Job order is permuted by this priority to distribute temporal and hardware drift across
    arms. It is derived from a distinct domain-separated digest so that permuting job order
    cannot correlate with the sampling seed.
    """
    payload = f"priority{_SEP}{master_seed}{_SEP}{key.key_string()}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / MAX_SEED
