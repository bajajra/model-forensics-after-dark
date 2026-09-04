"""Checkpoint restore and validation (§7.3, §E.15, PACKET-A2 §B5 T3.6).

§7.3's validity condition is not "the files came back" — it is that **a frozen prefix restores
and replays twice with identical deterministic workspace checks**. Restoring once proves the
archive is readable; restoring twice and getting the same tree hash both times proves the
restore is a *function* of the checkpoint and not of whatever happened to be on disk.

The subtlety this guards against is real and was hit in the source repo: a restore that leaves
files from a previous run behind, or that fails to reinstate ownership and mode, produces a tree
that matches neither the checkpoint nor a fresh workspace. The environment's own notes record a
version of this costing ~1-2 silently dropped rollouts per 50.

Mode matters specifically here: a `pre-commit` hook that comes back without its execute bit does
not run, so a restored prefix would silently behave as though no hook existed.
"""

from __future__ import annotations

import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..hashing import sha256_token_ids, sha256_tree
from ..rollout.workspace import chown_to_agent


class RestoreError(RuntimeError):
    """Raised when a checkpoint cannot be restored to the state it recorded."""


@dataclass(frozen=True, slots=True)
class RestoredPrefix:
    """A restored checkpoint: exact token IDs, exact workspace, and the budget left."""

    checkpoint_id: str
    rollout_id: str
    turn_idx: int
    token_ids: list[int]
    workspace: Path
    workspace_tree_hash: str
    cumulative_tokens: int
    remaining_context_budget: int
    record: dict[str, Any]

    def assert_matches_record(self) -> None:
        """Verify the restored state against what the checkpoint claims."""
        expected_tokens = self.record["token_ids_sha256"]
        actual_tokens = sha256_token_ids(self.token_ids)
        if actual_tokens != expected_tokens:
            raise RestoreError(
                f"{self.checkpoint_id}: token IDs hash {actual_tokens} != recorded {expected_tokens}"
            )
        expected_tree = self.record["workspace_tree_hash"]
        if self.workspace_tree_hash != expected_tree:
            raise RestoreError(
                f"{self.checkpoint_id}: workspace tree hash {self.workspace_tree_hash} != "
                f"recorded {expected_tree}"
            )


def restore_checkpoint(checkpoint_dir: Path, target: Path) -> RestoredPrefix:
    """Restore one checkpoint into ``target``, replacing whatever is there.

    ``target`` is removed first, deliberately. Restoring *over* an existing tree is how leftover
    files from a previous run survive into a supposedly pristine workspace.
    """
    record = json.loads((checkpoint_dir / "checkpoint.json").read_text())
    token_ids = json.loads(Path(record["token_ids_uri"]).read_text())

    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with tarfile.open(record["workspace_snapshot_uri"], "r") as tar:
        tar.extractall(staging, filter="tar")
    extracted = staging / "workspace"
    if not extracted.is_dir():
        raise RestoreError(f"{checkpoint_dir}: archive has no 'workspace' member")
    shutil.move(str(extracted), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    chown_to_agent(target)

    restored = RestoredPrefix(
        checkpoint_id=record["checkpoint_id"],
        rollout_id=record["rollout_id"],
        turn_idx=record["turn_idx"],
        token_ids=token_ids,
        workspace=target,
        workspace_tree_hash=sha256_tree(target),
        cumulative_tokens=record["cumulative_tokens"],
        remaining_context_budget=record["remaining_context_budget"],
        record=record,
    )
    restored.assert_matches_record()
    return restored


def verify_double_restore(checkpoint_dir: Path, scratch: Path) -> dict[str, Any]:
    """§7.3's validity condition: restore twice, require identical deterministic checks.

    Returns the evidence rather than a bare bool, so a failure is diagnosable from the record.
    """
    first = restore_checkpoint(checkpoint_dir, scratch / "restore_a")
    second = restore_checkpoint(checkpoint_dir, scratch / "restore_b")

    identical_tree = first.workspace_tree_hash == second.workspace_tree_hash
    identical_tokens = first.token_ids == second.token_ids
    if not (identical_tree and identical_tokens):
        raise RestoreError(
            f"{checkpoint_dir}: two restores diverged "
            f"(tree {first.workspace_tree_hash[:12]} vs {second.workspace_tree_hash[:12]}, "
            f"tokens identical={identical_tokens})"
        )
    return {
        "checkpoint_id": first.checkpoint_id,
        "restores": 2,
        "workspace_tree_hash": first.workspace_tree_hash,
        "token_ids_sha256": sha256_token_ids(first.token_ids),
        "n_tokens": len(first.token_ids),
        "remaining_context_budget": first.remaining_context_budget,
        "identical_across_restores": True,
    }
