"""Where a tail's frozen prefix lives, and how its outcome is read (Fastrack S1).

Two tail corpora exist and they were written by different studies with different record shapes.
Rather than special-casing at every call site, each is described once here:

``e1_h16``
    E1's resampled continuations from natural rollouts. Checkpoint prefixes under
    ``artifacts/frozen_prefixes/e1_natural``. Outcome is the record's own ``initiation`` block,
    whose ``basis`` is the deterministic grader over the workspace tree.

``e3`` / ``e3_short`` / ``e3_ladder``
    E3's continuations from *constructed* branches. Prefixes under
    ``artifacts/e3_branches{,_short,_ladder}/<prefix_short>/<cell>``. These records carry no
    ``initiation`` block, so the outcome is read from ``per_turn_signatures`` — the same
    tree-derived signal ``initiation`` is built from, one level lower.

**The tree overrides the judge wherever the tree can see it** (CLAUDE.md section 4a rule 5), and
both readings here are tree-derived. Neither corpus's outcome passes through a semantic label,
which is why the S1 result is not capped as judge-labelled the way most of this project's
semantic claims are.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path("/home/jupyter/active/interp/context-rot")


@dataclass(frozen=True, slots=True)
class TailSource:
    """One tail corpus: where its records and frozen prefixes are."""

    name: str
    tails_dir: Path
    branch_root: Path | None
    frozen_root: Path | None

    def checkpoint_dir(self, tail: dict[str, Any]) -> Path:
        if self.frozen_root is not None:
            return self.frozen_root / tail["checkpoint_id"]
        assert self.branch_root is not None
        # `e3_<prefix_short>_<cell>`; the cell may itself contain underscores (`D20_0`).
        rest = tail["checkpoint_id"].removeprefix("e3_")
        prefix_short, cell = rest.split("_", 1)
        return self.branch_root / prefix_short / cell


SOURCES: dict[str, TailSource] = {
    "e1_h16": TailSource(
        name="e1_h16",
        tails_dir=REPO / "data" / "raw" / "e1_tails_h16",
        branch_root=None,
        frozen_root=REPO / "artifacts" / "frozen_prefixes" / "e1_natural",
    ),
    "e3": TailSource(
        name="e3",
        tails_dir=REPO / "data" / "raw" / "e3_tails",
        branch_root=REPO / "artifacts" / "e3_branches",
        frozen_root=None,
    ),
    "e3_short": TailSource(
        name="e3_short",
        tails_dir=REPO / "data" / "raw" / "e3_tails_short",
        branch_root=REPO / "artifacts" / "e3_branches_short",
        frozen_root=None,
    ),
    "e3_ladder": TailSource(
        name="e3_ladder",
        tails_dir=REPO / "data" / "raw" / "e3_tails_ladder",
        branch_root=REPO / "artifacts" / "e3_branches_ladder",
        frozen_root=None,
    ),
}


def read_outcome(tail: dict[str, Any]) -> tuple[bool, int | None]:
    """``(task_game_initiated, first_signature_turn)`` from whichever record shape this is."""
    initiation = tail.get("initiation")
    if initiation is not None:
        return bool(initiation["task_game_initiated"]), initiation.get("first_signature_turn")
    turns = {
        int(k): v for k, v in (tail.get("per_turn_signatures") or {}).items() if v
    }
    if not turns:
        return False, None
    return True, min(turns)


def brief_length(tail: dict[str, Any], source: TailSource) -> int:
    """How many leading tokens are the task brief, for echo stripping.

    Both corpora's streams begin with the same rendered brief, because an E3 branch is a
    continuation of the E1 rollout named in its ``source_prefix_id``. So the boundary is read
    once from that rollout's first assistant turn rather than re-derived per corpus.
    """
    import json

    record_path = REPO / "data" / "raw" / "e1_natural" / f"{tail['source_prefix_id']}.json"
    record = json.loads(record_path.read_text())
    first = min(t["turn_idx"] for t in record["turns"])
    return next(t["token_start"] for t in record["turns"] if t["turn_idx"] == first)
