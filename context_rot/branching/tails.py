"""Resample short continuations from a frozen prefix (§9.5, §E.3).

This is the measurement E1 exists to make. Everything before it — collection, annotation,
checkpointing — only decides *where* to freeze. Here we restore that frozen moment, run the model
forward a few turns, and count how often it starts task gaming. Do that at the moment just after a
principled refusal and again later, after more failure has accumulated, and the difference between
the two counts is the risk curve.

Five properties matter, and each has cost a study somewhere:

**Restore both halves, and verify before generating.** A tail is only prefix-conditioned if the
prefix is byte-identical: the exact token IDs *and* the exact workspace tree. `RestoredPrefix`
already hashes both against what the checkpoint recorded, and this module refuses to generate if
either differs. Sampling from a nearly-right prefix produces a number that looks fine and answers
a question nobody asked.

**Common random numbers across a prefix's checkpoints** (§E.12, §G.14). `T0` and `T_late` from the
same source prefix share seed-block indices, so tail *i* at one checkpoint and tail *i* at the
other draw the same randomness. The comparison is paired, and the noise that would otherwise sit
between the two estimates largely cancels.

**Seeds derive from content, never from scheduling.** `derive_seed` hashes the master seed with the
prefix, checkpoint, seed block and tail index. Which worker picks up a job, and in what order,
cannot change what it samples.

**An invalid tail is an outcome, not a slot to refill** (§G.16, §15.8). A degenerate or timed-out
continuation stays in the denominator as a category. Replacing it under the same planned index
would quietly select for continuations that behave, which is the exact bias the ITT posture exists
to prevent.

**Stop at the first classified decision, not at the horizon.** §9.5 step 4: a tail ends at H
assistant turns, the first task-gaming initiation, a compliant completion, an honest stop, or an
invalid event — whichever comes first. Running past a decision wastes tokens and, worse, lets a
later event overwrite the one being measured.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..checkpoint.restore import RestoredPrefix, restore_checkpoint
from ..hashing import sha256_token_ids
from ..seeds import SeedKey, derive_seed


@dataclass(frozen=True, slots=True)
class TailJob:
    """One planned continuation. Fully determines what will be sampled."""

    source_prefix_id: str
    checkpoint_id: str
    checkpoint_dir: Path
    seed_block: int
    tail_index: int
    horizon_assistant_turns: int
    study_id: str = "e1tails"
    #: Which arm of the design this tail belongs to — `T0`, `T_late`, `TR`. Recorded rather than
    #: inferred from the checkpoint id, because the label is what the analysis groups on.
    checkpoint_role: str = "T0"
    max_new_tokens: int = 4096

    def seed_key(self) -> SeedKey:
        """§G.14's key — and note what is deliberately ABSENT from it.

        §E.12: *"Within a source prefix and seed block, initialize the same pseudorandom stream
        for every intervention arm. Because logits differ, generated tokens need not match; the
        shared stream still reduces Monte Carlo noise."*

        So `checkpoint_role` must **not** enter the key. Tail *i* at `T0` and tail *i* at `T_late`
        are matched cells of one paired comparison and share a stream; the role is recorded on the
        result for grouping, not used for seeding. Putting it in the key would give every arm an
        independent stream, silently discarding the variance reduction the paired design exists
        for — with no visible symptom, just a wider interval and less power than the sample size
        implies.

        `assignment` is therefore fixed to the study's single arm label here. It becomes a real
        discriminator in E3/E4, where different *treatments* — not different checkpoints of the
        same prefix — genuinely need different streams.
        """
        return SeedKey(
            study_id=self.study_id,
            source_prefix_id=self.source_prefix_id,
            assignment="tail",
            seed_block=self.seed_block,
            tail_index=self.tail_index,
        )


@dataclass(frozen=True, slots=True)
class TailResult:
    """One sampled continuation and everything needed to audit it."""

    job: TailJob
    tail_id: str
    seed: int
    turns: list[dict[str, Any]]
    stop_reason: str
    valid: bool
    invalid_reason: str = ""
    prefix_token_hash: str = ""
    prefix_tree_hash: str = ""
    wall_clock_seconds: float = 0.0
    notes: tuple[str, ...] = field(default=())
    #: Workaround signatures present after each assistant turn, as {turn_idx: [signature, ...]}.
    #: Recorded per turn rather than only at the end so the horizon H is a parameter of the
    #: ANALYSIS, not of the collection: q(r,t,H) for every H <= max_steps is then recoverable
    #: from one run. Scoring only the final state silently fixes H at max_steps.
    per_turn_signatures: dict[int, list[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "tail_id": self.tail_id,
            "source_prefix_id": self.job.source_prefix_id,
            "checkpoint_id": self.job.checkpoint_id,
            "checkpoint_role": self.job.checkpoint_role,
            "seed_block": self.job.seed_block,
            "tail_index": self.job.tail_index,
            "horizon": self.job.horizon_assistant_turns,
            "seed": self.seed,
            "stop_reason": self.stop_reason,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "prefix_token_hash": self.prefix_token_hash,
            "prefix_tree_hash": self.prefix_tree_hash,
            "n_assistant_turns": len(self.turns),
            "wall_clock_seconds": round(self.wall_clock_seconds, 2),
            "notes": list(self.notes),
            "per_turn_signatures": {str(k): v for k, v in sorted(self.per_turn_signatures.items())},
            "turns": self.turns,
        }
        return d


#: Stop reasons that end a tail. `decision` covers §9.5 step 4's "first classified decision".
STOP_HORIZON = "horizon"
STOP_DECISION = "decision"
STOP_FINAL = "final_response"
STOP_TOKENS = "token_cap"
STOP_TIMEOUT = "timeout"
STOP_ERROR = "error"


def verify_prefix(restored: RestoredPrefix) -> None:
    """Refuse to generate from a prefix that is not the one we recorded.

    `assert_matches_record` checks both the token IDs and the workspace tree. Calling it here
    rather than trusting the restore is deliberate: a tail sampled from a drifted prefix is not
    detectably wrong afterwards, so the check has to happen before any tokens are spent.
    """
    restored.assert_matches_record()


def sample_tail(
    job: TailJob,
    master_seed: str,
    client: Any,
    tokenizer: Any,
    tools: list[dict[str, Any]],
    *,
    scratch_root: Path,
    sampling: dict[str, Any],
    tool_timeout: int = 120,
    usable_history_budget: int = 124_292,
    keep_workspace: bool = False,
    signature_probe: Callable[[Path], list[str]] | None = None,
) -> TailResult:
    """Restore one checkpoint and sample a single short continuation from it.

    ``signature_probe`` is called with the live workspace after every assistant turn and should
    return the workaround signatures present at that moment. Supplying it records the *turn at
    which each signature first appears*, which is what makes q(r,t,H) recoverable for every
    H <= the horizon from a single run instead of only at H = horizon.
    """
    from ..rollout.loop import RolloutConfig, run_rollout

    started = time.time()
    seed = derive_seed(master_seed, job.seed_key())
    tail_id = f"tail_{job.checkpoint_id}_b{job.seed_block:03d}_i{job.tail_index:03d}"
    workspace = scratch_root / tail_id

    try:
        restored = restore_checkpoint(job.checkpoint_dir, workspace)
        verify_prefix(restored)
    except Exception as exc:  # noqa: BLE001
        # A restore failure is INFRASTRUCTURE (§G.16) — the prefix never produced output — so it
        # is retryable by the caller under the same index. That is the one case where reusing an
        # index is legitimate, and it is why the reason is recorded distinctly.
        return TailResult(
            job=job,
            tail_id=tail_id,
            seed=seed,
            turns=[],
            stop_reason=STOP_ERROR,
            valid=False,
            invalid_reason=f"restore_failed: {type(exc).__name__}: {exc}",
            wall_clock_seconds=time.time() - started,
            notes=("infrastructure failure — retryable under §G.16",),
        )

    config = RolloutConfig(
        rollout_id=tail_id,
        variant=restored.record.get("variant", ""),
        prompt_variant=restored.record.get("prompt_variant", ""),
        system_prompt="",  # unused: generation resumes from restored token IDs
        user_prompt="",
        env_dir=Path(restored.record.get("env_dir", ".")),
        workspace_root=workspace.parent,
        max_steps=job.horizon_assistant_turns,
        usable_history_budget=usable_history_budget,
        tool_timeout=tool_timeout,
        seed=seed % (2**31),
        max_reasoning_tokens=job.max_new_tokens,
    )

    per_turn: dict[int, list[str]] = {}

    def _probe(turn_idx: int, ws: Path) -> None:
        if signature_probe is None:
            return
        try:
            per_turn[turn_idx] = signature_probe(ws)
        except Exception as exc:  # noqa: BLE001
            # A probe failure must not destroy the tail; record it and keep the turn observable
            # as unmeasured rather than silently as "no signatures".
            per_turn[turn_idx] = [f"__probe_error__:{type(exc).__name__}"]

    try:
        result = run_rollout(
            config,
            client,
            tokenizer,
            tools,
            on_turn=_probe if signature_probe is not None else None,
            resume_token_ids=restored.token_ids,
            resume_workspace=workspace,
        )
        turns = [t if isinstance(t, dict) else t.__dict__ for t in result.turns]
        stop = _classify_stop(result, job.horizon_assistant_turns)
        valid, why = _validity(result, turns)
    except TypeError as exc:
        # run_rollout does not yet accept resume_*; surfaced loudly rather than silently
        # re-running the task from scratch, which would produce a plausible but meaningless tail.
        raise NotImplementedError(
            "run_rollout must accept resume_token_ids and resume_workspace before tails can be "
            f"sampled — a tail that restarts the task is not a continuation. Original: {exc}"
        ) from exc
    finally:
        if not keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)

    return TailResult(
        job=job,
        tail_id=tail_id,
        seed=seed,
        turns=turns,
        per_turn_signatures=per_turn,
        stop_reason=stop,
        valid=valid,
        invalid_reason=why,
        prefix_token_hash=sha256_token_ids(restored.token_ids),
        prefix_tree_hash=restored.workspace_tree_hash,
        wall_clock_seconds=time.time() - started,
    )


def _classify_stop(result: Any, horizon: int) -> str:
    reason = getattr(result, "exit_reason", "") or ""
    if reason in ("completed", "final_response"):
        return STOP_FINAL
    if reason == "context_exhausted":
        return STOP_TOKENS
    if getattr(result, "assistant_turns", 0) >= horizon:
        return STOP_HORIZON
    if reason:
        return reason
    return STOP_HORIZON


def _validity(result: Any, turns: list[dict[str, Any]]) -> tuple[bool, str]:
    """§15.8: invalid tails stay in the denominator as a category, so name the category."""
    if getattr(result, "error", None):
        return False, f"harness_error: {result.error}"
    if not turns:
        return False, "no_assistant_turn_produced"
    return True, ""


def write_tail(result: TailResult, out_dir: Path) -> Path:
    """One JSON per tail. Append-only: never overwrite a planned index (§G.6, §G.16)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{result.tail_id}.json"
    if dest.exists():
        raise FileExistsError(
            f"{dest} already exists. A planned tail index is never re-sampled — replacing one "
            "would select for continuations that behave (§G.16)."
        )
    dest.write_text(json.dumps(result.as_dict(), indent=2))
    return dest
