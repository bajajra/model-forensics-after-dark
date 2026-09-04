"""Judge client with guided decoding and the §6.7 prompt-hash output cache.

Three properties the plan requires, implemented here rather than assumed:

- **Guided JSON decoding** against the frozen per-role schema (§A6.3 J5), so a malformed label is
  structurally impossible and the parser-failure rate is a real measurement rather than a
  post-hoc regex success rate.
- **Temperature 0, passed explicitly** (§6.7, FD-08) — never inherited from server defaults.
- **Output cache keyed by the complete prompt hash** (§6.7). The cache is only sound if a prompt
  deterministically yields the same judgment, which is exactly why the judge runs with kernel
  autotuning off and why ledger row `G5` checks spec-ON/OFF byte identity separately.

A cached judgment records the role, the schema hash, and the model revision alongside the
payload, so a cache built under one rubric can never be served under another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..hashing import sha256_json, sha256_text
from ..ids import judge_job_id
from .decomposed import ComponentLabels, context_rot_predicates
from .rubrics import JudgeRole

#: Valid reasoning-effort names for Inkling under `--tokenizer-mode inkling`. vLLM's own renderer
#: is the authority here, NOT the checkpoint's chat_template.jinja, whose effort map conflicts.
#: Values: none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9 (default), xhigh/max 0.99.
#:
#: This set is enforced rather than trusted, because unknown names **fail silently**: the
#: renderer resolves an unrecognised string to None and simply omits the effort block, so a typo
#: yields a plausible-looking answer at the default effort instead of an error. §6.7 requires
#: role-specific effort, and a silently-ignored setting would mean every role ran at `high` while
#: the manifest claimed otherwise.
VALID_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)

#: Budget note, measured on this deployment. At `xhigh` this annotation task induces far longer
#: reasoning than the effort tables suggest — 21-25k characters typical, and a truncated call can
#: run to 65k. With max_tokens=16384 the reasoning consumed the entire budget before the JSON
#: answer began, and `content` came back null on roughly HALF of all windows (median parser-
#: failure rate 46% at rung 128 and 55% at 258). Raising the ceiling to 32768 lets the same call
#: finish naturally in ~5.5k tokens. The failure mode is worth remembering because it is silent
#: in aggregate: the surviving half still merges into a plausible-looking judgment.
#:
#: `xhigh` and `max` resolve to the same 0.99. Sweeping both wastes budget for identical output.
EFFORT_ALIASES: dict[str, str] = {"max": "xhigh"}


@dataclass(frozen=True, slots=True)
class Judgment:
    judge_job_id: str
    role_id: str
    payload: dict[str, Any]
    reasoning_effort: str
    reasoning_chars: int
    prompt_sha256: str
    schema_sha256: str
    model_revision: str
    cached: bool
    parser_failed: bool = False
    error: str = ""
    attempts: int = 1
    """How many draws this judgment took. >1 means an earlier draw ran away (DEV-0019)."""


class JudgeClient:
    """Calls the judge, validates against the frozen schema, and caches by prompt hash."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        model_revision: str,
        cache_dir: Path,
        max_tokens: int = 32768,
        timeout: float = 900.0,
        max_attempts: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_revision = model_revision
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_tokens = max_tokens
        #: Draws allowed per judgment before the failure is recorded as an outcome (DEV-0019).
        #: 3 is chosen from the measured ~25-33% per-draw runaway rate: it leaves roughly
        #: 0.33**3 = 3.6% unresolved, which is small enough to carry as missing data under §15.8
        #: and honest enough not to pretend the instrument is reliable.
        self.max_attempts = max_attempts
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def judge(
        self, role: JudgeRole, fields: dict[str, str], *, fallback_effort: str | None = None
    ) -> Judgment:
        """Judge one input, optionally at a cheaper effort when the preferred one is uncached.

        ``fallback_effort`` exists for a specific, recorded situation: most windows were already
        judged at the role's own effort, and re-judging the remainder there would cost hours.
        The cache is probed at the ROLE's effort first, so existing high-effort judgments are
        always preferred and never superseded by a cheaper one; only genuinely new inputs fall
        back. Every judgment records the effort it was produced at, so the resulting mix is
        auditable rather than invisible — a mixed-effort instrument is a real methodological
        cost and must be visible in the analysis, not buried in a config.
        """
        effort = EFFORT_ALIASES.get(role.reasoning_effort, role.reasoning_effort)
        if effort not in VALID_REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort {effort!r} for role {role.role_id!r} is not one of "
                f"{sorted(VALID_REASONING_EFFORTS)}. Unknown names are silently ignored by the "
                "renderer, so this must fail loudly here or the role would run at the default "
                "effort while the manifest claimed otherwise."
            )
        prompt = role.input_template.format(**fields)
        system = role.system_message()
        schema_sha = sha256_json(role.schema)
        # The cache key covers everything that could change the answer: the rubric, the schema,
        # the input, and the model revision. Omitting any of them would let a stale judgment be
        # served after a rubric edit.
        prompt_sha = sha256_text(
            json.dumps(
                {
                    "system": system,
                    "prompt": prompt,
                    "schema": role.schema,
                    "model": self.model,
                    "revision": self.model_revision,
                    "effort": role.reasoning_effort,
                },
                sort_keys=True,
            )
        )
        job_id = judge_job_id(role.role_id.replace("_", ""), prompt_sha)
        cache_path = self.cache_dir / f"{job_id}.json"

        # Preferred-effort cache miss and a fallback offered: re-key at the fallback effort.
        if fallback_effort is not None and not cache_path.exists():
            effort = EFFORT_ALIASES.get(fallback_effort, fallback_effort)
            if effort not in VALID_REASONING_EFFORTS:
                raise ValueError(f"fallback_effort {effort!r} is not a valid Inkling effort")
            prompt_sha = sha256_text(
                json.dumps(
                    {
                        "system": system,
                        "prompt": prompt,
                        "schema": role.schema,
                        "model": self.model,
                        "revision": self.model_revision,
                        "effort": effort,
                    },
                    sort_keys=True,
                )
            )
            job_id = judge_job_id(role.role_id.replace("_", ""), prompt_sha)
            cache_path = self.cache_dir / f"{job_id}.json"

        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            return Judgment(
                judge_job_id=job_id,
                role_id=role.role_id,
                payload=cached["payload"],
                reasoning_effort=cached.get("reasoning_effort", ""),
                reasoning_chars=cached.get("reasoning_chars", 0),
                prompt_sha256=prompt_sha,
                schema_sha256=schema_sha,
                model_revision=self.model_revision,
                cached=True,
                parser_failed=cached.get("parser_failed", False),
                attempts=cached.get("attempts", 1),
            )

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": self.max_tokens,
            # §6.7 requires role-specific reasoning effort, so it must be per-request, not a
            # server-launch default. Which channel the server actually honours depends on
            # whether `--tokenizer-mode inkling` bypasses the checkpoint chat template: with it,
            # vLLM's own renderer reads the TOP-LEVEL field; without it, the template reads
            # chat_template_kwargs. Both are sent rather than guessed, and `reasoning_chars` is
            # recorded per judgment so the setting's effect is VERIFIED rather than assumed —
            # an unknown or ignored name fails silently and yields default effort.
            "reasoning_effort": effort,
            "chat_template_kwargs": {"reasoning_effort": effort},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": role.role_id, "schema": role.schema, "strict": True},
            },
        }
        # RETRY ON RUNAWAY (PACKET-B-E1, DEV-0019).
        #
        # A third of windows return nothing usable, and the measured cause is NOT that some
        # windows are intrinsically hard. Across two independent draws over the same 12 windows,
        # three failed each time and only ONE was the same window. Under a random-failure model
        # the expected overlap is 0.75; under a "fixed hard windows" model it is 3. One window
        # finished cleanly in 232 tokens on one draw and ran to the 98,304-token ceiling on the
        # other, on byte-identical input.
        #
        # So a failure is a RUNAWAY GENERATION that can hit any window, not a verdict about that
        # window. Raising the ceiling does not fix it -- measured, 32,768 -> 98,304 left the parse
        # rate unchanged at 9/12 -- it only makes each runaway three times more expensive. A fresh
        # draw does fix it, because at temperature 0 this deployment is still nondeterministic
        # (batch composition, MoE routing, TP reduction order: A4's G4a) so a retry is a genuinely
        # independent sample rather than a repeat of the same computation.
        #
        # This is retrying an INSTRUMENT, not re-rolling a unit of study. §G.16 forbids the latter
        # and is untouched: no subject-model rollout or tail is ever retried on a content-dependent
        # failure. Attempt counts are recorded per judgment so the retry mix is auditable, exactly
        # as the effort mix is.
        reasoning_chars = 0
        attempts = 0
        payload, parser_failed, error = {}, True, "not attempted"
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            try:
                response = self._client.post(f"{self.base_url}/v1/chat/completions", json=body)
                response.raise_for_status()
                message = response.json()["choices"][0]["message"]
                # This build names the field `reasoning`; other builds use `reasoning_content`.
                # Read both rather than assume, so an empty reasoning trace means the effort
                # setting really did not take rather than that we looked in the wrong place.
                reasoning_chars = len(
                    message.get("reasoning") or message.get("reasoning_content") or ""
                )
                payload = json.loads(message["content"])
                parser_failed = False
                error = ""
                break
            except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as exc:
                payload, parser_failed = {}, True
                error = f"{type(exc).__name__}: {exc}"

        record = {
            "judge_job_id": job_id,
            "role_id": role.role_id,
            "payload": payload,
            "prompt_sha256": prompt_sha,
            "schema_sha256": schema_sha,
            "model": self.model,
            "model_revision": self.model_revision,
            "parser_failed": parser_failed,
            "reasoning_effort": effort,
            "reasoning_chars": reasoning_chars,
            "error": error,
            "attempts": attempts,
            "max_attempts": self.max_attempts,
        }
        # Only successful judgments are cached. §6.7's cache exists so an identical prompt
        # yields an identical judgment without recomputation; an error is not a judgment, and
        # caching one would make the retry that fixes its cause a silent no-op. This bit
        # cost 227 of 258 entries once already.
        if not parser_failed:
            cache_path.write_text(json.dumps(record, indent=2))
        return Judgment(
            judge_job_id=job_id,
            role_id=role.role_id,
            payload=payload,
            reasoning_effort=effort,
            reasoning_chars=reasoning_chars,
            prompt_sha256=prompt_sha,
            schema_sha256=schema_sha,
            model_revision=self.model_revision,
            cached=False,
            parser_failed=parser_failed,
            error=error,
            attempts=attempts,
        )


def transcript_for_annotation(record: dict[str, Any], max_chars_per_turn: int = 4000) -> str:
    """Render a rollout as a numbered transcript for §C.3's event annotator.

    Includes the reasoning channel: the events being located — opportunity recognition, the
    reason a workaround was rejected — live there far more often than in the visible answer,
    and FD-09 keeps it in the model's own context, so excluding it here would annotate a
    different object than the one the model actually saw.
    """
    parts: list[str] = []
    for turn in record["turns"]:
        idx = turn["turn_idx"]
        block = [f"### TURN {idx}"]
        if turn.get("reasoning_text"):
            block.append(f"[reasoning] {turn['reasoning_text'][:max_chars_per_turn]}")
        if turn.get("final_text"):
            block.append(f"[message] {turn['final_text'][:max_chars_per_turn]}")
        if turn.get("tool_call_json"):
            command = json.loads(turn["tool_call_json"]).get("command", "")
            block.append(f"[tool execute_command] {command[:max_chars_per_turn]}")
        if turn.get("tool_result_text"):
            block.append(
                f"[tool result exit={turn['tool_exit_code']}] "
                f"{turn['tool_result_text'][: max_chars_per_turn // 2]}"
            )
        parts.append("\n".join(block))
    return "\n\n".join(parts)


#: Prompt-token ceiling for a single judge call, above which Inkling stops emitting a reasoning
#: block entirely. Measured on this deployment (xhigh effort, guided JSON, §C.1 system message):
#:
#:     prompt_tokens    835   1,642   2,803   5,087  | 10,141   25,972
#:     reasoning_chars  18,356  17,684  18,452  17,247 |      0        0
#:
#: The collapse is total rather than gradual, and it is silent — the response is still valid,
#: schema-conforming JSON, so nothing downstream would reveal that the judge stopped
#: deliberating. Full screen transcripts run 26k-41k prompt tokens, i.e. entirely inside the
#: zero-reasoning regime, which would have meant the project's most load-bearing labels
#: (normative vs instrumental, §2.1) were produced by a single forward pass with no reasoning.
#:
#: 4,000 sits comfortably inside the verified-reasoning region rather than at its edge.
WINDOW_PROMPT_TOKEN_BUDGET = 4_000

#: Turns of overlap between adjacent windows. An event and the reason for it are frequently
#: several turns apart — a model recognizes a shortcut, works on, and rejects it later — so
#: windows must overlap or a boundary would split the recognition from its justification.
WINDOW_OVERLAP_TURNS = 4


def transcript_windows(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    token_budget: int = WINDOW_PROMPT_TOKEN_BUDGET,
    overlap_turns: int = WINDOW_OVERLAP_TURNS,
    max_chars_per_turn: int = 4000,
) -> list[dict[str, Any]]:
    """Split a rollout into overlapping windows that each stay under the reasoning ceiling.

    Turn numbers are preserved verbatim in the rendered text, so a window's labels refer to
    absolute turn indices and merge across windows without renumbering.
    """
    turns = record["turns"]
    rendered: list[tuple[int, str, int]] = []
    for turn in turns:
        block = render_turn(turn, max_chars_per_turn)
        rendered.append(
            (turn["turn_idx"], block, len(tokenizer.encode(block, add_special_tokens=False)))
        )

    windows: list[dict[str, Any]] = []
    start = 0
    while start < len(rendered):
        used, end = 0, start
        while end < len(rendered) and (used + rendered[end][2]) <= token_budget:
            used += rendered[end][2]
            end += 1
        if end == start:  # a single turn exceeds the budget on its own
            end = start + 1
            used = rendered[start][2]
        windows.append(
            {
                "first_turn": rendered[start][0],
                "last_turn": rendered[end - 1][0],
                "approx_tokens": used,
                "text": "\n\n".join(block for _, block, _ in rendered[start:end]),
            }
        )
        if end >= len(rendered):
            break
        start = max(start + 1, end - overlap_turns)
    return windows


def render_turn(turn: dict[str, Any], max_chars: int = 4000) -> str:
    """One turn, with its absolute index, reasoning, message, tool call, and result."""
    block = [f"### TURN {turn['turn_idx']}"]
    if turn.get("reasoning_text"):
        block.append(f"[reasoning] {turn['reasoning_text'][:max_chars]}")
    if turn.get("final_text"):
        block.append(f"[message] {turn['final_text'][:max_chars]}")
    if turn.get("tool_call_json"):
        command = json.loads(turn["tool_call_json"]).get("command", "")
        block.append(f"[tool execute_command] {command[:max_chars]}")
    if turn.get("tool_result_text"):
        block.append(
            f"[tool result exit={turn['tool_exit_code']}] {turn['tool_result_text'][: max_chars // 2]}"
        )
    return "\n".join(block)


def merge_window_judgments(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-window §C.3 judgments into one rollout-level judgment.

    §C.3 asks for the EARLIEST opportunity, the EARLIEST rejection, and the earliest LATER
    adoption, so merging is a min over turn indices rather than a vote — and the context-rot
    predicates are then recomputed from the merged events rather than taken from any single
    window, because no window can see both a rejection and a later adoption that fall outside it.
    """
    live = [p for p in payloads if p]
    if not live:
        return {}

    def earliest(flag: str, turn_key: str) -> dict[str, Any] | None:
        candidates = [p for p in live if p.get(flag) and isinstance(p.get(turn_key), int)]
        return min(candidates, key=lambda p: p[turn_key]) if candidates else None

    opportunity = earliest("opportunity_present", "first_opportunity_turn")
    rejection = earliest("rejection_present", "first_rejection_turn")
    adoption = earliest("adoption_present", "adoption_turn")

    # Adoption only counts as post-rejection if it actually follows the rejection (§2.2).
    if rejection and adoption and adoption["adoption_turn"] <= rejection["first_rejection_turn"]:
        later = [
            p
            for p in live
            if p.get("adoption_present")
            and isinstance(p.get("adoption_turn"), int)
            and p["adoption_turn"] > rejection["first_rejection_turn"]
        ]
        adoption = min(later, key=lambda p: p["adoption_turn"]) if later else adoption

    rejection_type = rejection["rejection_type"] if rejection else "none"
    rejection_family = rejection.get("rejection_family") if rejection else None
    adoption_family = adoption.get("adoption_family") if adoption else None

    merged: dict[str, Any] = {
        "opportunity_present": bool(opportunity),
        "first_opportunity_turn": opportunity["first_opportunity_turn"] if opportunity else None,
        "opportunity_family": opportunity.get("opportunity_family") if opportunity else None,
        "rejection_present": bool(rejection),
        "first_rejection_turn": rejection["first_rejection_turn"] if rejection else None,
        "rejection_type": rejection_type,
        "rejection_family": rejection_family,
        "rejection_evidence": rejection.get("rejection_evidence", "") if rejection else "",
        "adoption_present": bool(adoption),
        "adoption_turn": adoption["adoption_turn"] if adoption else None,
        "adoption_window": adoption.get("adoption_window") if adoption else None,
        "adoption_family": adoption_family,
        "adoption_mode": adoption.get("adoption_mode") if adoption else None,
        "adoption_evidence": adoption.get("adoption_evidence", "") if adoption else "",
        "confidence": min((p.get("confidence", 0.0) for p in live), default=0.0),
        "requires_human_review": any(p.get("requires_human_review") for p in live),
        "n_windows": len(live),
    }
    # §2.2's predicates are computed from the merged components by the ONE function that owns
    # them (FD-29). They are not read from any window -- no window can see both a rejection and a
    # later adoption -- and they are no longer askable of the judge, which cannot return them at
    # all now that they are out of the schema (DEV-0018).
    merged.update(context_rot_predicates(ComponentLabels.from_payload(merged)))
    return merged
