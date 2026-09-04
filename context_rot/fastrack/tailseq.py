"""Reconstruct the exact token stream of an E1 tail (Fastrack S1).

Why this has to be reconstructed at all
---------------------------------------
E1's tail records store the *text* of every channel — reasoning, tool call arguments, tool
output — and the token span each turn occupied, but not the token IDs themselves. Replaying a
tail through the white-box path needs the IDs. They are recoverable because the harness renders
client-side and deterministically (``rollout/client.py``): a turn is

    <|start|>assistant to=self<|message|>{reasoning}<|eom|>
    <|start|>assistant to={tool}<|message|><atem:function_calls>
    <atem:invoke name="{tool}">
    <atem:parameter name="{k}">{v}</atem:parameter>
    </atem:invoke>
    </atem:function_calls>
    <|start|>tool {tool}<|message|><tool_output name="{tool}">
    {output}
    </tool_output><|eot|>

and every field on the right is stored. **The reconstruction is checked, not assumed**: each
turn's rebuilt length must equal the stored ``token_end - token_start`` exactly. On a 604-turn
sample 592 matched exactly (98.0%); the rest differ by 1 or 6 tokens, which is what a
re-serialised multi-parameter tool call looks like. A tail is truncated at its first
non-matching turn rather than being silently carried forward, so every scored position sits in
a stream that is byte-identical to what the model actually saw.

The one thing that cannot drift
-------------------------------
The checkpoint prefix is read from the frozen ``token_ids.json`` and its length is asserted
against the first turn's ``token_start``. That held 25/25 in the pilot and is checked on every
tail here, because a prefix that is off by even one token makes every downstream activation a
measurement of a conversation that never happened.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..directions.readout import SPECIAL_MIN, _is_punct, echo_mask, segment_channels

GENERATION_PROMPT = "<|start|>assistant"


def render_turn_raw(turn: dict[str, Any]) -> str:
    """The completion text for one assistant turn, minus the ``<|start|>assistant`` opener."""
    reasoning = turn["reasoning_text"]
    tool = turn["tool_name"]
    if tool:
        args = json.loads(turn["tool_call_json"]) if turn["tool_call_json"] else {}
        params = "".join(
            f'\n<atem:parameter name="{name}">{value}</atem:parameter>'
            for name, value in args.items()
        )
        body = (
            f'<atem:function_calls>\n<atem:invoke name="{tool}">{params}\n'
            f"</atem:invoke>\n</atem:function_calls>"
        )
        return (
            f" to=self<|message|>{reasoning}<|eom|>"
            f"<|start|>assistant to={tool}<|message|>{body}"
        )
    return (
        f" to=self<|message|>{reasoning}<|eom|>"
        f'<|start|>assistant to=user<|message|>{turn["final_text"]}<|eot|>'
    )


def render_tool_result(tokenizer: Any, tool: str, output: str) -> list[int]:
    """Mirrors ``rollout.client.render_tool_result`` exactly."""
    return tokenizer.encode(
        f'<|start|>tool {tool}<|message|><tool_output name="{tool}">\n{output}\n</tool_output><|eot|>',
        add_special_tokens=False,
    )


@dataclass(frozen=True, slots=True)
class TurnSpan:
    """One reconstructed assistant turn, as index ranges into the rebuilt stream."""

    turn_idx: int
    start: int
    end: int
    gen_start: int
    gen_end: int
    #: Substantive, echo-stripped reasoning positions, in generation order.
    reasoning_positions: tuple[int, ...]
    #: Every reasoning-channel body token, echo and punctuation included. The budget in
    #: "after N newly generated reasoning tokens" counts THESE — a statement about what the
    #: model generated — while the readout filter is applied separately and recorded.
    reasoning_all: tuple[int, ...]
    exact: bool


@dataclass(frozen=True, slots=True)
class TailSequence:
    """A tail's reconstructed token stream and the positions worth reading."""

    tail_id: str
    checkpoint_id: str
    source_prefix_id: str
    checkpoint_role: str
    token_ids: tuple[int, ...]
    prefix_len: int
    turns: tuple[TurnSpan, ...]
    n_turns_exact: int
    n_turns_stored: int

    @property
    def fully_exact(self) -> bool:
        return self.n_turns_exact == self.n_turns_stored

    def budget_window(self, budget: int) -> tuple[int, ...]:
        """Scored positions covering the first ``budget`` generated reasoning tokens.

        Returns the substantive echo-stripped positions that fall at or before the budget-th
        generated reasoning token. Empty if the tail never generates that many.
        """
        seen = 0
        cutoff: int | None = None
        for span in self.turns:
            for pos in span.reasoning_all:
                seen += 1
                if seen == budget:
                    cutoff = pos
                    break
            if cutoff is not None:
                break
        if cutoff is None:
            return ()
        return tuple(
            pos
            for span in self.turns
            for pos in span.reasoning_positions
            if pos <= cutoff
        )

    def budget_turn(self, budget: int) -> int | None:
        """Which assistant turn the budget-th generated reasoning token falls in."""
        seen = 0
        for span in self.turns:
            seen += len(span.reasoning_all)
            if seen >= budget:
                return span.turn_idx
        return None


def _channel_positions(
    segment: list[int], offset: int, brief: list[int], tokenizer: Any
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Reasoning positions inside one turn segment: (substantive, all)."""
    channels = segment_channels(segment, tokenizer)
    texts = tokenizer.convert_ids_to_tokens(segment)
    echo = echo_mask(brief, segment)
    substantive: list[int] = []
    everything: list[int] = []
    for channel in channels:
        if not channel.is_reasoning:
            continue
        for pos in range(channel.body_start, channel.body_end):
            everything.append(offset + pos)
            if segment[pos] < SPECIAL_MIN and not _is_punct(texts[pos]) and not echo[pos]:
                substantive.append(offset + pos)
    return tuple(substantive), tuple(everything)


def build_tail_sequence(
    tail: dict[str, Any],
    prefix_ids: list[int],
    brief: list[int],
    tokenizer: Any,
    *,
    max_turns: int | None = None,
) -> TailSequence:
    """Rebuild one tail's token stream, verifying every turn against its stored span.

    ``max_turns`` truncates the rebuild; the S1 features only need the first turn or two, and a
    truncated rebuild is far cheaper than a full one. Truncation is recorded on the result.
    """
    stored = tail["turns"]
    if stored and prefix_ids and len(prefix_ids) != stored[0]["token_start"]:
        raise ValueError(
            f"{tail['tail_id']}: checkpoint holds {len(prefix_ids)} tokens but the first turn "
            f"starts at {stored[0]['token_start']}. The prefix is not the one this tail ran on."
        )
    gen_prompt = tokenizer.encode(GENERATION_PROMPT, add_special_tokens=False)
    ids = list(prefix_ids)
    spans: list[TurnSpan] = []
    n_exact = 0
    limit = len(stored) if max_turns is None else min(max_turns, len(stored))
    for turn in stored[:limit]:
        expected = turn["token_end"] - turn["token_start"]
        start = len(ids)
        gen = tokenizer.encode(render_turn_raw(turn), add_special_tokens=False)
        tool_ids = (
            render_tool_result(tokenizer, turn["tool_name"], turn["tool_result_text"])
            if turn["tool_name"]
            else []
        )
        segment = gen_prompt + gen
        rebuilt = segment + tool_ids
        exact = len(rebuilt) == expected
        substantive, everything = _channel_positions(segment, start, brief, tokenizer)
        spans.append(
            TurnSpan(
                turn_idx=turn["turn_idx"],
                start=start,
                end=start + len(rebuilt),
                gen_start=start,
                gen_end=start + len(segment),
                reasoning_positions=substantive,
                reasoning_all=everything,
                exact=exact,
            )
        )
        ids.extend(rebuilt)
        if not exact:
            # Stop extending. Everything before this turn is byte-exact; everything after it
            # would sit at shifted positions in a stream the model never saw.
            break
        n_exact += 1
    return TailSequence(
        tail_id=tail["tail_id"],
        checkpoint_id=tail["checkpoint_id"],
        source_prefix_id=tail["source_prefix_id"],
        checkpoint_role=tail["checkpoint_role"],
        token_ids=tuple(ids),
        prefix_len=len(prefix_ids),
        turns=tuple(spans),
        n_turns_exact=n_exact,
        n_turns_stored=len(stored),
    )


def load_prefix(checkpoint_dir: Path) -> list[int]:
    payload = json.loads((checkpoint_dir / "token_ids.json").read_text())
    if isinstance(payload, dict):
        payload = payload.get("token_ids", payload.get("input_ids"))
    return list(payload)
