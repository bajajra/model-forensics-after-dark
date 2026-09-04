"""Per-turn activation readout spans (PACKET-D-E2 T44; plan ss.10.7).

ss.10.7 defines four readouts over "substantive assistant-reasoning tokens" and lists what
to exclude: role markers, template separators, pure punctuation, tool-call serialization,
and final-answer tokens. This module adds a fifth exclusion and treats it as load-bearing:

**verbatim prompt echo.**

Measured on this model at "Reasoning strength: high", the first thing the reasoning channel
does is restate the user turn word for word. For an S2 pair that means the frustration
side's reasoning literally contains *Express genuine impatience, stress, or exasperation*
and the calm side's contains *Remain calm, methodical, and emotionally neutral*. A linear
probe fitted on those tokens separates the sides perfectly while learning nothing about
affect -- it learns which instruction is being quoted. E1 hit the same phenomenon from the
other end: the judge's `normative` label tracked verbatim prompt echo (8/8 at 12+ word runs
versus 16/76 when the model used its own words). CLAUDE.md ss.9a therefore requires the text
baseline to carry prompt-echo features; the same reasoning requires the *readout* to drop
those tokens.

The filter is applied identically to synthetic contrastive examples and to real rollout
turns, so it cannot manufacture a train/test asymmetry.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

TOK_START: Final[int] = 200022
TOK_MESSAGE: Final[int] = 200023
TOK_EOM: Final[int] = 200007
TOK_EOT: Final[int] = 200008
SPECIAL_MIN: Final[int] = 200000

REASONING_RECIPIENT: Final[str] = "assistant to=self"
ANSWER_RECIPIENT: Final[str] = "assistant to=user"

#: Token-run length at which a match counts as echo. E1 characterised prompt echo in words
#: ("12+ word runs"); 12 tokens is the nearest token-space analogue for this tokenizer and
#: is deliberately conservative -- shorter runs are ordinary shared technical vocabulary,
#: which ss.10.1 requires the two sides to share anyway.
ECHO_NGRAM: Final[int] = 12

_PUNCT = set(".,;:!?()[]{}<>-_/\\|\"'`~@#$%^&*+=") | {"...", "--", "``", "''"}


@dataclass(frozen=True, slots=True)
class Channel:
    """One ATEM channel inside a completion, as token index ranges."""

    recipient: str
    header_start: int
    body_start: int
    body_end: int  # exclusive; excludes the closing <|eom|>/<|eot|>

    @property
    def is_reasoning(self) -> bool:
        return self.recipient == REASONING_RECIPIENT

    @property
    def is_answer(self) -> bool:
        return self.recipient == ANSWER_RECIPIENT

    @property
    def is_tool(self) -> bool:
        return not self.is_reasoning and not self.is_answer


def segment_channels(completion_ids: Sequence[int], tokenizer: Any) -> list[Channel]:
    """Split a token stream into ATEM channels.

    Two entry conditions must both work, and conflating them is a real bug that cost a
    silent zero-reasoning result on every real rollout turn:

    * **A fresh generation begins *inside* the first channel.** The generation prompt ended
      with ``<|start|>assistant`` and the model continues `` to=self<|message|>...``, so the
      leading ``<|start|>`` and the word ``assistant`` are not in the stream and the
      recipient has to be reconstructed as ``assistant <suffix>``.
    * **A slice of a stored conversation begins *at* the channel.** A saved rollout turn
      starts at its own ``<|start|>`` token, so the recipient is already complete and
      prefixing ``assistant`` again yields ``assistant assistant to=self``, which matches no
      known recipient -- every turn then looks as though it had no reasoning at all.

    The leading token decides which case applies.
    """
    ids = list(completion_ids)
    channels: list[Channel] = []
    starts_at_marker = bool(ids) and ids[0] == TOK_START
    i = 1 if starts_at_marker else 0
    header_start = i
    first = not starts_at_marker
    while i < len(ids):
        if ids[i] == TOK_MESSAGE:
            suffix = tokenizer.decode(ids[header_start:i], skip_special_tokens=True).strip()
            recipient = f"assistant {suffix}".strip() if first else suffix
            body_start = i + 1
            j = body_start
            while j < len(ids) and ids[j] not in (TOK_EOM, TOK_EOT):
                j += 1
            channels.append(Channel(recipient, header_start, body_start, j))
            i = j + 1
            if i < len(ids) and ids[i] == TOK_START:
                i += 1
            header_start = i
            first = False
            continue
        i += 1
    return channels


def echo_mask(
    prompt_ids: Sequence[int], completion_ids: Sequence[int], n: int = ECHO_NGRAM
) -> list[bool]:
    """Mark completion tokens that sit inside an ``n``-token run copied from the prompt."""
    if n <= 0:
        raise ValueError("echo n-gram length must be positive")
    mask = [False] * len(completion_ids)
    if len(prompt_ids) < n or len(completion_ids) < n:
        return mask
    prompt = list(prompt_ids)
    grams = {tuple(prompt[i : i + n]) for i in range(len(prompt) - n + 1)}
    comp = list(completion_ids)
    for i in range(len(comp) - n + 1):
        if tuple(comp[i : i + n]) in grams:
            for j in range(i, i + n):
                mask[j] = True
    return mask


def _is_punct(token_text: str) -> bool:
    stripped = token_text.replace("\u0120", "").replace("\u010a", "").strip()
    return stripped == "" or all(ch in _PUNCT for ch in stripped)


@dataclass(frozen=True, slots=True)
class TurnReadout:
    """Token index sets for one turn's readouts, plus the covariates ss.10.7 needs."""

    substantive: tuple[int, ...]
    final_sentence: tuple[int, ...]
    last_substantive: int | None
    boundary: int | None
    answer: tuple[int, ...]
    reasoning_token_count: int
    substantive_token_count: int
    echo_token_count: int
    echo_fraction: float
    channels: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "n_substantive": self.substantive_token_count,
            "n_reasoning": self.reasoning_token_count,
            "n_echo": self.echo_token_count,
            "echo_fraction": self.echo_fraction,
            "n_final_sentence": len(self.final_sentence),
            "has_last_substantive": self.last_substantive is not None,
            "has_boundary": self.boundary is not None,
            "n_answer": len(self.answer),
            "channels": list(self.channels),
        }


def build_readout(
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    tokenizer: Any,
    *,
    echo_n: int = ECHO_NGRAM,
    strip_echo: bool = True,
) -> TurnReadout:
    """Compute ss.10.7's readout spans over one completion.

    Index space is positions within ``completion_ids``. T44 teacher-forces
    ``prompt_ids + completion_ids`` and reads hidden states at ``len(prompt_ids) + index``.
    """
    channels = segment_channels(completion_ids, tokenizer)
    token_texts = tokenizer.convert_ids_to_tokens(list(completion_ids))
    echo = echo_mask(prompt_ids, completion_ids, echo_n)

    reasoning_positions: list[int] = []
    for channel in channels:
        if channel.is_reasoning:
            reasoning_positions.extend(range(channel.body_start, channel.body_end))
    answer_positions: list[int] = []
    for channel in channels:
        if channel.is_answer:
            answer_positions.extend(range(channel.body_start, channel.body_end))

    substantive = [
        pos
        for pos in reasoning_positions
        if completion_ids[pos] < SPECIAL_MIN
        and not _is_punct(token_texts[pos])
        and not (strip_echo and echo[pos])
    ]
    n_echo = sum(1 for pos in reasoning_positions if echo[pos])

    # Final reasoning sentence: walk back from the last substantive token to the previous
    # sentence terminator, over the reasoning positions only.
    final_sentence: list[int] = []
    if substantive:
        last = substantive[-1]
        collected: list[int] = []
        for pos in reversed([p for p in reasoning_positions if p <= last]):
            text = token_texts[pos].replace("\u0120", "")
            if collected and (text.endswith((".", "!", "?")) or text == "\u010a\u010a"):
                break
            if pos in set(substantive):
                collected.append(pos)
        final_sentence = sorted(collected)

    return TurnReadout(
        substantive=tuple(substantive),
        final_sentence=tuple(final_sentence),
        last_substantive=substantive[-1] if substantive else None,
        # ss.10.7's "final assistant-boundary token for steering diagnostics": the last
        # position of the completion, which is the closing marker of the final channel.
        boundary=len(completion_ids) - 1 if len(completion_ids) else None,
        answer=tuple(answer_positions),
        reasoning_token_count=len(reasoning_positions),
        substantive_token_count=len(substantive),
        echo_token_count=n_echo,
        echo_fraction=(n_echo / len(reasoning_positions)) if reasoning_positions else 0.0,
        channels=tuple(c.recipient for c in channels),
    )
