"""Subject-model generation for E2 contrastive pairs (PACKET-D-E2 T43; plan D.1 step 5).

Differs from `rollout/client.py` in one respect that matters for T44: it asks vLLM for
``return_token_ids`` and stores the **sampled** token IDs rather than re-encoding the
decoded text. Re-encoding is not guaranteed to round-trip, and teacher-forced replay reads
activations at token positions -- a single re-tokenisation difference silently shifts every
readout span. The server echoes ``prompt_token_ids`` too, so both halves are exact.

Determinism, stated honestly: E1 measured that a fixed seed does **not** reproduce output
under vLLM's continuous batching (CLAUDE.md ss.4a rule 3). Seeds are still derived per
ss.G.14 so scheduling cannot change them, but the generated corpus is reproducible only as a
hashed artifact, not as a rerunnable computation.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from ..hashing import sha256_text, sha256_token_ids
from ..rollout.client import parse_channels
from ..seeds import SeedKey, derive_seed

GENERATION_PROMPT = "<|start|>assistant"
TOK_EOM = 200007
TOK_EOT = 200008

#: E1's convention (`scripts/run_rollouts.py:242`). See :func:`side_seed`.
SEED_MODULUS = 2**31

#: FD-08: every sampling parameter is sent explicitly on every request.
FROZEN_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
    "repetition_penalty": 1.0,
}


@dataclass(frozen=True, slots=True)
class Generation:
    """One completed generation, with exact token IDs on both sides."""

    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    text: str
    reasoning_text: str
    final_text: str
    channels: tuple[dict[str, str], ...]
    finish_reason: str
    seed: int

    def to_record(self) -> dict[str, Any]:
        return {
            "prompt_token_ids_sha256": sha256_token_ids(self.prompt_token_ids),
            "prompt_token_count": len(self.prompt_token_ids),
            "completion_token_ids": list(self.completion_token_ids),
            "completion_token_count": len(self.completion_token_ids),
            "text": self.text,
            "text_sha256": sha256_text(self.text),
            "reasoning_text": self.reasoning_text,
            "final_text": self.final_text,
            "channel_recipients": [c["recipient"] for c in self.channels],
            "finish_reason": self.finish_reason,
            "seed": self.seed,
        }


def render_prompt_ids(tokenizer: Any, system_prompt: str, user_content: str) -> list[int]:
    """Render a one-turn conversation to exact token IDs.

    The system message is supplied explicitly rather than defaulted: the packaged chat
    template interpolates the current date when no system turn is given, which would make
    every prompt hash depend on the day the dataset was built.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    # `apply_chat_template` returns a plain list, a nested list, or a `BatchEncoding` --
    # which is a `UserDict`, *not* a `dict`, so an `isinstance(..., dict)` test silently
    # falls through and `list()` yields the key names. That failure reaches the server as a
    # two-token text prompt; the prompt-echo check in `generate` is what surfaced it.
    if hasattr(rendered, "keys") and "input_ids" in rendered:
        rendered = rendered["input_ids"]
    ids = list(rendered)
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise RuntimeError(f"expected one rendered sequence, got {len(ids)}")
        ids = list(ids[0])
    if not ids or not all(isinstance(i, int) for i in ids):
        raise RuntimeError(f"chat template did not yield token IDs: {type(rendered)!r}")
    return ids


def render_followup_ids(
    tokenizer: Any,
    prior_prompt_ids: Sequence[int],
    prior_completion_ids: Sequence[int],
    user_content: str,
) -> tuple[list[int], bool]:
    """Extend a conversation by one user turn, reusing the prior turn's **exact** token IDs.

    Re-rendering the whole conversation through ``apply_chat_template`` would run the
    previous assistant turn back through the tokenizer, and a decode/encode round trip is
    not guaranteed to reproduce the sampled IDs (ss.6.5). Concatenating the recorded IDs and
    appending only the new markers keeps every earlier position byte-identical to what the
    model actually produced.

    Returns the token IDs and whether a closing ``<|eot|>`` had to be supplied, which
    happens only when the prior turn hit its token ceiling.
    """
    ids = list(prior_prompt_ids) + list(prior_completion_ids)
    patched = False
    if not ids or ids[-1] not in (TOK_EOM, TOK_EOT):
        ids.append(TOK_EOT)
        patched = True
    ids += tokenizer.encode(
        f"<|start|>user<|message|>{user_content}<|eot|><|start|>assistant",
        add_special_tokens=False,
    )
    return ids, patched


class GenerationClient:
    """Thread-safe completions client for E2 dataset construction."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 900.0,
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout, limits=httpx.Limits(max_connections=128, max_keepalive_connections=64)
        )
        self._lock = threading.Lock()
        self.infra_retries = 0

    def close(self) -> None:
        self._client.close()

    def generate(
        self, prompt_token_ids: Sequence[int], *, seed: int, max_tokens: int
    ) -> Generation:
        payload = {
            "model": self.model,
            "prompt": list(prompt_token_ids),
            "seed": seed,
            "max_tokens": max_tokens,
            "skip_special_tokens": False,
            "return_token_ids": True,
            **FROZEN_SAMPLING,
        }
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.post(f"{self.base_url}/v1/completions", json=payload)
                if response.status_code == 400:
                    # A 400 is the server rejecting the request, not a transport fault.
                    # Retrying it four times hides the reason; surface it immediately.
                    raise RuntimeError(
                        f"server rejected the request (400): {response.text[:500]} "
                        f"[prompt_tokens={len(payload['prompt'])}, "
                        f"max_tokens={payload['max_tokens']}]"
                    )
                response.raise_for_status()
                break
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                # ss.G.16: retries are for infrastructure only. A transport error before any
                # output is infrastructure; a returned-but-unusable generation is an outcome
                # and is handled downstream by the D.10 quality filters, never re-rolled here.
                last = exc
                with self._lock:
                    self.infra_retries += 1
                if attempt == self.max_retries - 1:
                    raise
        else:  # pragma: no cover - loop always breaks or raises
            raise RuntimeError(f"generation failed: {last}")

        data = response.json()
        choice = data["choices"][0]
        text = choice["text"]
        completion_ids = choice.get("token_ids")
        if completion_ids is None:
            raise RuntimeError(
                "server did not return token_ids; activation replay needs exact IDs. "
                "Check that the pinned vLLM build supports return_token_ids."
            )
        echoed = choice.get("prompt_token_ids")
        if echoed is not None and list(echoed) != list(prompt_token_ids):
            raise RuntimeError(
                "server echoed different prompt_token_ids than were sent "
                f"({len(echoed)} vs {len(prompt_token_ids)}); refusing to record the pair"
            )

        channels = parse_channels(GENERATION_PROMPT + text)
        reasoning = "\n".join(
            c["body"] for c in channels if c["recipient"] == "assistant to=self"
        )
        finals = [
            c["body"]
            for c in channels
            if c["recipient"] != "assistant to=self"
            and not (
                c["recipient"].startswith("assistant to=")
                and c["recipient"] != "assistant to=user"
            )
        ]
        return Generation(
            prompt_token_ids=tuple(prompt_token_ids),
            completion_token_ids=tuple(completion_ids),
            text=text,
            reasoning_text=reasoning,
            final_text="\n".join(finals).strip(),
            channels=tuple(channels),
            finish_reason=choice.get("finish_reason", "unknown"),
            seed=seed,
        )


def side_seed(master_seed: str, pair_id: str, side: str, stage: int, attempt: int) -> int:
    """ss.G.14 seed derivation for one generation.

    ``pair_id`` occupies the source-prefix slot because it is this job's stable identity;
    ``stage`` and ``attempt`` occupy the seed-block and tail-index slots so that a two-stage
    construct and a regenerated example never collide with each other.

    The ``% SEED_MODULUS`` reduction matches `scripts/run_rollouts.py:242`, which E1 used:
    ``derive_seed`` returns a full 64 unsigned bits and the server validates ``seed`` as a
    signed 64-bit integer, so roughly half of all derived seeds are rejected outright. Same
    modulus as E1 so the two studies draw from the same seed space.
    """
    return derive_seed(
        master_seed,
        SeedKey(
            study_id="E2",
            source_prefix_id=pair_id,
            assignment=side,
            seed_block=stage,
            tail_index=attempt,
        ),
    ) % SEED_MODULUS


def write_jsonl(path: Any, records: list[dict[str, Any]]) -> None:
    """Append records to an append-only JSONL shard (methodology rule 6)."""
    with open(path, "a", encoding="utf-8") as handle:
        handle.writelines(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
