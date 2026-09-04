"""Batched agentic generation with per-row interventions (causal control experiment).

Why batched
-----------
Measured on this model: single-stream decode runs at **7.6 tok/s**, batch-16 at **135.9 tok/s
aggregate** — 18x. At single-stream rates the 672-tail design costs ~49 GPU-hours; batched it
costs a few. That factor is the difference between running the experiment and not.

The batch is built from rows that share a prefix
------------------------------------------------
Every condition for one frozen checkpoint starts from the *same* token prefix and the *same*
workspace tar. So a batch is "one checkpoint, N conditions x samples", which makes the design
and the implementation agree: rows differ only in their intervention and their sampling seed,
which is exactly the contrast the experiment estimates.

Per-row intervention through one hook
-------------------------------------
The write is linear in alpha, so a row's displacement is ``alpha_r * (V @ (sigma * u))``. All
rows can therefore be served by a single hook holding an ``[N, d]`` matrix of per-row
displacements and an ``[N]`` armed mask. No per-row hook juggling, and the control rows
(alpha=0) get an exactly-zero displacement rather than a separate code path that might differ
in some other way.

Cache handling, and a road not taken
------------------------------------
Rows diverge in length once they generate, and their tool outputs differ in size. The cache is
therefore rebuilt once per assistant turn by a **left-padded, chunked prefill**, and decoded
incrementally within the turn. Left padding keeps every row's next-token position on the right
edge so the decode step is uniform; ``position_ids`` are passed explicitly rather than inferred
from the mask, because pad slots sit to the left of real tokens and an inferred position would
be off by the pad width.

An obvious optimisation was tried and abandoned: prefill the shared ~22,000-token prefix once,
then each turn crop the cache back to it and re-prefill only the divergent suffix. It fails on
this model. 39 of its 52 layers use a 2,048-token sliding window and evict history beyond it, so
the prefix cannot be restored by cropping; enabling past-recording to make rollback possible
then produced a sliding-window mask/query length mismatch. That is exactly the class of bug that
yields fluent, plausible, *wrong* generations, so the optimisation was dropped rather than
debugged under time pressure. Throughput was recovered instead by bounding prefill memory with
chunking, which lets the batch grow — the same speed, none of the risk.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..rollout.client import GENERATION_PROMPT, parse_channels, parse_tool_calls
from .hf_client import EOS_IDS, TOK_EOM, TOK_EOT, TOK_MESSAGE, _sample

PAD_ID = 200018


@dataclass
class Row:
    """One tail: its condition, its seed, its workspace, and its accumulating record."""

    row_id: str
    alpha: float
    condition: str
    seed: int
    workspace: Path
    #: [d] displacement delivered at every armed position; zero for controls.
    delta: torch.Tensor
    tokens: list[int] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    applied: int = 0
    in_reasoning: bool = False
    finished: bool = False
    stop_reason: str = "horizon"
    per_turn_signatures: dict[int, list[str]] = field(default_factory=dict)


class BatchedSteeredRunner:
    """Run N interventionally-distinct tails from one frozen prefix, in lockstep."""

    def __init__(
        self,
        model: Any,
        decoder: list[Any],
        tokenizer: Any,
        *,
        device: str,
        layer: int,
        budget: int,
        max_new_tokens: int = 1024,
        prefill_chunk: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> None:
        self.model = model
        self.decoder = decoder
        self.tok = tokenizer
        self.device = device
        self.layer = layer
        self.budget = budget
        self.max_new_tokens = max_new_tokens
        self.sampling = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
        self.gen_prompt = tokenizer.encode(GENERATION_PROMPT, add_special_tokens=False)
        self._deltas: torch.Tensor | None = None
        self._armed: torch.Tensor | None = None
        self._handle: Any = None
        self.prefill_chunk = prefill_chunk

    # -- hook ------------------------------------------------------------------------
    def _install(self, n_rows: int, d_model: int) -> None:
        self._deltas = torch.zeros(n_rows, d_model, device=self.device)
        self._armed = torch.zeros(n_rows, dtype=torch.bool, device=self.device)

        def hook(_m: Any, _i: Any, output: Any) -> Any:
            from ..activations.replay import hidden_tensor

            hidden = hidden_tensor(output)
            if self._armed is None or not bool(self._armed.any()):
                return output
            patched = hidden.clone()
            mask = self._armed.unsqueeze(-1).to(hidden.dtype)
            patched[:, -1, :] = hidden[:, -1, :] + mask * self._deltas.to(hidden.dtype)
            if isinstance(output, tuple):
                return (patched,) + tuple(output[1:])
            return patched

        self._handle = self.decoder[self.layer].register_forward_hook(hook)

    def _remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # -- prefill -----------------------------------------------------------------------
    @torch.inference_mode()
    def _chunked_prefill(self, ids: Any, mask: Any, pos: Any) -> tuple[Any, Any]:
        """Prefill in sequence-length chunks so activation memory does not scale with L.

        A single-shot prefill of B x 22,000 tokens peaked at 136 GB at B=16, which capped the
        batch and left the GPU underused. Feeding the same sequence in chunks bounds transient
        activation memory at ``B x chunk`` instead of ``B x L``, so the batch can grow until the
        KV cache — which is small on this model, 2 KV heads and 39 of 52 layers windowed —
        becomes the limit instead. This is the ordinary incremental-prefill path: the cache is
        only ever appended to, never cropped or rolled back.
        """
        past = None
        logits = None
        total = ids.shape[1]
        for start in range(0, total, self.prefill_chunk):
            stop = min(start + self.prefill_chunk, total)
            out = self.model(
                input_ids=ids[:, start:stop],
                attention_mask=mask[:, :stop],
                position_ids=pos[:, start:stop],
                past_key_values=past, use_cache=True, logits_to_keep=1,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()
        return past, logits

    # -- one assistant turn ----------------------------------------------------------
    @torch.inference_mode()
    def _generate_turn(self, rows: list[Row]) -> None:
        """Generate one assistant turn for every unfinished row, in lockstep."""
        live = [r for r in rows if not r.finished]
        if not live:
            return
        for row in live:
            row.tokens.extend(self.gen_prompt)
            row.in_reasoning = False

        width = max(len(r.tokens) for r in live)
        ids = torch.full((len(live), width), PAD_ID, dtype=torch.long, device=self.device)
        pos = torch.zeros((len(live), width), dtype=torch.long, device=self.device)
        mask = torch.zeros((len(live), width), dtype=torch.long, device=self.device)
        for i, row in enumerate(live):
            k = len(row.tokens)
            # LEFT padding, so every row's next-token position is the last column and the
            # decode step is uniform across rows.
            ids[i, width - k:] = torch.tensor(row.tokens, device=self.device)
            pos[i, width - k:] = torch.arange(0, k, device=self.device)
            mask[i, width - k:] = 1

        index = {r.row_id: i for i, r in enumerate(live)}
        assert self._armed is not None and self._deltas is not None
        armed = torch.zeros(len(live), dtype=torch.bool, device=self.device)
        deltas = torch.stack([r.delta for r in live]).to(self.device)
        self._armed, self._deltas = armed, deltas

        past, logits = self._chunked_prefill(ids, mask, pos)
        next_pos = torch.tensor(
            [len(r.tokens) for r in live], dtype=torch.long, device=self.device
        )

        generators = {
            r.row_id: torch.Generator(device=self.device).manual_seed(r.seed % (2**63))
            for r in live
        }
        active = {r.row_id for r in live}
        generated: dict[str, list[int]] = {r.row_id: [] for r in live}

        for _ in range(self.max_new_tokens):
            if not active:
                break
            next_ids = torch.full((len(live), 1), PAD_ID, dtype=torch.long, device=self.device)
            new_armed = torch.zeros(len(live), dtype=torch.bool, device=self.device)
            for row in live:
                i = index[row.row_id]
                if row.row_id not in active:
                    continue
                token = _sample(logits[i : i + 1], generators[row.row_id], **self.sampling)
                token_id = int(token.item())
                generated[row.row_id].append(token_id)
                if token_id in EOS_IDS:
                    active.discard(row.row_id)
                    continue
                was = row.in_reasoning
                if token_id == TOK_MESSAGE:
                    row.in_reasoning = True
                elif token_id in (TOK_EOM, TOK_EOT):
                    row.in_reasoning = False
                arm = bool(
                    row.in_reasoning and was and row.applied < self.budget
                    and float(row.delta.abs().sum()) > 0
                )
                if arm:
                    row.applied += 1
                new_armed[i] = arm
                next_ids[i, 0] = token_id
            if not active:
                break
            mask = torch.cat(
                [mask, torch.tensor(
                    [[1 if r.row_id in active else 0] for r in live],
                    device=self.device, dtype=torch.long)], dim=1
            )
            self._armed = new_armed
            out = self.model(
                input_ids=next_ids, attention_mask=mask,
                position_ids=next_pos.unsqueeze(1), past_key_values=past,
                use_cache=True, logits_to_keep=1,
            )
            next_pos = next_pos + 1
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()

        self._armed = torch.zeros(len(live), dtype=torch.bool, device=self.device)
        for row in live:
            row.tokens.extend(generated[row.row_id])
            row._raw = self.tok.decode(  # type: ignore[attr-defined]
                [t for t in generated[row.row_id] if t not in EOS_IDS],
                skip_special_tokens=False,
            )
        del out

    # -- public ----------------------------------------------------------------------
    def run(
        self,
        prefix_ids: list[int],
        rows: list[Row],
        *,
        horizon: int,
        dispatch: Any,
        observe: Any,
        d_model: int = 6656,
        max_workers: int = 8,
    ) -> list[Row]:
        """Run every row for ``horizon`` assistant turns.

        ``dispatch(row, call) -> (output, exit_code)`` executes one tool call against that row's
        own workspace. ``observe(row, turn_idx) -> dict`` reads whatever per-turn state the
        analysis needs (tree hash, workaround signatures, error count).
        """
        for row in rows:
            row.tokens = list(prefix_ids)
        self._install(len(rows), d_model)
        try:
            for turn_idx in range(horizon):
                self._generate_turn(rows)
                live = [r for r in rows if not r.finished]
                if not live:
                    break
                # Tool execution is I/O bound (mypy, git) and independent per row, so it runs
                # in parallel; the model is idle during it either way.
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    results = list(pool.map(
                        lambda r: self._finish_turn(r, turn_idx, dispatch), live
                    ))
                for row, record in zip(live, results, strict=True):
                    row.turns.append(record)
                    row.per_turn_signatures[turn_idx] = observe(row, turn_idx)
        finally:
            self._remove()
            torch.cuda.empty_cache()
        return rows

    def _finish_turn(self, row: Row, turn_idx: int, dispatch: Any) -> dict[str, Any]:
        """Parse one row's completion, run its tool, append the result to its stream."""
        raw = getattr(row, "_raw", "")
        channels = parse_channels(GENERATION_PROMPT + raw)
        reasoning = "\n".join(
            c["body"] for c in channels if c["recipient"] == "assistant to=self"
        )
        calls: list[Any] = []
        finals: list[str] = []
        for channel in channels:
            recipient = channel["recipient"]
            if recipient == "assistant to=self":
                continue
            if recipient.startswith("assistant to=") and recipient != "assistant to=user":
                calls.extend(parse_tool_calls(channel["body"]))
            else:
                finals.append(channel["body"])

        record: dict[str, Any] = {
            "turn_idx": turn_idx, "reasoning_text": reasoning,
            "final_text": "\n".join(finals).strip(),
            "tool_name": None, "tool_call_json": None,
            "tool_result_text": None, "tool_exit_code": None,
        }
        if calls:
            call = calls[0]
            output, exit_code = dispatch(row, call)
            record.update({
                "tool_name": call.name,
                "tool_call_json": json.dumps(call.arguments, ensure_ascii=False, sort_keys=True),
                "tool_result_text": output, "tool_exit_code": exit_code,
            })
            row.tokens.extend(self.tok.encode(
                f'<|start|>tool {call.name}<|message|><tool_output name="{call.name}">\n'
                f"{output}\n</tool_output><|eot|>",
                add_special_tokens=False,
            ))
        else:
            # No tool call: the model gave a final answer or produced nothing usable. Either
            # way the tail ends here, which is E1's own stop rule.
            row.finished = True
            row.stop_reason = "final_response" if finals else "degenerate"
        return record
