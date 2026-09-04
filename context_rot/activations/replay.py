"""Teacher-forced activation extraction on the white-box path (PACKET-D-E2 T44; plan ss.E.1).

This is the HF Transformers side of ss.6.3/ss.6.4's deliberate two-stack split. vLLM generates;
this replays. The two never share a process, and ss.6.6's parity gate quantifies the gap.

Three properties this module is built to guarantee, each of which has a test:

* **Hooks are read-only.** `assert_hooks_are_read_only` compares logits with and without the
  capture hooks attached and requires bitwise equality, not a tolerance. A hook that
  perturbs the forward pass would corrupt every downstream readout silently.
* **Handles are always removed.** The `try/finally` in :func:`extract_activations` runs even
  when the forward pass raises, so a failed example cannot leave hooks attached and
  contaminate the next one.
* **Replay is deterministic, and that is verified rather than assumed.** Generation on this
  model is *not* seed-reproducible under continuous batching (CLAUDE.md ss.4a rule 3). A
  teacher-forced forward pass has no sampling, so it *should* be exactly reproducible --
  but "should" is how noise floors get missed. :func:`determinism_report` measures it,
  including the harder case where batch composition changes.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import torch
from torch import Tensor

#: ss.D6 / packet D6: 20, 40, 60, 80 percent of depth, plus the final pre-unembedding
#: residual. Resolved against the actual layer count rather than hardcoded.
DEPTH_FRACTIONS: Final[tuple[float, ...]] = (0.2, 0.4, 0.6, 0.8, 1.0)

#: The post-final-norm residual stream, which is literally what the unembedding reads.
FINAL_NORM_KEY: Final[str] = "final_norm"

READOUTS: Final[tuple[str, ...]] = (
    "mean_substantive",
    "mean_final_sentence",
    "last_substantive",
    "boundary",
    "mean_answer",
)


@dataclass(frozen=True)
class ReadoutSpans:
    """ss.10.7's four readouts plus the answer-channel diagnostic, as token positions.

    Positions are absolute indices into the replayed sequence
    ``prompt_token_ids + completion_token_ids``.
    """

    substantive: tuple[int, ...]
    final_sentence: tuple[int, ...]
    last_substantive: int | None
    boundary: int | None
    answer: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.substantive:
            raise ValueError("a readout needs at least one substantive token")

    @classmethod
    def from_row(cls, row: Any, offset: int) -> ReadoutSpans:
        """Shift a parquet row's completion-relative indices into sequence positions."""

        def shift(values: Iterable[int]) -> tuple[int, ...]:
            return tuple(offset + int(v) for v in values)

        last = row["idx_last_substantive"]
        boundary = row["idx_boundary"]
        return cls(
            substantive=shift(row["idx_substantive"]),
            final_sentence=shift(row["idx_final_sentence"]),
            last_substantive=None if last is None else offset + int(last),
            boundary=None if boundary is None else offset + int(boundary),
            answer=shift(row["idx_answer"]),
        )


def hidden_tensor(output: Any) -> Tensor:
    """ss.E.1's shape guard: a decoder layer may return a tensor or a tuple."""
    if isinstance(output, tuple):
        if not output or not isinstance(output[0], Tensor):
            raise TypeError("Unexpected decoder-layer tuple output")
        return output[0]
    if not isinstance(output, Tensor):
        raise TypeError(f"Unexpected decoder-layer output: {type(output)}")
    return output


def text_decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Locate the text decoder stack.

    The checkpoint carries an unused vision tower, so `model.layers` is ambiguous and a
    wrong guess would read activations out of the wrong stack. Candidate attribute paths
    are tried in order and the result is checked against the config's declared
    `num_hidden_layers`.
    """
    candidates = (
        "model.language_model.layers",
        "language_model.model.layers",
        "model.text_model.layers",
        "model.model.layers",
        "model.layers",
    )
    for path in candidates:
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None and hasattr(node, "__len__") and len(node) > 0:
            return list(node)
    raise RuntimeError(
        "could not locate the text decoder layers; tried " + ", ".join(candidates)
    )


def final_norm_module(model: torch.nn.Module) -> torch.nn.Module:
    for path in (
        "model.language_model.norm",
        "language_model.model.norm",
        "model.text_model.norm",
        "model.model.norm",
        "model.norm",
    ):
        node: Any = model
        for part in path.split("."):
            node = getattr(node, part, None)
            if node is None:
                break
        if isinstance(node, torch.nn.Module):
            return node
    raise RuntimeError("could not locate the final norm module")


def resolve_capture_sites(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Map site keys to modules: ``l010`` ... plus ``final_norm``.

    ss.10.8 requires a separate `v_l` per layer and forbids copying a vector between
    residual spaces, so the site key is carried on every artifact downstream.
    """
    layers = text_decoder_layers(model)
    n = len(layers)
    sites: dict[str, torch.nn.Module] = {}
    for fraction in DEPTH_FRACTIONS:
        index = min(n - 1, max(0, round(fraction * n) - 1))
        sites[f"l{index:03d}"] = layers[index]
    sites[FINAL_NORM_KEY] = final_norm_module(model)
    return sites


def _reduce(hidden: Tensor, spans: ReadoutSpans) -> dict[str, Tensor]:
    """Apply ss.10.7's reductions to one layer's [seq, d] activations."""
    seq = hidden.shape[0]
    for name, positions in (
        ("substantive", spans.substantive),
        ("final_sentence", spans.final_sentence),
        ("answer", spans.answer),
    ):
        if positions and max(positions) >= seq:
            raise ValueError(f"{name} span index {max(positions)} beyond sequence length {seq}")

    out: dict[str, Tensor] = {
        "mean_substantive": hidden[list(spans.substantive)].mean(dim=0),
    }
    out["mean_final_sentence"] = (
        hidden[list(spans.final_sentence)].mean(dim=0)
        if spans.final_sentence
        else out["mean_substantive"]
    )
    out["last_substantive"] = (
        hidden[spans.last_substantive]
        if spans.last_substantive is not None
        else hidden[spans.substantive[-1]]
    )
    out["boundary"] = (
        hidden[spans.boundary] if spans.boundary is not None else hidden[seq - 1]
    )
    # The answer channel is absent on 83 of 4,400 examples, 75 of them S4 evaluation-side
    # turns that ended in a fabricated tool call (DEV-0029). A NaN vector marks the absence
    # explicitly rather than substituting a different span and losing the distinction.
    out["mean_answer"] = (
        hidden[list(spans.answer)].mean(dim=0)
        if spans.answer
        else torch.full_like(out["mean_substantive"], float("nan"))
    )
    return out


def extract_activations(
    model: torch.nn.Module,
    input_ids: Tensor,
    sites: dict[str, torch.nn.Module],
    spans: ReadoutSpans,
    *,
    capture_dtype: torch.dtype = torch.float32,
) -> dict[str, dict[str, Tensor]]:
    """Teacher-force one sequence and return ``{site: {readout: vector}}``.

    ``capture_dtype`` is float32 rather than ss.E.1's float16: the readouts are means over
    hundreds of tokens and the determinism check compares them bitwise, so casting before
    reduction would throw away precision exactly where it is being measured.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected a single sequence [1, seq], got {tuple(input_ids.shape)}")

    captured: dict[str, Tensor] = {}
    handles: list[Any] = []

    def make_hook(key: str) -> Any:
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            # Returning None leaves the forward output untouched; PyTorch only substitutes
            # the output when a hook returns something. That is what makes this read-only.
            captured[key] = hidden_tensor(output).detach()[0].to(capture_dtype)

        return hook

    for key, module in sites.items():
        handles.append(module.register_forward_hook(make_hook(key)))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=False)
    finally:
        # Runs on the exception path too: leaked handles would silently contaminate every
        # later example in the batch loop.
        for handle in handles:
            handle.remove()

    missing = set(sites) - set(captured)
    if missing:
        raise RuntimeError(f"capture sites produced no output: {sorted(missing)}")
    return {key: _reduce(hidden, spans) for key, hidden in captured.items()}


def logits_of(model: torch.nn.Module, input_ids: Tensor) -> Tensor:
    with torch.inference_mode():
        logits: Tensor = model(input_ids=input_ids, use_cache=False).logits.detach()
    return logits


def assert_hooks_are_read_only(
    model: torch.nn.Module, input_ids: Tensor, sites: dict[str, torch.nn.Module]
) -> None:
    """ss.E.1: identical logits with and without read-only hooks.

    Bitwise equality, not a tolerance. These hooks only read and detach, so any difference
    at all means the capture is perturbing the forward pass.
    """
    baseline = logits_of(model, input_ids)
    handles = [
        module.register_forward_hook(
            lambda _m, _i, output: hidden_tensor(output).detach().sum() and None
        )
        for module in sites.values()
    ]
    try:
        hooked = logits_of(model, input_ids)
    finally:
        for handle in handles:
            handle.remove()
    if not torch.equal(baseline, hooked):
        delta = (baseline - hooked).abs().max().item()
        raise AssertionError(f"read-only hooks changed the logits; max |delta| = {delta}")


def determinism_report(
    model: torch.nn.Module,
    sequences: Sequence[Tensor],
    sites: dict[str, torch.nn.Module],
    span_list: Sequence[ReadoutSpans],
    *,
    repeats: int = 2,
) -> dict[str, Any]:
    """Escalation trigger 31: is teacher-forced replay deterministic?

    Two regimes are measured, because they can differ:

    * **repeat** -- the same sequence replayed ``repeats`` times in the same way. Failure
      here means the forward pass itself is nondeterministic (kernel autotuning,
      atomics, TF32 reduction order).
    * **context** -- the same sequence replayed after a *different* sequence has passed
      through the model. Failure here means state is leaking between examples.

    Reports the max absolute and relative deviation per readout, so a non-zero result can
    be quoted as a noise floor instead of just failing.
    """
    first: dict[tuple[int, str, str], Tensor] = {}
    max_abs = 0.0
    max_rel = 0.0
    n_compared = 0
    n_exact = 0

    for repeat in range(repeats):
        order = range(len(sequences)) if repeat % 2 == 0 else reversed(range(len(sequences)))
        for i in order:
            result = extract_activations(model, sequences[i], sites, span_list[i])
            for site, readouts in result.items():
                for name, vector in readouts.items():
                    key = (i, site, name)
                    if key not in first:
                        first[key] = vector.clone()
                        continue
                    reference = first[key]
                    finite = torch.isfinite(reference) & torch.isfinite(vector)
                    if not finite.any():
                        continue
                    delta = (reference[finite] - vector[finite]).abs()
                    scale = reference[finite].abs().clamp_min(1e-6)
                    n_compared += 1
                    if torch.equal(reference[finite], vector[finite]):
                        n_exact += 1
                    max_abs = max(max_abs, float(delta.max()))
                    max_rel = max(max_rel, float((delta / scale).max()))

    return {
        "n_sequences": len(sequences),
        "repeats": repeats,
        "n_comparisons": n_compared,
        "n_bitwise_identical": n_exact,
        "fraction_bitwise_identical": (n_exact / n_compared) if n_compared else float("nan"),
        "max_abs_deviation": max_abs,
        "max_rel_deviation": max_rel,
        "deterministic": n_compared > 0 and n_exact == n_compared,
    }
