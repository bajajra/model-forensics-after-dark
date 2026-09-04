"""J-lens and R-lens for Muse-Glimmer-30B, in probe form (E2 addendum; DEV-0033).

References
----------
* Gurnee, Sofroniew, Lindsey et al., *Verbalizable Representations Form a Global Workspace
  in Language Models*, Transformer Circuits, 2026-07-06 — the Jacobian lens.
* Blank, Bhatia & Nanda, *R-lens: Making J-lens More Faithful on Early Layers*,
  Alignment Forum, 2026-08-05 — LRP rules in the backward pass.

What is computed
----------------
The J-lens vector for vocabulary token ``t`` at layer ``l`` is the row ``t`` of ``W_U J_l``,
where ``J_l = E[ d h_final,t' / d h_l,s ]`` averaged over source positions ``s``, later
positions ``t' >= s``, and a corpus of contexts. Equivalently, and this is what makes the
probe form affordable::

    v_t^(l) = E[ d ( W_U[t] . h_final,t' ) / d h_l,s ]

so **one backward pass** from the scalar ``W_U[t] . h_final,t'`` yields that token's lens
vector at *every* layer and *every* source position at once. Cost scales with the number of
target tokens, not with the 202,048-token vocabulary — which matters, because a full lens here
would be 202,048 x 6,656 per layer, about 280 GB across our capture sites.

Three architecture wrinkles this model has and neither paper covers
-------------------------------------------------------------------
1. **Logit softcapping** (``final_logit_softcapping = 20.0``). The lens is defined on
   ``W_U h``, so the target scalar is formed *directly* from the pre-softcap, pre-final-norm
   residual stream rather than from the model's returned logits. Softcapping never enters.
2. **Sliding-window attention**, 2,048 tokens, on 3 of our 5 capture sites. The Jacobian is
   then **structurally zero** for position pairs further apart than the window, so averaging
   over all pairs would silently attenuate the lens at exactly those sites. Pairs are
   restricted to ``0 < t' - s <= window`` and the average is normalised over in-window pairs
   only.
3. **Vocabulary scale**, handled by the probe form above.

R-lens
------
Identical, with LRP stop-gradients installed so relevance rather than raw gradient is
transported:

* **LN-rule** — detach the RMSNorm denominator, making the norm locally linear.
* **Identity-rule** — ``silu(x) = x * sigmoid(x)`` becomes ``x * stopgrad(sigmoid(x))``,
  so the backward pass is a per-element linear map.
* **Half-rule** — a gated MLP's product ``a * b`` splits relevance evenly instead of
  double-counting through both branches.

All three preserve the forward value exactly and only alter gradients, so the fitted lens
describes the same network.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

# --------------------------------------------------------------------------------------
# LRP rules (R-lens). Each preserves the forward value bitwise and changes only gradients.
# --------------------------------------------------------------------------------------


def _lrp_rmsnorm_forward(self: Any, hidden_states: Tensor) -> Tensor:
    """LN-rule for ``MuseGlimmerRMSNorm``: detach the normalisation denominator.

    Mirrors that class's own ``forward`` exactly, including its ``torch.pow(..., -0.5)``
    formulation and its optional ``with_scale``. Writing a single generic RMSNorm rule was a
    real bug: the *other* norm class in this model uses a ``(1 + weight)`` convention with
    weight initialised to zeros, so a shared rule multiplied by ~0 and moved the logits by
    24.1. The forward-value assertion is what caught it.
    """
    x = hidden_states.float()
    mean_squared = x.pow(2).mean(-1, keepdim=True) + self.eps
    out = x * torch.pow(mean_squared, -0.5).detach()
    if self.with_scale:
        out = out * self.weight.float()
    return out.type_as(hidden_states)


def _lrp_centered_rmsnorm_forward(self: Any, x: Tensor) -> Tensor:
    """LN-rule for ``MuseGlimmerTextCenteredRMSNorm``.

    Despite the name it does not subtract the mean; it is RMSNorm with the Gemma-style
    ``(1 + weight)`` scale. Mirrored exactly, denominator detached.
    """
    h = x.float()
    out = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps).detach()
    out = out * (1.0 + self.weight.float())
    return out.type_as(x)


def _half_product(a: Tensor, b: Tensor) -> Tensor:
    """Half-rule: forward is exactly ``a * b``; each branch receives half the relevance."""
    prod = (a * b).detach()
    return prod + 0.5 * (a - a.detach()) * b.detach() + 0.5 * a.detach() * (b - b.detach())


def _lrp_mlp_forward(self: Any, x: Tensor) -> Tensor:
    """Identity-rule on SiLU plus half-rule on the gate/up product."""
    gate = self.gate_proj(x)
    # Identity-rule via a straight-through form: forward is the model's OWN fused
    # ``act_fn(gate)`` bit for bit, backward is the per-element linear map sigmoid(gate).
    # Writing ``gate * sigmoid(gate).detach()`` instead is algebraically identical but not
    # numerically -- the fused SiLU kernel rounds differently, which showed up as a residual
    # forward drift of ~1-2 bf16 ULPs.
    fused = self.act_fn(gate)
    act = fused.detach() + (gate - gate.detach()) * torch.sigmoid(gate).detach()
    return self.down_proj(_half_product(act, self.up_proj(x)))


@contextlib.contextmanager
def lrp_rules(model: torch.nn.Module, *, enabled: bool = True) -> Iterator[None]:
    """Install the R-lens backward rules for the duration of the block."""
    if not enabled:
        yield
        return
    from transformers.models.muse_glimmer import modeling_muse_glimmer as mg

    targets = [
        (mg.MuseGlimmerRMSNorm, "forward", _lrp_rmsnorm_forward),
        (mg.MuseGlimmerTextMLP, "forward", _lrp_mlp_forward),
        (mg.MuseGlimmerTextCenteredRMSNorm, "forward", _lrp_centered_rmsnorm_forward),
    ]
    saved = [(cls, name, getattr(cls, name)) for cls, name, _ in targets]
    try:
        for cls, name, fn in targets:
            setattr(cls, name, fn)
        yield
    finally:
        for cls, name, fn in saved:
            setattr(cls, name, fn)


# --------------------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LensConfig:
    """Everything that defines a fitted lens, recorded with the artifact."""

    layers: tuple[int, ...]
    token_ids: tuple[int, ...]
    n_contexts: int
    positions_per_context: int
    sliding_window: int
    lrp: bool
    context_tokens: int

    def to_record(self) -> dict[str, Any]:
        return {
            "layers": list(self.layers),
            "n_target_tokens": len(self.token_ids),
            "n_contexts": self.n_contexts,
            "positions_per_context": self.positions_per_context,
            "sliding_window": self.sliding_window,
            "lens": "R-lens (LRP)" if self.lrp else "J-lens (raw gradient)",
            "context_tokens": self.context_tokens,
        }


def unembedding_rows(model: torch.nn.Module, token_ids: Sequence[int]) -> Tensor:
    """``W_U[t]`` for each target token, as a [n_tokens, d_model] matrix."""
    head = model.get_output_embeddings()
    if head is None:
        raise RuntimeError("model exposes no output embedding")
    return head.weight[list(token_ids)].detach().float()


def _text_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    from context_rot.activations.replay import text_decoder_layers

    return text_decoder_layers(model)


def enable_lens_grad(model: torch.nn.Module) -> None:
    """Make intermediate activations differentiable without training anything.

    With every parameter frozen and integer inputs, no autograd graph is built at all, so
    there is nothing to differentiate. Turning grad on for the **input embedding only** makes
    the whole downstream graph live while leaving all other parameters frozen. Gradients are
    taken with ``torch.autograd.grad`` against explicit inputs, so no ``.grad`` is ever
    accumulated on a weight.

    Note this must NOT be done by detaching and re-attaching at each capture site: detaching
    at layer 20 would cut the graph above layer 9 and silently return a zero lens there.
    """
    for param in model.parameters():
        param.requires_grad_(False)
    model.get_input_embeddings().weight.requires_grad_(True)


def accumulate_context(
    model: torch.nn.Module,
    input_ids: Tensor,
    w_u_rows: Tensor,
    layers: Sequence[int],
    *,
    positions: Sequence[int],
    sliding_window: int,
    lrp: bool,
    accum: dict[int, Tensor],
    counts: dict[int, Tensor],
) -> None:
    """Add one context's contribution to the running lens sums.

    ``accum[layer]`` has shape [n_tokens, d_model]; ``counts[layer]`` is [n_tokens] and holds
    the number of in-window (source, target) position pairs contributing to each row, so the
    sliding-window zeros never enter the denominator.
    """
    decoder = _text_layers(model)
    captured: dict[int, Tensor] = {}
    handles = []

    def make_hook(idx: int) -> Any:
        def hook(_m: Any, _i: Any, output: Any) -> None:
            from context_rot.activations.replay import hidden_tensor

            h = hidden_tensor(output)
            if h.requires_grad:
                h.retain_grad()
            captured[idx] = h

        return hook

    for idx in layers:
        handles.append(decoder[idx].register_forward_hook(make_hook(idx)))

    try:
        with lrp_rules(model, enabled=lrp), torch.enable_grad():
            out = model(input_ids=input_ids, use_cache=False, output_hidden_states=True)
            # Pre-final-norm residual stream. The J-lens formula composes W_U with the
            # Jacobian into the FINAL RESIDUAL, so the target scalar must be built from the
            # residual stream, not from the model's softcapped logits.
            h_final = out.hidden_states[-1][0].float()

            seq = input_ids.shape[1]
            for t_prime in positions:
                if t_prime >= seq:
                    continue
                for row, w_u in enumerate(w_u_rows):
                    scalar = torch.dot(w_u, h_final[t_prime])
                    grads = torch.autograd.grad(
                        scalar,
                        [captured[i] for i in layers],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    for idx, grad in zip(layers, grads, strict=True):
                        if grad is None:
                            continue
                        g = grad[0].float()
                        lo = max(0, t_prime - sliding_window)
                        window = g[lo : t_prime + 1]
                        accum[idx][row] += window.sum(dim=0)
                        counts[idx][row] += window.shape[0]
    finally:
        for handle in handles:
            handle.remove()


def accumulate_batch(
    model: torch.nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    lengths: Sequence[int],
    w_u_rows: Tensor,
    layers: Sequence[int],
    *,
    positions: Sequence[Sequence[int]],
    sliding_window: int,
    lrp: bool,
    accum: dict[int, Tensor],
    counts: dict[int, Tensor],
) -> None:
    """Batched form of :func:`accumulate_context`.

    Sequences do not interact under causal attention, so for a padded batch the gradient of
    ``sum_b scalar_b`` with respect to ``h[b]`` equals the gradient of ``scalar_b`` alone.
    One backward pass therefore serves the whole batch, which is what makes a 1,000-context
    fit affordable.

    Right padding with an attention mask keeps every real position's index -- and therefore
    its rotary embedding -- identical to the unbatched case. `tests` verify the batched and
    unbatched accumulators agree to floating-point tolerance.
    """
    decoder = _text_layers(model)
    captured: dict[int, Tensor] = {}
    handles = []

    def make_hook(idx: int) -> Any:
        def hook(_m: Any, _i: Any, output: Any) -> None:
            from context_rot.activations.replay import hidden_tensor

            h = hidden_tensor(output)
            if h.requires_grad:
                h.retain_grad()
            captured[idx] = h

        return hook

    for idx in layers:
        handles.append(decoder[idx].register_forward_hook(make_hook(idx)))
    try:
        with lrp_rules(model, enabled=lrp), torch.enable_grad():
            out = model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False, output_hidden_states=True,
            )
            h_final = out.hidden_states[-1].float()
            n_pos = max(len(p) for p in positions)
            for slot in range(n_pos):
                for row, w_u in enumerate(w_u_rows):
                    terms = []
                    live: list[tuple[int, int]] = []
                    for b, pos_list in enumerate(positions):
                        if slot >= len(pos_list):
                            continue
                        t_prime = pos_list[slot]
                        if t_prime >= lengths[b]:
                            continue
                        terms.append(torch.dot(w_u, h_final[b, t_prime]))
                        live.append((b, t_prime))
                    if not terms:
                        continue
                    grads = torch.autograd.grad(
                        torch.stack(terms).sum(),
                        [captured[i] for i in layers],
                        retain_graph=True, allow_unused=True,
                    )
                    for idx, grad in zip(layers, grads, strict=True):
                        if grad is None:
                            continue
                        g = grad.float()
                        for b, t_prime in live:
                            lo = max(0, t_prime - sliding_window)
                            window = g[b, lo : t_prime + 1]
                            accum[idx][row] += window.sum(dim=0)
                            counts[idx][row] += window.shape[0]
    finally:
        for handle in handles:
            handle.remove()


def finalize(accum: dict[int, Tensor], counts: dict[int, Tensor]) -> dict[int, Tensor]:
    """Mean over contributing pairs, then unit-normalise each lens vector."""
    out: dict[int, Tensor] = {}
    for layer, total in accum.items():
        denom = counts[layer].clamp_min(1.0).unsqueeze(1)
        vectors = total / denom
        norms = vectors.norm(dim=1, keepdim=True).clamp_min(1e-12)
        out[layer] = vectors / norms
    return out


def lens_scores(
    hidden: Tensor, vectors: Tensor, *, normalise: bool = True
) -> Tensor:
    """Score a residual-stream vector against each lens vector.

    Cosine rather than raw inner product by default: residual-stream norms grow steeply with
    depth on this model (median 90.6 at l009 to 501.2 at l051), so raw dot products are not
    comparable across sites and CLAUDE.md ss.9 forbids unnormalised cross-layer comparisons.
    """
    h = hidden.float()
    if normalise:
        h = h / h.norm().clamp_min(1e-12)
    return vectors @ h
