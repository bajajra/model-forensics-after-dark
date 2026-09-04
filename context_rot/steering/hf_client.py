"""HF generation client with a live residual-stream intervention (causal control experiment).

Why this exists at all
----------------------
Every measurement in this project so far has been *teacher-forced replay*: the tokens were
already fixed and we only read activations. A causal experiment has to write into the residual
stream **while the model is choosing its next token**, which the vLLM serving path cannot do
(§6.4 is explicit that research hooks do not belong on the production server). So generation
moves to HF Transformers for this study only.

The class deliberately matches ``rollout.client.SubjectClient``'s interface — same
``complete(prompt_token_ids, seed=, tokenizer=) -> Completion`` — so the entire agentic loop,
tool dispatch, workspace handling, checkpointing and grading in ``rollout/loop.py`` are reused
unchanged. Only the token source differs, which is the point: the intervention is the sole
difference between conditions.

Where the write lands
---------------------
The generation prompt ends with ``<|start|>assistant``; the model then emits
`` to=self<|message|>`` and begins its reasoning. So the reasoning body starts at the first
token after ``<|message|>``. The hook is armed when that marker is generated and disarmed at
``<|eom|>`` or when the declared token budget is exhausted, whichever comes first.

The write is applied to the residual stream at the position currently being processed — the
position holding the representation of the most recently generated token. That position's value
is what produces the next token, and its modified value persists in the KV cache, so the
intervention's influence carries forward exactly as a steering intervention should.

Sampling matches FD-08 explicitly on every step (temperature 1.0, top-p 0.95, top-k 64) rather
than relying on any generation-config default, because a silent default change would make two
conditions differ by more than the intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ..rollout.client import GENERATION_PROMPT, Completion, parse_channels, parse_tool_calls

TOK_MESSAGE = 200023
TOK_EOM = 200007
TOK_EOT = 200008
EOS_IDS = (200001, 200008)


@dataclass
class InterventionSpec:
    """What to write, how hard, and for how long. Frozen before any outcome is generated."""

    operator: Any | None
    alpha: float
    #: Number of newly generated REASONING positions to intervene on.
    budget: int = 32
    label: str = "control"

    @property
    def active(self) -> bool:
        return self.operator is not None and self.alpha != 0.0


class HFSubjectClient:
    """Drop-in replacement for ``SubjectClient`` that generates locally, with a hook."""

    def __init__(
        self,
        model: Any,
        decoder_layers: list[Any],
        *,
        device: str,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.decoder = decoder_layers
        self.device = device
        self.sampling = {
            "temperature": temperature, "top_p": top_p, "top_k": top_k,
            "repetition_penalty": 1.0, "max_tokens": max_tokens,
        }
        self.spec = InterventionSpec(operator=None, alpha=0.0)
        #: Reasoning positions intervened on so far, across the whole TAIL (not per turn).
        self.applied = 0
        self.kl_samples: list[float] = []
        self._armed = False
        self._handle = None

    # -- lifecycle -------------------------------------------------------------------
    def set_intervention(self, spec: InterventionSpec) -> None:
        """Install a new intervention and reset its per-tail counters.

        The existing hook MUST be torn down first. It closed over the previous spec's operator
        and alpha, and ``_install_hook`` returns early when a handle already exists — so
        without this, every call after the first silently keeps steering at the *old* strength.
        Caught during strength calibration, where a sweep from alpha=2 to alpha=160 produced
        byte-identical continuations while the measured top-1 flip rate went from 0.00 to 1.00.
        """
        self._remove_hook()
        self.spec = spec
        self.applied = 0
        self.kl_samples = []

    def sampling_config(self) -> dict[str, Any]:
        return dict(self.sampling)

    def close(self) -> None:
        self._remove_hook()

    def __enter__(self) -> HFSubjectClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- hook ------------------------------------------------------------------------
    def _install_hook(self) -> None:
        if self._handle is not None or not self.spec.active:
            return
        operator = self.spec.operator
        alpha = self.spec.alpha

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            if not self._armed:
                return output
            from ..activations.replay import hidden_tensor

            hidden = hidden_tensor(output)
            # Only the LAST position is ever a newly generated position: during decode the
            # block is length 1, and during prefill nothing has been generated yet, so the
            # hook is disarmed. Writing to every position of a prefill block would silently
            # steer the entire conversation history.
            patched = hidden.clone()
            patched[:, -1, :] = operator.apply(hidden[:, -1, :], alpha)
            if isinstance(output, tuple):
                return (patched,) + tuple(output[1:])
            return patched

        self._handle = self.decoder[operator.layer].register_forward_hook(hook)

    def _remove_hook(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # -- generation ------------------------------------------------------------------
    @torch.inference_mode()
    def complete(
        self, prompt_token_ids: list[int], *, seed: int, tokenizer: Any
    ) -> Completion:
        """Sample one assistant turn, intervening on its first reasoning positions."""
        self._install_hook()
        generator = torch.Generator(device=self.device).manual_seed(int(seed) % (2**63))
        ids = torch.tensor([prompt_token_ids], device=self.device)

        self._armed = False
        past = None
        generated: list[int] = []
        finish = "length"
        in_reasoning = False

        for step in range(int(self.sampling["max_tokens"])):
            step_ids = ids if past is None else ids[:, -1:]
            out = self.model(
                input_ids=step_ids, past_key_values=past, use_cache=True, logits_to_keep=1
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :].float()

            token = _sample(logits, generator, **{
                k: self.sampling[k] for k in ("temperature", "top_p", "top_k")
            })
            token_id = int(token.item())
            generated.append(token_id)

            if token_id in EOS_IDS:
                finish = "stop"
                break
            # Arm on the reasoning-channel marker, disarm at end-of-message or budget.
            #
            # The marker token itself is NOT an intervened position. Arming on it would write
            # to the position holding `<|message|>`, so a declared budget of 32 reasoning
            # positions would deliver 31 plus the header — an off-by-one that no downstream
            # check could catch, because the count would still read 32.
            was_reasoning = in_reasoning
            if token_id == TOK_MESSAGE:
                in_reasoning = True
            elif token_id in (TOK_EOM, TOK_EOT):
                in_reasoning = False
            self._armed = bool(
                self.spec.active
                and in_reasoning
                and was_reasoning
                and self.applied < self.spec.budget
            )
            if self._armed:
                self.applied += 1
            ids = torch.cat([ids, token.view(1, 1)], dim=1)

        self._armed = False
        raw = tokenizer.decode(
            [t for t in generated if t not in EOS_IDS], skip_special_tokens=False
        )
        channels = parse_channels(GENERATION_PROMPT + raw)
        reasoning = "\n".join(
            c["body"] for c in channels if c["recipient"] == "assistant to=self"
        )
        tool_calls: list[Any] = []
        finals: list[str] = []
        for channel in channels:
            recipient = channel["recipient"]
            if recipient == "assistant to=self":
                continue
            if recipient.startswith("assistant to=") and recipient != "assistant to=user":
                tool_calls.extend(parse_tool_calls(channel["body"]))
            else:
                finals.append(channel["body"])
        return Completion(
            token_ids=tuple(generated),
            raw_text=raw,
            reasoning_text=reasoning,
            final_text="\n".join(finals).strip(),
            tool_calls=tuple(tool_calls),
            finish_reason=finish,
            prompt_tokens=len(prompt_token_ids),
            completion_tokens=len(generated),
            channels=tuple(channels),
        )


def _sample(
    logits: torch.Tensor, generator: torch.Generator, *,
    temperature: float, top_p: float, top_k: int,
) -> torch.Tensor:
    """Temperature / top-k / top-p sampling, applied in that order (FD-08's policy)."""
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(ordered, dim=-1) > top_p
        ordered = ordered.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, index, ordered)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)[0]


def _kl(clean_logits: torch.Tensor, steered_logits: torch.Tensor) -> float:
    p = torch.log_softmax(clean_logits, dim=-1)
    q = torch.log_softmax(steered_logits, dim=-1)
    return float((p.exp() * (p - q)).sum(dim=-1).mean().item())
