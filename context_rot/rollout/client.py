"""Subject-model client: client-side rendering, exact token IDs (§6.5, §7.3, §A6.2).

Samples from `/v1/completions` with the prompt supplied as **token IDs**, not text. §A4.1 item 5
is explicit that this is the point: a chat-completions endpoint renders server-side and hides
both the exact prompt and the exact token IDs, and §7.3 requires storing the conversation's token
IDs at every checkpoint while §6.5 requires replaying *exact* token IDs. Rendering here also
avoids a failure this project already hit — passing a rendered string back through the server's
tokenizer produced a **double BOS**.

Muse-Glimmer's ATEM template has no JSON tool calls and no `<think>` tags; every turn is a
sequence of channel-scoped messages. Parsing is therefore done here rather than by the server's
`--tool-call-parser`, which keeps the raw token IDs authoritative.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

#: `<|start|>{recipient}<|message|>{body}` closed by `<|eom|>` (turn continues) or `<|eot|>`.
_CHANNEL = re.compile(
    r"<\|start\|>(?P<recipient>[^<]*?)<\|message\|>(?P<body>.*?)(?=<\|eom\|>|<\|eot\|>|\Z)",
    re.DOTALL,
)
_ATEM_INVOKE = re.compile(r'<atem:invoke name="(?P<name>[^"]+)">(?P<params>.*?)</atem:invoke>', re.DOTALL)
GENERATION_PROMPT = "<|start|>assistant"

_ATEM_PARAM = re.compile(r'<atem:parameter name="(?P<name>[^"]+)">(?P<value>.*?)</atem:parameter>', re.DOTALL)


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str


@dataclass(frozen=True, slots=True)
class Completion:
    """One assistant turn, with its channels separated (§8.2's first condition)."""

    token_ids: tuple[int, ...]
    raw_text: str
    reasoning_text: str
    final_text: str
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    channels: tuple[dict[str, str], ...] = field(default_factory=tuple)

    @property
    def has_tool_call(self) -> bool:
        return bool(self.tool_calls)


def parse_channels(raw: str) -> list[dict[str, str]]:
    """Split a generation into ATEM channels, in order."""
    out: list[dict[str, str]] = []
    for match in _CHANNEL.finditer(raw):
        out.append(
            {"recipient": match.group("recipient").strip(), "body": match.group("body")}
        )
    return out


def parse_tool_calls(body: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for invoke in _ATEM_INVOKE.finditer(body):
        args = {
            p.group("name"): p.group("value")
            for p in _ATEM_PARAM.finditer(invoke.group("params"))
        }
        calls.append(ToolCall(invoke.group("name"), args, invoke.group(0)))
    return calls


class SubjectClient:
    """Thin, explicit client. Every sampling parameter is sent on every request (FD-08)."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float = 1.0,
        max_tokens: int = 4096,
        timeout: float = 1800.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sampling = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
        }
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def sampling_config(self) -> dict[str, Any]:
        """The exact policy, for the model manifest (§G.3) and the §A.1 hash."""
        return dict(self.sampling)

    def complete(self, prompt_token_ids: list[int], *, seed: int, tokenizer: Any) -> Completion:
        """Sample one assistant turn from an exact token-ID prefix."""
        payload = {
            "model": self.model,
            "prompt": prompt_token_ids,
            "seed": seed,
            # The ATEM markers ARE the channel structure. Stripping special tokens collapses
            # reasoning and content into one blob — the exact failure vLLM PR #51655 calls out.
            "skip_special_tokens": False,
            **self.sampling,
        }
        response = self._client.post(f"{self.base_url}/v1/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        raw = choice["text"]

        token_ids = tuple(tokenizer.encode(raw, add_special_tokens=False))
        # The generation prompt ends with `<|start|>assistant`, so the completion begins INSIDE
        # the first channel (` to=self<|message|>...`). Re-attach the opener before parsing, or
        # the reasoning channel is silently dropped and every turn looks like it had no
        # reasoning at all — which would quietly falsify FD-09's retained-reasoning accounting.
        channels = parse_channels(GENERATION_PROMPT + raw)
        reasoning = "\n".join(c["body"] for c in channels if c["recipient"] == "assistant to=self")
        tool_calls: list[ToolCall] = []
        finals: list[str] = []
        for channel in channels:
            recipient = channel["recipient"]
            if recipient == "assistant to=self":
                continue
            if recipient.startswith("assistant to=") and recipient != "assistant to=user":
                tool_calls.extend(parse_tool_calls(channel["body"]))
            else:
                finals.append(channel["body"])

        usage = data.get("usage", {})
        return Completion(
            token_ids=token_ids,
            raw_text=raw,
            reasoning_text=reasoning,
            final_text="\n".join(finals).strip(),
            tool_calls=tuple(tool_calls),
            finish_reason=choice.get("finish_reason", "unknown"),
            prompt_tokens=usage.get("prompt_tokens", len(prompt_token_ids)),
            completion_tokens=usage.get("completion_tokens", len(token_ids)),
            channels=tuple(channels),
        )


def render_tool_result(tokenizer: Any, tool_name: str, output: str) -> list[int]:
    """Render a tool result exactly as the chat template would, and return its token IDs."""
    text = (
        f'<|start|>tool {tool_name}<|message|><tool_output name="{tool_name}">\n'
        f"{output}\n</tool_output><|eot|>"
    )
    return tokenizer.encode(text, add_special_tokens=False)


def render_user_message(tokenizer: Any, content: str) -> list[int]:
    return tokenizer.encode(
        f"<|start|>user<|message|>{content}<|eot|>", add_special_tokens=False
    )


def generation_prompt_ids(tokenizer: Any) -> list[int]:
    return tokenizer.encode("<|start|>assistant", add_special_tokens=False)


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
