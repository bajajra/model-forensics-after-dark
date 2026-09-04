"""Appendix D prompt templates, loaded from the frozen extraction (PACKET-D-E2 T43).

`configs/prompts/e2_direction_templates.json` is produced by
`scripts/e2/extract_templates.py`, which lifts the fenced blocks straight out of
`execution_plan.md` Appendix D. Nothing here retypes protocol text.

Two integrity layers, both derived rather than restated (the DEV-0018 lesson: a constant
restated in one place and derived nowhere will drift):

1. on import, every block is re-hashed and compared to the hash recorded beside it, and
   the whole file is compared to :data:`MANIFEST_SHA256`;
2. `tests/unit/test_e2_templates.py` re-runs the extraction against `execution_plan.md`
   and compares, so a change to the plan fails CI instead of silently diverging.

Placeholder discipline: Appendix D writes slots as ``{NAME}``. Rendering therefore uses
explicit ``str.replace`` on the exact slot strings, not ``str.format`` -- the S1 blocks
contain no braces today, but a future block containing a literal brace would make
``format`` raise or, worse, silently interpolate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

from ..hashing import sha256_text

CONFIG_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "configs" / "prompts" / "e2_direction_templates.json"
)

#: sha256 of `configs/prompts/e2_direction_templates.json` as extracted on 2026-08-23 from
#: `execution_plan.md` Appendix D. Changing Appendix D must change this, visibly, in a diff.
MANIFEST_SHA256: Final[str] = (
    "daa231b0e764fbb81f3a1411919a47e78e8bdfc80934bffd73e3e8abaf97c4f2"
)


def _load() -> dict[str, str]:
    raw = CONFIG_PATH.read_bytes()
    actual = sha256_text(raw.decode("utf-8"))
    if actual != MANIFEST_SHA256:
        raise RuntimeError(
            f"E2 template manifest hash mismatch: expected {MANIFEST_SHA256}, got {actual}. "
            "Appendix D changed, or the extraction did. Do not re-pin without a deviation record."
        )
    payload = json.loads(raw)
    blocks: dict[str, str] = {}
    for key, block in payload["blocks"].items():
        text = block["text"]
        if sha256_text(text) != block["sha256"]:
            raise RuntimeError(f"E2 template block {key!r} does not match its recorded hash")
        blocks[key] = text
    return blocks


TEMPLATES: Final[dict[str, str]] = _load()


def template(key: str) -> str:
    """Return one frozen Appendix D block, raising on an unknown key."""
    try:
        return TEMPLATES[key]
    except KeyError:
        raise KeyError(f"unknown E2 template {key!r}; have {sorted(TEMPLATES)}") from None


def fill(key: str, slots: dict[str, str]) -> str:
    """Substitute ``{SLOT}`` placeholders in a frozen block.

    Every declared slot must be supplied and must occur in the template, and no
    ``{UPPER_CASE}`` slot may survive: an unfilled slot reaching the model is a corrupted
    prompt, and Appendix D.10 rejects examples containing malformed template tokens.
    """
    text = template(key)
    for name, value in slots.items():
        marker = "{" + name + "}"
        if marker not in text:
            raise ValueError(f"template {key!r} has no slot {marker}")
        text = text.replace(marker, value)
    leftover = [
        tok
        for tok in _slot_names(template(key))
        if ("{" + tok + "}") in text
    ]
    if leftover:
        raise ValueError(f"template {key!r} left slots unfilled: {sorted(leftover)}")
    return text


def _slot_names(text: str) -> list[str]:
    out: list[str] = []
    depth_start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            depth_start = i
        elif ch == "}" and depth_start >= 0:
            name = text[depth_start + 1 : i]
            if name and name.replace("_", "").isalnum() and name.upper() == name:
                out.append(name)
            depth_start = -1
    return out


def slots_of(key: str) -> list[str]:
    """The ``{UPPER_CASE}`` slots a frozen block declares, in order of first appearance."""
    seen: dict[str, None] = {}
    for name in _slot_names(template(key)):
        seen.setdefault(name, None)
    return list(seen)
