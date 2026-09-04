"""Stable identifier conventions (`execution_plan.md` §G.2).

Identifiers are stable strings, not row numbers, and **must not encode the observed
outcome** (§G.2 final line, enforced by :func:`assert_no_outcome_fields` and the CI check
"no outcome fields in IDs or assignment inputs", §G.20).

The grammar, verbatim from §G.2::

    model_id          = mdl_<short-name>_<revision8>
    env_id            = env_<name>_<commit8>_<variant>
    prompt_id         = prm_<variant>_<hash8>
    rollout_id        = rol_<study>_<model8>_<env8>_<seed16>
    checkpoint_id     = chk_<rollout_id>_t<turn4>
    source_prefix_id  = pre_<checkpoint_id>_<rejection-type>
    branch_id         = br_<source_prefix_id>_<history><reminder>_<replicate2>
    tail_id           = tail_<branch-or-checkpoint-id>_sb<seedblock3>_k<tail4>
    direction_id      = dir_<construct>_l<layer3>_<readout>_<hash8>
    steering_id       = str_<source-prefix-id>_<history>_<arm>_sb<seedblock3>_k<tail4>
    judge_job_id      = jdg_<role>_<inputhash12>
    analysis_id       = ana_<study>_<endpoint>_<version>

Note on ``source_prefix_id``: the trailing ``<rejection-type>`` is *not* an outcome. It is
the eligibility criterion that defines the prefix as a unit of study (normative vs
instrumental rejection, §2.1), fixed at prefix construction and known before any
continuation is sampled. The rollout's outcome (§2.4) never appears in any ID.
"""

from __future__ import annotations

import re
from typing import Final

# Fields whose value is an observation of what the model did. None may appear in an ID
# or in a randomization-assignment input (§G.2, §G.15 step 6, §G.20).
OUTCOME_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "taskgame",
        "task_game",
        "compliantcomplete",
        "compliant_complete",
        "honeststop",
        "honest_stop",
        "unresolved",
        "invalid",
        "cheat",
        "cheated",
        "gamed",
        "gaming",
        "adopted",
        "adoption",
        "rot",
        "flip",
        "flipped",
        "success",
        "failure",
        "failed",
        "passed",
        "w1",
        "w2",
        "w3",
        "w4",
        "w5",
        "w6",
        "w7",
        "w8",
        "w9",
    }
)

# The rejection type is the eligibility criterion for a source prefix, not an outcome.
REJECTION_TYPES: Final[frozenset[str]] = frozenset({"normative", "instrumental", "mixed", "unclear"})

_SLUG = r"[a-z0-9]+"
_HEX8 = r"[0-9a-f]{8}"

PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "model_id": re.compile(rf"^mdl_{_SLUG}_{_HEX8}$"),
    "env_id": re.compile(rf"^env_{_SLUG}_{_HEX8}_{_SLUG}$"),
    "prompt_id": re.compile(rf"^prm_{_SLUG}_{_HEX8}$"),
    "rollout_id": re.compile(rf"^rol_{_SLUG}_{_HEX8}_{_HEX8}_[0-9a-f]{{16}}$"),
    "checkpoint_id": re.compile(rf"^chk_rol_{_SLUG}_{_HEX8}_{_HEX8}_[0-9a-f]{{16}}_t\d{{4}}$"),
    "source_prefix_id": re.compile(
        rf"^pre_chk_rol_{_SLUG}_{_HEX8}_{_HEX8}_[0-9a-f]{{16}}_t\d{{4}}_"
        rf"(?:{'|'.join(sorted(REJECTION_TYPES))})$"
    ),
    "judge_job_id": re.compile(rf"^jdg_{_SLUG}_[0-9a-f]{{12}}$"),
    "analysis_id": re.compile(rf"^ana_{_SLUG}_{_SLUG}_v\d+$"),
}


class IdError(ValueError):
    """Raised when an identifier violates §G.2."""


def _slug(value: str, field: str) -> str:
    """Lowercase alphanumeric slug; raises rather than silently coercing."""
    cleaned = re.sub(r"[^a-z0-9]", "", value.lower())
    if not cleaned:
        raise IdError(f"{field}={value!r} reduces to an empty slug")
    return cleaned


def _short_hex(value: str, width: int, field: str) -> str:
    """First ``width`` hex characters of ``value``; ``value`` must be lowercase hex."""
    if not re.fullmatch(r"[0-9a-f]+", value):
        raise IdError(f"{field}={value!r} is not lowercase hex")
    if len(value) < width:
        raise IdError(f"{field}={value!r} is shorter than the required {width} hex chars")
    return value[:width]


def assert_no_outcome_fields(identifier: str) -> None:
    """Raise if ``identifier`` contains a token that names an observed outcome (§G.2).

    ``source_prefix_id`` and ``branch_id`` legitimately carry a rejection type; that is the
    prefix's eligibility criterion, not the rollout's outcome, and is exempted.
    """
    parts = {p for p in re.split(r"[_\-]", identifier.lower()) if p}
    if identifier.startswith(("pre_", "br_", "str_")):
        parts -= REJECTION_TYPES
    offending = sorted(parts & OUTCOME_TOKENS)
    if offending:
        raise IdError(f"identifier {identifier!r} encodes outcome token(s): {offending}")


def validate(identifier: str, kind: str) -> str:
    """Validate ``identifier`` against the §G.2 grammar for ``kind``; return it unchanged."""
    pattern = PATTERNS.get(kind)
    if pattern is None:
        raise IdError(f"no validation pattern for id kind {kind!r}")
    if not pattern.match(identifier):
        raise IdError(f"{kind} {identifier!r} does not match {pattern.pattern}")
    assert_no_outcome_fields(identifier)
    return identifier


def model_id(short_name: str, weight_revision: str) -> str:
    """``mdl_<short-name>_<revision8>``."""
    return validate(
        f"mdl_{_slug(short_name, 'short_name')}_{_short_hex(weight_revision, 8, 'weight_revision')}",
        "model_id",
    )


def env_id(name: str, commit: str, variant: str) -> str:
    """``env_<name>_<commit8>_<variant>`` — ``variant`` names the difficulty, e.g. ``e258``."""
    return validate(
        f"env_{_slug(name, 'name')}_{_short_hex(commit, 8, 'commit')}_{_slug(variant, 'variant')}",
        "env_id",
    )


def prompt_id(variant: str, prompt_sha256: str) -> str:
    """``prm_<variant>_<hash8>`` — ``variant`` is ``sourcefaithful`` or ``clarified``."""
    return validate(
        f"prm_{_slug(variant, 'variant')}_{_short_hex(prompt_sha256, 8, 'prompt_sha256')}",
        "prompt_id",
    )


def rollout_id(study: str, model_id_: str, env_id_: str, seed: int) -> str:
    """``rol_<study>_<model8>_<env8>_<seed16>``.

    ``<model8>``/``<env8>`` are the revision/commit segments already embedded in the
    parent IDs, so a rollout ID resolves back to its exact model and environment variant.
    """
    validate(model_id_, "model_id")
    validate(env_id_, "env_id")
    if not 0 <= seed < 1 << 64:
        raise IdError(f"seed {seed} out of range for a 16-hex-digit field")
    model_short = model_id_.split("_")[-1]
    env_short = env_id_.split("_")[2]
    return validate(
        f"rol_{_slug(study, 'study')}_{model_short}_{env_short}_{seed:016x}", "rollout_id"
    )


def checkpoint_id(rollout_id_: str, turn_idx: int) -> str:
    """``chk_<rollout_id>_t<turn4>`` — checkpoints sit at tool-result boundaries (§7.3)."""
    validate(rollout_id_, "rollout_id")
    if not 0 <= turn_idx <= 9999:
        raise IdError(f"turn_idx {turn_idx} does not fit the 4-digit turn field")
    return validate(f"chk_{rollout_id_}_t{turn_idx:04d}", "checkpoint_id")


def source_prefix_id(checkpoint_id_: str, rejection_type: str) -> str:
    """``pre_<checkpoint_id>_<rejection-type>`` (§2.1 rejection taxonomy)."""
    validate(checkpoint_id_, "checkpoint_id")
    if rejection_type not in REJECTION_TYPES:
        raise IdError(f"rejection_type {rejection_type!r} not in {sorted(REJECTION_TYPES)}")
    return validate(f"pre_{checkpoint_id_}_{rejection_type}", "source_prefix_id")


def tail_id(parent_id: str, seed_block: int, tail_index: int) -> str:
    """``tail_<branch-or-checkpoint-id>_sb<seedblock3>_k<tail4>``.

    ``tail_index`` is the *planned* index. §E.3 forbids reusing a planned index for a
    replacement tail: an invalid tail keeps its index and stays in the denominator (§15.8).
    """
    if not 0 <= seed_block <= 999:
        raise IdError(f"seed_block {seed_block} does not fit the 3-digit field")
    if not 0 <= tail_index <= 9999:
        raise IdError(f"tail_index {tail_index} does not fit the 4-digit field")
    identifier = f"tail_{parent_id}_sb{seed_block:03d}_k{tail_index:04d}"
    assert_no_outcome_fields(identifier)
    return identifier


def judge_job_id(role: str, input_sha256: str) -> str:
    """``jdg_<role>_<inputhash12>`` — the input hash is the §6.7 output-cache key."""
    return validate(
        f"jdg_{_slug(role, 'role')}_{_short_hex(input_sha256, 12, 'input_sha256')}", "judge_job_id"
    )


def analysis_id(study: str, endpoint: str, version: int) -> str:
    """``ana_<study>_<endpoint>_<version>``."""
    return validate(
        f"ana_{_slug(study, 'study')}_{_slug(endpoint, 'endpoint')}_v{version:d}", "analysis_id"
    )
