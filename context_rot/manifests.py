"""Machine-readable manifests: model (§G.3), prompt (§A.15), provenance (§G.17).

Everything that can drift between runs is pinned here: weight revision, tokenizer revision,
chat-template hash, backend commit, container digest, quantization, reasoning effort
(CLAUDE.md §9, `execution_plan.md` §18.22). :func:`assert_pinned` implements the §G.21
release-verifier rule that a manifest must never carry a mutable ``latest`` image or a
mutable model revision — PACKET-A-E0 §A6.1 C1/C2 and §A6.3 J1 are exactly this check applied
to the subject model and the judge.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .hashing import sha256_json

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")

#: Tags that name a moving target rather than an immutable artifact (§G.21).
MUTABLE_TAGS = ("latest", "main", "master", "head", "dev", "nightly", "stable")


class ManifestError(ValueError):
    """Raised when a manifest is unpinned, incomplete, or internally inconsistent."""


def assert_pinned(container_digest: str, weight_revision: str, tokenizer_revision: str) -> None:
    """Fail on a mutable image reference or a mutable model revision (§G.21).

    A container must be referenced by ``sha256:<64 hex>`` — a tag, even a specific-looking
    one, can be repointed. A weight or tokenizer revision must be a full 40-hex commit;
    a branch name is a moving target.
    """
    if not _DIGEST.match(container_digest):
        raise ManifestError(
            f"container_digest {container_digest!r} is not an immutable sha256 digest; "
            "§G.21 fails the release on a mutable image reference"
        )
    for label, revision in (
        ("weight_revision", weight_revision),
        ("tokenizer_revision", tokenizer_revision),
    ):
        if not _REVISION.match(revision):
            raise ManifestError(
                f"{label}={revision!r} is not a 40-hex commit; "
                f"{'a mutable branch name' if revision.lower() in MUTABLE_TAGS else 'an unpinned revision'} "
                "makes the served policy revision-dependent (§18.22)"
            )


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """Subject-model manifest (§G.3 schema, field order preserved)."""

    model_id: str
    hub_id: str
    weight_revision: str
    weight_format: str
    tokenizer_revision: str
    config_sha256: str
    chat_template_sha256: str
    reasoning_mode: str
    transformers_commit: str
    fast_backend: str
    fast_backend_commit: str
    whitebox_backend_commit: str
    container_digest: str
    notes: str = ""

    #: Sampling policy, pinned explicitly rather than inherited from server defaults
    #: (§5.6, PACKET-A-E0 FD-08 / §A6.1 C3). ``generation_config_sha256`` records the hash of
    #: the checkpoint's own ``generation_config.json`` so a silent upstream change fails loudly.
    sampling: dict[str, Any] = field(default_factory=dict)
    generation_config_sha256: str = ""
    speculative_config: dict[str, Any] | None = None

    def validate(self) -> None:
        assert_pinned(self.container_digest, self.weight_revision, self.tokenizer_revision)
        required = ("temperature", "top_p", "top_k")
        missing = [k for k in required if k not in self.sampling]
        if missing:
            raise ManifestError(
                f"sampling policy is missing {missing}; §5.6 requires temperature, top-p, top-k "
                "and any repetition penalties to be pinned, and FD-08 requires them passed "
                "explicitly on every request"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JudgeManifest:
    """Judge-model manifest (§G.3 final paragraph, §A6.3 required fields)."""

    model_id: str
    hub_id: str
    weight_revision: str
    weight_format: str
    tokenizer_revision: str
    tokenizer_mode: str
    chat_template_sha256: str
    container_digest: str
    remote_code_commit: str
    tensor_parallel_size: int
    expert_parallel: bool
    kv_cache_dtype: str
    flashinfer_autotune: bool
    speculative_config: dict[str, Any] | None
    parser_config: dict[str, Any]
    reasoning_effort_by_role: dict[str, str]
    max_model_len: int
    notes: str = ""

    def validate(self) -> None:
        assert_pinned(self.container_digest, self.weight_revision, self.tokenizer_revision)
        if self.flashinfer_autotune:
            raise ManifestError(
                "flashinfer autotune must stay OFF for the whole project: kernel autotuning "
                "makes numerics vary run to run, and the §6.7 prompt-hash output cache is only "
                "sound if a prompt deterministically yields the same judgment (§A6.3)"
            )
        if not _REVISION.match(self.remote_code_commit):
            raise ManifestError(
                "remote_code_commit must be a pinned 40-hex commit: --trust-remote-code "
                "otherwise executes whatever the hub currently serves (§A6.3 J1)"
            )
        if not self.reasoning_effort_by_role:
            raise ManifestError(
                "reasoning_effort_by_role is empty; §6.7 requires role-specific reasoning effort"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptManifest:
    """Prompt manifest (§A.15 schema, verbatim field set).

    ``subject_token_counts`` exists to enforce §A.1 rule 5: an intervention message and its
    control must have approximately the same subject-model token length (CLAUDE.md §10
    tightens "approximately" to within 5%).
    """

    manifest_version: str
    environment_commit: str
    system_prompt_sha256: str
    user_prompt_sha256: str
    rendered_prompt_sha256: str
    tool_schema_sha256: str
    chat_template_sha256: str
    reasoning_mode: str
    constraint_reminder_sha256: str = ""
    neutral_reminder_sha256: str = ""
    subject_token_counts: dict[str, int] = field(
        default_factory=lambda: {"constraint_reminder": 0, "neutral_reminder": 0}
    )

    def validate(self, *, require_reminders: bool = False) -> None:
        """Validate hashes are present. ``require_reminders`` applies from E3 onward."""
        for name in (
            "system_prompt_sha256",
            "user_prompt_sha256",
            "rendered_prompt_sha256",
            "tool_schema_sha256",
            "chat_template_sha256",
        ):
            value = getattr(self, name)
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ManifestError(f"{name}={value!r} is not a sha256 hex digest")
        if require_reminders:
            self.assert_reminder_token_match()

    def assert_reminder_token_match(self, tolerance: float = 0.05) -> None:
        """Reminder and control must be token-matched within 5% (CLAUDE.md §10, §A.1 rule 5)."""
        counts = self.subject_token_counts
        constraint = counts.get("constraint_reminder", 0)
        neutral = counts.get("neutral_reminder", 0)
        if constraint <= 0 or neutral <= 0:
            raise ManifestError("reminder token counts must both be positive before use")
        relative = abs(constraint - neutral) / max(constraint, neutral)
        if relative > tolerance:
            raise ManifestError(
                f"constraint reminder ({constraint} tok) and neutral control ({neutral} tok) "
                f"differ by {relative:.1%}, above the {tolerance:.0%} tolerance (§A.1 rule 5)"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Provenance for one report artifact (§G.17)."""

    artifact: str
    artifact_sha256: str
    analysis_id: str
    analysis_code_commit: str
    analysis_environment_digest: str
    input_table_sha256: str
    randomization_manifest_sha256: str
    generated_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json(path: str | Path, payload: Any) -> str:
    """Write ``payload`` as indented JSON and return its canonical content hash."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return sha256_json(payload)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())
