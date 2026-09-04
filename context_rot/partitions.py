"""Partition manifest and partition discipline (`execution_plan.md` §4.3, §G.4).

Every source rollout, checkpoint, contrastive example, and source prefix receives a frozen
partition before confirmatory analysis. The invariant this module enforces is §G.4's:

    A parent and its descendants cannot cross into a partition that would leak outcomes.

Concretely (§G.4): tails from one source rollout remain in the source rollout's partition;
contrastive pair sides remain together; source prefixes used for alpha calibration never
appear in confirmatory steering; source prefixes used to debug branch construction are not
reused in confirmatory branch outcomes.

All E0 data is ``exploration`` (PACKET-A-E0 FD-11). The practical consequence, spelled out
in §A13.3 of the packet: **a prefix spent on the E0.5 pilot is burned for confirmatory E3/E4**.
:func:`burned_prefixes` reports that accounting so it is visible rather than discovered late.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Final, Literal

Partition = Literal["exploration", "discovery", "calibration", "confirmatory", "replication"]

PARTITIONS: Final[tuple[Partition, ...]] = (
    "exploration",
    "discovery",
    "calibration",
    "confirmatory",
    "replication",
)

#: Partitions whose members may never later appear in confirmatory analysis (§4.3, §G.4).
#: Anything selected using outcome-bearing data is selected *outside* the confirmatory
#: partition, so exposure to any of these burns the object for confirmatory use.
OUTCOME_EXPOSED: Final[frozenset[str]] = frozenset({"exploration", "discovery", "calibration"})

OBJECT_TYPES: Final[frozenset[str]] = frozenset(
    {
        "source_rollout",
        "checkpoint",
        "source_prefix",
        "contrastive_example",
        "branch",
        "tail",
    }
)

PARTITION_RULE_VERSION: Final[str] = "2.2-e0"


class PartitionError(ValueError):
    """Raised when a partition assignment violates §4.3 or §G.4."""


@dataclass(frozen=True, slots=True)
class PartitionRecord:
    """One row of ``manifests/partitions.parquet`` (§G.4 schema, field order preserved)."""

    object_id: str
    object_type: str
    parent_id: str | None
    partition: Partition
    partition_rule_version: str
    partition_hash_input: str
    assigned_at: str
    superseded_by: str | None = None

    def __post_init__(self) -> None:
        if self.object_type not in OBJECT_TYPES:
            raise PartitionError(f"object_type {self.object_type!r} not in {sorted(OBJECT_TYPES)}")
        if self.partition not in PARTITIONS:
            raise PartitionError(f"partition {self.partition!r} not in {list(PARTITIONS)}")
        if not self.object_id:
            raise PartitionError("object_id must be non-empty")
        if not self.assigned_at:
            raise PartitionError("assigned_at must be non-empty")

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PartitionManifest:
    """In-memory partition manifest with the §G.4 inheritance and overlap invariants."""

    records: dict[str, PartitionRecord] = field(default_factory=dict)

    def assign(
        self,
        object_id: str,
        object_type: str,
        partition: Partition,
        *,
        parent_id: str | None = None,
        partition_hash_input: str = "",
        assigned_at: str | None = None,
    ) -> PartitionRecord:
        """Assign ``object_id`` to ``partition``, inheriting and validating against its parent.

        A descendant inherits its parent's partition (CLAUDE.md §7 rule 2: "Descendants
        inherit their parent's partition"). Passing a ``partition`` that differs from the
        parent's is an error, not a silent override.
        """
        if object_id in self.records and self.records[object_id].superseded_by is None:
            existing = self.records[object_id]
            raise PartitionError(
                f"{object_id} already assigned to {existing.partition!r}; partitions are frozen "
                "before confirmatory analysis and are corrected by supersession, not overwrite"
            )
        if parent_id is not None:
            parent = self.records.get(parent_id)
            if parent is None:
                raise PartitionError(f"parent {parent_id!r} of {object_id!r} is not in the manifest")
            if parent.partition != partition:
                raise PartitionError(
                    f"{object_id!r} would enter partition {partition!r} but its parent "
                    f"{parent_id!r} is {parent.partition!r}; descendants inherit (§G.4)"
                )
        record = PartitionRecord(
            object_id=object_id,
            object_type=object_type,
            parent_id=parent_id,
            partition=partition,
            partition_rule_version=PARTITION_RULE_VERSION,
            partition_hash_input=partition_hash_input,
            assigned_at=assigned_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self.records[object_id] = record
        return record

    def partition_of(self, object_id: str) -> Partition:
        record = self.records.get(object_id)
        if record is None:
            raise PartitionError(f"{object_id!r} has no partition assignment")
        return record.partition

    def members(self, partition: Partition) -> set[str]:
        return {
            oid
            for oid, rec in self.records.items()
            if rec.partition == partition and rec.superseded_by is None
        }

    def burned_prefixes(self) -> set[str]:
        """Source prefixes that may never appear in confirmatory E3/E4 (§4.3, §G.4, FD-11).

        A prefix is burned once it has been used anywhere outcomes were inspected — which,
        for E0, is every prefix that entered the E0.5 resampling pilot.
        """
        return {
            oid
            for oid, rec in self.records.items()
            if rec.object_type == "source_prefix"
            and rec.partition in OUTCOME_EXPOSED
            and rec.superseded_by is None
        }

    def as_rows(self) -> list[dict[str, object]]:
        return [rec.as_row() for rec in self.records.values()]


def assert_no_partition_overlap(manifest: PartitionManifest) -> None:
    """CI check "no overlap across partitions" (§G.20).

    An object may hold exactly one live partition assignment. Superseded records are
    excluded, since supersession is the sanctioned correction path (§G.1 append-only).
    """
    seen: dict[str, str] = {}
    for record in manifest.records.values():
        if record.superseded_by is not None:
            continue
        prior = seen.get(record.object_id)
        if prior is not None and prior != record.partition:
            raise PartitionError(
                f"{record.object_id!r} appears in both {prior!r} and {record.partition!r}"
            )
        seen[record.object_id] = record.partition


def assert_confirmatory_is_clean(
    manifest: PartitionManifest, confirmatory_ids: Iterable[str]
) -> None:
    """Release-verifier check: no confirmatory object appears in discovery or calibration (§G.21)."""
    burned = {
        oid
        for oid, rec in manifest.records.items()
        if rec.partition in OUTCOME_EXPOSED and rec.superseded_by is None
    }
    leaked = sorted(set(confirmatory_ids) & burned)
    if leaked:
        raise PartitionError(
            f"{len(leaked)} confirmatory object(s) were previously outcome-exposed: {leaked[:10]}"
        )


def inherit(parent: Mapping[str, object]) -> Partition:
    """Return the partition a child of ``parent`` must take."""
    partition = parent.get("partition")
    if partition not in PARTITIONS:
        raise PartitionError(f"parent carries no valid partition: {partition!r}")
    return partition  # type: ignore[return-value]
