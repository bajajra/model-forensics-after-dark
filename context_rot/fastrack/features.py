"""Feature construction for the matched-branch prediction test (Fastrack S1).

The improvement plan asks for a family score built as a log-sum-exp over a family's lens
"logits". Our readout is a **cosine**, not a logit, and cosines here sit around 0.01-0.05 — so
``LSE`` over raw cosines degenerates to ``log|G| + mean(z)`` and the soft-max does nothing at
all. Standardising each token against its own distribution first restores the intended
behaviour: after standardisation a token that is unusually present for *that token* dominates
the family sum, which is what "a short-lived ignition should not be averaged away" means.

Standardisation statistics come from the **discovery partition only** and are applied unchanged
to every held-out read. Fitting them on the pooled corpus would leak the test set's marginal
distribution into the test set's features — a small leak, but the kind that turns a null into a
0.53 and cannot be detected afterwards.

Four feature families are built, because the plan asks for the decomposition and because they
answer different questions:

``hand``      the plan's own score, ``S_shortcut - S_norm``. One number. No fitting.
``families``  one LSE per declared family. ~11 numbers per (lens, layer, pooling).
``bank``      every one of the 162 standardised token scores. Maximal, needs regularisation.
``residual``  the pooled residual stream ``h``, its projection into the lens span ``h_J``, and
              the remainder ``h - h_J``. This is the plan's test of whether a signal is a
              *verbalizable workspace* signal or merely linearly decodable somewhere.

**Within-checkpoint centering** is offered on every one of them. Subtracting a checkpoint's
sibling mean removes exactly the between-checkpoint variance that the paired primary metric
already ignores, so it aligns the feature with the estimand rather than making the probe spend
capacity modelling a nuisance axis. It is a lever, declared in AMENDMENT_2, and selected on
discovery like any other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Families the plan names, mapped onto the fitted groups. The E2 groups keep their ``e2_``
#: prefix so it is always visible which artifact a number came from.
PLAN_FAMILIES = {
    "shortcut": ("shortcut_affordance", "e2_shortcut"),
    "norm": ("normative_constraint",),
    "repair": ("legitimate_repair",),
}

#: Carried for comparison, not part of the plan's primary score.
EXTRA_FAMILIES = (
    "e2_impatience", "e2_persistence", "e2_morale_low", "e2_morale_high",
    "e2_evaluation", "e2_progress", "e2_control",
)


@dataclass(frozen=True, slots=True)
class ScoreBank:
    """The extraction output, indexed for feature building."""

    meta: pd.DataFrame
    #: [rows, variants, layers, stats, tokens]
    scores: np.ndarray
    tokens: tuple[str, ...]
    layers: tuple[int, ...]
    variants: tuple[str, ...]
    stats: tuple[str, ...]
    groups: dict[str, tuple[int, ...]]

    def axis(self, variant: str, layer: int, stat: str) -> tuple[int, int, int]:
        return (
            self.variants.index(variant),
            self.layers.index(layer),
            self.stats.index(stat),
        )

    def block(self, variant: str, layer: int, stat: str) -> np.ndarray:
        vi, li, si = self.axis(variant, layer, stat)
        return self.scores[:, vi, li, si, :]


def standardisation_stats(
    bank: ScoreBank, fit_mask: np.ndarray
) -> dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]]:
    """Per-token mean and SD for every (lens, layer, pooling), from the fit rows only."""
    out: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]] = {}
    for variant in bank.variants:
        for layer in bank.layers:
            for stat in bank.stats:
                block = bank.block(variant, layer, stat)[fit_mask]
                mean = block.mean(axis=0)
                sd = block.std(axis=0)
                # A token with no variance in the fit set carries no information; a zero SD
                # would otherwise produce inf and quietly dominate every family sum.
                sd = np.where(sd < 1e-9, 1.0, sd)
                out[(variant, layer, stat)] = (mean, sd)
    return out


def standardised(
    bank: ScoreBank,
    stats: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
    variant: str,
    layer: int,
    stat: str,
) -> np.ndarray:
    mean, sd = stats[(variant, layer, stat)]
    return (bank.block(variant, layer, stat) - mean) / sd


def family_lse(z: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Log-sum-exp of a family's standardised scores, per row.

    ``logsumexp`` rather than ``max`` because the plan asks for it and because it degrades
    gracefully: with one strongly elevated token it approaches that token's value, with a
    family elevated as a whole it approaches ``log|G| + mean``. Both are meaningful readings of
    "how present is this family".
    """
    from scipy.special import logsumexp

    if cols.size == 0:
        return np.zeros(z.shape[0], dtype=np.float64)
    return logsumexp(z[:, cols], axis=1)


def resolve_family(bank: ScoreBank, name: str) -> np.ndarray:
    """Column indices for a plan family, unioned over the groups that make it up."""
    if name in PLAN_FAMILIES:
        cols: list[int] = []
        for group in PLAN_FAMILIES[name]:
            cols.extend(bank.groups.get(group, ()))
        return np.asarray(sorted(set(cols)), dtype=int)
    return np.asarray(sorted(set(bank.groups.get(name, ()))), dtype=int)


def build_family_frame(
    bank: ScoreBank,
    stats: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
    *,
    variants: tuple[str, ...] | None = None,
    layers: tuple[int, ...] | None = None,
    pooling: str = "max",
) -> pd.DataFrame:
    """One LSE per family per (lens, layer), plus the plan's hand-designed margin."""
    variants = variants or bank.variants
    layers = layers or bank.layers
    columns: dict[str, np.ndarray] = {}
    for variant in variants:
        for layer in layers:
            z = standardised(bank, stats, variant, layer, pooling)
            values: dict[str, np.ndarray] = {}
            for family in (*PLAN_FAMILIES, *EXTRA_FAMILIES):
                cols = resolve_family(bank, family)
                values[family] = family_lse(z, cols)
                columns[f"{variant}_l{layer:03d}_{family}"] = values[family]
            # The plan's primary hand-designed score.
            columns[f"{variant}_l{layer:03d}_margin"] = values["shortcut"] - values["norm"]
            # A second contrast the plan implies but does not name: shortcut against the
            # legitimate alternative it competes with.
            columns[f"{variant}_l{layer:03d}_margin_repair"] = (
                values["shortcut"] - values["repair"]
            )
    return pd.DataFrame(columns)


def build_bank_frame(
    bank: ScoreBank,
    stats: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray]],
    *,
    variants: tuple[str, ...] | None = None,
    layers: tuple[int, ...] | None = None,
    pooling: str = "max",
) -> pd.DataFrame:
    """Every standardised token score, for a regularised probe over the full bank."""
    variants = variants or bank.variants
    layers = layers or bank.layers
    columns: dict[str, np.ndarray] = {}
    for variant in variants:
        for layer in layers:
            z = standardised(bank, stats, variant, layer, pooling)
            for j, token in enumerate(bank.tokens):
                columns[f"{variant}_l{layer:03d}_tok{j:03d}"] = z[:, j]
    return pd.DataFrame(columns)


def center_within(frame: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Subtract each checkpoint's sibling mean from every column.

    After this, a feature says "how unusual is this continuation *for this hotspot*" rather
    than "how unusual is this hotspot". The paired primary metric only ever compares siblings,
    so the removed component is by definition not usable by it.
    """
    return frame.groupby(groups.to_numpy(), sort=False).transform(lambda s: s - s.mean())


def rank_within(frame: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Rank each column within its checkpoint, scaled to [0, 1].

    The plan's robustness list asks for rank-based aggregation as an alternative to
    score-based. Ranking inside the checkpoint is the natural version here: it is invariant to
    any monotone per-checkpoint transformation of the score.
    """
    return frame.groupby(groups.to_numpy(), sort=False).transform(
        lambda s: s.rank(pct=True) if s.notna().any() else s
    )


def lens_span_projection(
    h: np.ndarray, lens_vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split a pooled residual into its lens-span component and the remainder.

    ``lens_vectors`` is [n_tokens, d]. The lens rows are not orthogonal — cosines between them
    run to 0.5 for semantically close tokens — so the projector is built from an orthonormal
    basis of their span via QR rather than from the rows directly. Returns ``(h_J, h - h_J)``.
    """
    q, _ = np.linalg.qr(lens_vectors.T)
    h_j = (h @ q) @ q.T
    return h_j, h - h_j
