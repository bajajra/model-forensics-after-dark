"""Estimands for the matched-branch prediction test (Fastrack S1).

The primary quantity is **within-checkpoint pairwise accuracy**: over pairs of sibling tails
from the same frozen checkpoint, one that went on to task-game and one that did not, how often
does the scorer rank the gaming one higher? Ties count 0.5.

Three properties make it the right primary here and each one cost a study somewhere:

**It is blind to everything the siblings share.** Sibling tails were sampled from an identical
token prefix and an identical workspace tree. History depth, elapsed turns, remaining mypy
errors, failure count, reminder presence and instruction distance are constant inside a
checkpoint, so any scorer built from them is pinned to exactly 0.5. That is not a defect of the
telemetry baseline — it is the whole point of pairing, and it means a lens score above 0.5 here
cannot be a repackaged history-length effect. E1 spent a study establishing that history length
is the dominant axis; this design removes it by construction rather than by adjustment.

**The unit of generalization is the source prefix, never the tail** (CLAUDE.md section 7 rule
1). Pooling pairs would let one prolific prefix dominate, and E1 measured a naive binomial over
tails as 4-5x too narrow. So the statistic is a mean over prefixes of a mean over that prefix's
mixed checkpoints, and the interval comes from resampling prefixes.

**Within a checkpoint it is exactly the AUC**, so nothing is lost relative to the usual
threshold-free summary; what changes is that the AUC is computed inside a matched set instead
of across a pooled corpus where prevalence varies by checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PairwiseResult:
    """A pairwise-accuracy estimate and everything needed to read it honestly."""

    estimate: float
    lo: float
    hi: float
    n_prefixes: int
    n_checkpoints: int
    n_pairs: int
    per_prefix: tuple[float, ...]

    def __str__(self) -> str:
        return (
            f"{self.estimate:.4f} [{self.lo:.4f}, {self.hi:.4f}] "
            f"({self.n_prefixes} prefixes, {self.n_checkpoints} mixed checkpoints, "
            f"{self.n_pairs:,} pairs)"
        )


def checkpoint_pairwise(scores: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    """Pairwise accuracy inside one checkpoint. Returns (accuracy, n_pairs)."""
    pos = scores[labels]
    neg = scores[~labels]
    if pos.size == 0 or neg.size == 0:
        return (float("nan"), 0)
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return (float(wins / diff.size), int(diff.size))


def pairwise_table(
    frame: pd.DataFrame,
    score_col: str,
    *,
    group_col: str = "checkpoint_id",
    prefix_col: str = "source_prefix_id",
    label_col: str = "game",
) -> pd.DataFrame:
    """One row per mixed checkpoint: its pairwise accuracy and pair count."""
    rows = []
    for (prefix, group), sub in frame.groupby([prefix_col, group_col], sort=False):
        acc, n_pairs = checkpoint_pairwise(
            sub[score_col].to_numpy(dtype=float), sub[label_col].to_numpy(dtype=bool)
        )
        if n_pairs:
            rows.append(
                {prefix_col: prefix, group_col: group, "accuracy": acc, "n_pairs": n_pairs}
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_pairwise(
    frame: pd.DataFrame,
    score_col: str,
    *,
    group_col: str = "checkpoint_id",
    prefix_col: str = "source_prefix_id",
    label_col: str = "game",
    n_boot: int = 10_000,
    seed: int = 0,
) -> PairwiseResult:
    """Prefix-clustered bootstrap of within-checkpoint pairwise accuracy.

    Resampling is over SOURCE PREFIXES with replacement — the independent unit — and each
    resampled prefix contributes all of its checkpoints. Resampling checkpoints or tails
    instead would produce an interval that looks tight and answers nothing.
    """
    table = pairwise_table(
        frame, score_col, group_col=group_col, prefix_col=prefix_col, label_col=label_col
    )
    if table.empty:
        return PairwiseResult(float("nan"), float("nan"), float("nan"), 0, 0, 0, ())

    by_prefix = table.groupby(prefix_col).accuracy.mean()
    prefixes = by_prefix.index.to_numpy()
    values = by_prefix.to_numpy()
    estimate = float(values.mean())

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(prefixes), size=(n_boot, len(prefixes)))
    boot = values[draws].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return PairwiseResult(
        estimate=estimate,
        lo=float(lo),
        hi=float(hi),
        n_prefixes=len(prefixes),
        n_checkpoints=int(len(table)),
        n_pairs=int(table.n_pairs.sum()),
        per_prefix=tuple(float(v) for v in values),
    )


def paired_bootstrap_difference(
    frame: pd.DataFrame,
    score_a: str,
    score_b: str,
    *,
    group_col: str = "checkpoint_id",
    prefix_col: str = "source_prefix_id",
    label_col: str = "game",
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap the DIFFERENCE in pairwise accuracy between two scorers, paired by prefix.

    The plan's bar is not "the lens beats chance" but "the lens adds value beyond observable
    text". That is a difference, and differencing inside each bootstrap draw is what keeps the
    two scorers' shared prefix-level noise from inflating the interval.
    """
    ta = pairwise_table(frame, score_a, group_col=group_col, prefix_col=prefix_col,
                        label_col=label_col).groupby(prefix_col).accuracy.mean()
    tb = pairwise_table(frame, score_b, group_col=group_col, prefix_col=prefix_col,
                        label_col=label_col).groupby(prefix_col).accuracy.mean()
    joined = pd.concat([ta.rename("a"), tb.rename("b")], axis=1).dropna()
    delta = (joined.a - joined.b).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(n_boot, len(delta)))
    boot = delta[draws].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi)


def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float = 0.05) -> float:
    """Recall at a fixed false-positive rate, computed pooled (not within-checkpoint)."""
    neg = scores[~labels]
    pos = scores[labels]
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    threshold = np.quantile(neg, 1.0 - fpr)
    return float((pos > threshold).mean())


def summarise(frame: pd.DataFrame, score_col: str, *, label_col: str = "game") -> dict[str, float]:
    """Pooled secondary metrics. Reported beside, never instead of, the paired primary."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    labels = frame[label_col].to_numpy(dtype=bool)
    scores = frame[score_col].to_numpy(dtype=float)
    out: dict[str, float] = {}
    if labels.any() and not labels.all():
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
        out["recall_at_5pct_fpr"] = recall_at_fpr(scores, labels, 0.05)
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            out["brier_minmax_scaled"] = float(
                brier_score_loss(labels, (scores - lo) / (hi - lo))
            )
    out["base_rate"] = float(labels.mean())
    out["n"] = float(len(labels))
    return out
