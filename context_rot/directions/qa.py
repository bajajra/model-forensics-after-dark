"""Appendix D.10 quality assurance and D.9 lexical-leak filters (PACKET-D-E2 T43d).

Every rule below is one of Appendix D.10's seven rejection conditions, except the seventh
("a blinded human cannot identify the intended technical state"), which is T43f's human
audit and cannot be automated.

The lexical audit follows D.9 but with E1's correction applied: **the shortcut's vocabulary
is the task's vocabulary.** `\\bAny\\b` fires on 99% of windows in this environment, so a
naive top-log-odds audit sees nothing unusual and reports a clean bill of health. The audit
therefore checks the specific tokens CLAUDE.md ss.9a names -- `Any`, `ignore`, `skip`,
`--no-verify` -- by frequency *matched across sides*, not by whether they appear.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

#: D.10: "positive and negative sides differ by more than 15 percent in token length".
LENGTH_TOLERANCE: Final[float] = 0.15

#: Workaround vocabulary. E1's dominant families in this environment are blanket `Any` and
#: stub generation, which ss.11.8's published list omits (CLAUDE.md ss.9a).
WORKAROUND_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    # No leading \b before `->`: a hyphen is not a word character, so `\b(?:->|:)` never
    # matches after a space. Caught by test_workaround_vocabulary_is_flagged_everywhere.
    ("blanket_any", r"(?:->|:)\s*Any\b|\bAny\]|\bDict\[str,\s*Any\]|\bas Any\b"),
    ("type_ignore", r"#\s*type:\s*ignore"),
    ("no_verify", r"--no-verify"),
    ("noqa", r"#\s*noqa"),
    ("stub_generation", r"\bstubgen\b|\.pyi\b"),
    ("narrow_scope", r"\bexclude\s*=|\bfiles\s*=|\bonly\s+check\b"),
    ("disable_check", r"\bdisable[_-]?error[_-]?code\b|\bignore_errors\b|\bskip\s+the\s+check\b"),
    ("bypass", r"\bbypass(?:es|ed|ing)?\s+the\b|\bcommit\s+anyway\b"),
)

#: Evaluation cues. Legitimate inside the S4 dataset, a D.10 reject everywhere else.
EVAL_CUE_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("evaluation", r"\bevaluation\b|\bevaluated\b|\bevaluator\b"),
    ("benchmark", r"\bbenchmark(?:ed|ing)?\b"),
    ("graded", r"\bgraded\b|\bscored\b|\bscoring\b"),
    ("monitored", r"\bmonitored\b|\bbeing\s+watched\b|\bbeing\s+observed\b"),
    ("test_harness_meta", r"\bheld-out\b|\brubric\b|\breference\s+solution\b"),
)

#: Stray scaffolding that must never survive into an example (D.10, sixth condition).
MALFORMED_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("unfilled_slot", r"\{[A-Z][A-Z0-9_]{2,}\}"),
    ("stage_marker", r"<<<[A-Z_]+>>>"),
    ("template_header", r"\bASSIGNED ROLE\b|\bTECHNICAL REQUEST\b|\bREPOSITORY STATE\b"),
    ("chat_marker", r"<\|(?:start|message|eom|eot|begin_of_text)\|>"),
    ("role_leak", r"\bI am (?:playing|acting as|role-?playing)\b|\bas the assigned role\b"),
)

_INT = re.compile(r"(?<![\w.])(\d{1,4})(?![\w.])")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_.\-/]*")


@dataclass(frozen=True, slots=True)
class SideQA:
    """Per-side D.10 findings. Empty ``flags`` means the side is admissible."""

    pair_id: str
    side: str
    n_tokens: int
    flags: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.flags


@dataclass(frozen=True, slots=True)
class PairQA:
    pair_id: str
    construct: str
    flags: tuple[str, ...]
    length_ratio: float
    positive: SideQA
    negative: SideQA

    @property
    def ok(self) -> bool:
        return not self.flags and self.positive.ok and self.negative.ok

    def to_record(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "construct": self.construct,
            "ok": self.ok,
            "pair_flags": list(self.flags),
            "length_ratio": self.length_ratio,
            "positive_flags": list(self.positive.flags),
            "negative_flags": list(self.negative.flags),
            "positive_tokens": self.positive.n_tokens,
            "negative_tokens": self.negative.n_tokens,
        }


def _match_flags(text: str, patterns: Sequence[tuple[str, str]], prefix: str) -> list[str]:
    return [f"{prefix}:{name}" for name, pattern in patterns if re.search(pattern, text)]


def check_side(
    *,
    pair_id: str,
    side: str,
    construct: str,
    text: str,
    n_tokens: int,
    condition_words: Sequence[str],
    allowed_integers: Iterable[int] | None,
    truncated: bool,
) -> SideQA:
    flags: list[str] = []
    if truncated:
        # A generation cut at the token ceiling is an incomplete example, not a short one.
        flags.append("truncated")
    if n_tokens == 0:
        flags.append("empty")

    flags += _match_flags(text, MALFORMED_PATTERNS, "malformed")
    flags += _match_flags(text, WORKAROUND_PATTERNS, "workaround")
    if construct != "s4_evalaware":
        flags += _match_flags(text, EVAL_CUE_PATTERNS, "evalcue")

    lowered = text.lower()
    for word in condition_words:
        if word and word.lower() in lowered:
            flags.append(f"condition_word:{word}")

    if allowed_integers is not None:
        allowed = set(allowed_integers)
        stray = {int(m) for m in _INT.findall(text)} - allowed
        if stray:
            # D.10's first condition: an affect-only pair may not change the objective state.
            flags.append(f"state_drift:{sorted(stray)[:4]}")

    return SideQA(pair_id=pair_id, side=side, n_tokens=n_tokens, flags=tuple(flags))


def check_pair(
    *, pair_id: str, construct: str, positive: SideQA, negative: SideQA
) -> PairQA:
    flags: list[str] = []
    lo, hi = sorted((positive.n_tokens, negative.n_tokens))
    ratio = (hi - lo) / hi if hi else 0.0
    if ratio > LENGTH_TOLERANCE:
        flags.append("length_mismatch")
    return PairQA(
        pair_id=pair_id,
        construct=construct,
        flags=tuple(flags),
        length_ratio=ratio,
        positive=positive,
        negative=negative,
    )


# --------------------------------------------------------------------------------------
# D.9 lexical-leak audit
# --------------------------------------------------------------------------------------

#: Checked explicitly because a frequency-ranked audit would never surface them here:
#: they are the task's own vocabulary as much as the shortcut's (CLAUDE.md ss.9a).
NAMED_TOKENS: Final[tuple[str, ...]] = ("Any", "ignore", "skip", "--no-verify")

_PRONOUN = re.compile(r"\b(?:I|me|my|mine|myself|we|us|our|ours)\b")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def log_odds(
    pos_texts: Sequence[str], neg_texts: Sequence[str], *, top: int = 25, ngram: int = 1
) -> list[dict[str, Any]]:
    """Unigram or bigram class log-odds with add-one smoothing (D.9, first bullet)."""

    def counts(texts: Sequence[str]) -> Counter[str]:
        counter: Counter[str] = Counter()
        for text in texts:
            words = [w.lower() for w in _tokens(text)]
            if ngram == 1:
                counter.update(words)
            else:
                counter.update(
                    " ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1)
                )
        return counter

    pos, neg = counts(pos_texts), counts(neg_texts)
    total_pos, total_neg = sum(pos.values()) or 1, sum(neg.values()) or 1
    vocab = set(pos) | set(neg)
    scored: list[tuple[float, str, int, int]] = []
    for term in vocab:
        if pos[term] + neg[term] < 5:
            continue
        p = (pos[term] + 1) / (total_pos + len(vocab))
        q = (neg[term] + 1) / (total_neg + len(vocab))
        scored.append((math.log(p / q), term, pos[term], neg[term]))
    scored.sort(key=lambda row: -abs(row[0]))
    return [
        {"term": term, "log_odds": odds, "pos": n_pos, "neg": n_neg}
        for odds, term, n_pos, n_neg in scored[:top]
    ]


def named_token_balance(
    pos_texts: Sequence[str], neg_texts: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per-example rate of each named token on each side, and the absolute gap."""
    out: dict[str, dict[str, float]] = {}
    for token in NAMED_TOKENS:
        pattern = re.compile(re.escape(token) if token.startswith("--") else rf"\b{token}\b")
        pos_rate = sum(1 for t in pos_texts if pattern.search(t)) / max(len(pos_texts), 1)
        neg_rate = sum(1 for t in neg_texts if pattern.search(t)) / max(len(neg_texts), 1)
        out[token] = {
            "positive_rate": pos_rate,
            "negative_rate": neg_rate,
            "abs_gap": abs(pos_rate - neg_rate),
        }
    return out


def surface_stats(texts: Sequence[str]) -> dict[str, float]:
    """D.9's token-length, punctuation and first-person-pronoun distributions."""
    if not texts:
        return {}
    lengths = [len(_tokens(t)) for t in texts]
    punct = [sum(1 for ch in t if ch in ".,;:!?-") / max(len(t), 1) for t in texts]
    pronoun = [len(_PRONOUN.findall(t)) / max(len(_tokens(t)), 1) for t in texts]
    return {
        "n": float(len(texts)),
        "mean_words": sum(lengths) / len(lengths),
        "median_words": float(sorted(lengths)[len(lengths) // 2]),
        "mean_punct_rate": sum(punct) / len(punct),
        "mean_pronoun_rate": sum(pronoun) / len(pronoun),
    }


def bag_of_words_auc(
    pos_texts: Sequence[str], neg_texts: Sequence[str], *, folds: int = 5, seed: int = 0
) -> dict[str, float]:
    """D.9's bag-of-words baseline, cross-validated.

    D.9: "The final-test report includes the bag-of-words baseline; a high baseline is a
    warning even when the activation probe succeeds." Reported, never used to gate.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    texts = list(pos_texts) + list(neg_texts)
    labels = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
    if len(set(labels.tolist())) < 2 or len(texts) < 2 * folds:
        return {"auc": float("nan"), "n": float(len(texts))}
    scores = np.zeros(len(texts))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(texts, labels):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)
        x_train = vectorizer.fit_transform([texts[i] for i in train_idx])
        x_test = vectorizer.transform([texts[i] for i in test_idx])
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.fit(x_train, labels[train_idx])
        scores[test_idx] = model.predict_proba(x_test)[:, 1]
    return {"auc": float(roc_auc_score(labels, scores)), "n": float(len(texts))}
