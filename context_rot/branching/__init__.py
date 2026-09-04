"""Prefix-conditioned continuation sampling (§9.5, §E.3)."""

from __future__ import annotations

from .tails import TailJob, TailResult, sample_tail

__all__ = ["TailJob", "TailResult", "sample_tail"]
