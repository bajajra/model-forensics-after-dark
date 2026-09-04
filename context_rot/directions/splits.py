"""Grouped deterministic splitting for contrastive datasets (plan D.11).

group_key = task_card_id + template_family + seed_family + role_family
bucket    = uint64(sha256(group_key)[:16], 16) mod 100
train 0-59 | validation 60-79 | final test 80-99

Examples are NEVER moved across splits to improve performance (D.11).
"""
from __future__ import annotations

import hashlib

SPLITS = ("train", "validation", "test")


def group_key(task_card_id: str, template_family: str, seed_family: str,
              role_family: str = "") -> str:
    return f"{task_card_id}|{template_family}|{seed_family}|{role_family}"


def bucket_of(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % 100


def split_of(key: str) -> str:
    b = bucket_of(key)
    if b < 60:
        return "train"
    if b < 80:
        return "validation"
    return "test"
