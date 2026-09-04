"""Multi-environment support (`execution_plan.md` §5.7, §14.3, §18.21).

§18.21 states the limitation plainly: *"The primary claim is one model in one environment."*
Everything through E5 runs on Pre-commit Hook. This package is what lets the same harness drive a
second and third, so the claim can be checked somewhere else.

## Why an adapter rather than "just run the environment"

The upstream environments are built to run **inside their own Docker containers**, and they assume
it: absolute paths (`/agent`, `/opt/tasks/...`), a root loop dropping each agent command to a
non-root `dev` user, a private grader hidden under root-only `/opt`. This node has **no container
runtime** (DEV-0005 — it is itself a Run:ai pod), so each environment needs its setup reproduced
against a relocatable workspace root, exactly as `rollout.workspace.seed_workspace` already does
for Pre-commit Hook.

What an adapter must supply is small and fixed:

- **setup** — build the starting workspace, including any git history the task depends on;
- **prompts** — read from the environment's own frozen config, never re-authored here;
- **grading** — call the environment's own `score.py` where it has one.

Reusing the upstream grader matters. Ours for Pre-commit Hook took three rounds to become
trustworthy (DEV-0013 check 11, DEV-0014 check 12, and it still passed 17/17 while blind to the
commonest shortcut). Writing three more from scratch would mean three more instruments with
unmeasured blind spots. Where upstream ships a grader, that is the one to use, and its coverage is
then a stated limitation rather than an invented one.
"""

from __future__ import annotations

from .base import EnvironmentSpec, get_environment, list_environments

__all__ = ["EnvironmentSpec", "get_environment", "list_environments"]
