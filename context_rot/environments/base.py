"""Environment adapter contract and registry."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ENVS_ROOT: Final = Path(__file__).resolve().parents[3].parent / "agent-interp-envs"
REPO_ROOT: Final = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class EnvironmentSpec:
    """Everything the rollout harness needs to drive one environment.

    ``setup`` receives an empty workspace directory and the variant key, and must leave the tree
    in the exact state the agent starts from — including git history, when the task's shortcuts
    involve git. Environments shipping a single variant (FD-38: no difficulty ladder in this
    packet) ignore the variant argument but still accept it, so the runner has one code path.

    ``grade`` receives the finished workspace and the agent's command history, and returns the
    environment's own verdict. It is deliberately the *upstream* grader: see the package docstring
    for why we do not write our own per environment.

    The remaining fields exist because "the environment" is not only the file tree. Which tools
    the model is offered, which commands are refused, and what is on the agent's ``PATH`` all
    change which strategies are reachable, and every one of them differs between Pre-commit Hook
    and Revert-or-Refactor. Leaving them implicit in the runner is how a second environment
    silently inherits the first one's affordances.
    """

    env_key: str
    config_path: Path
    setup: Callable[[Path, str], None]
    grade: Callable[[Path, list[str]], dict[str, Any]]
    max_steps: int
    #: Short description of the noncompliant shortcuts this environment affords, so a reader can
    #: see what "task gaming" means here without opening the upstream README.
    shortcuts: tuple[str, ...]
    #: Stated coverage limits of the upstream grader. Never left empty — a grader whose blind
    #: spots are unknown is more dangerous than one whose blind spots are written down.
    grader_limits: tuple[str, ...]
    #: Upstream commit the environment files are pinned at, for the §5.8 environment record.
    env_commit: str
    #: Short slug used in `env_id` (`ids.env_id`), kept stable across studies.
    env_slug: str
    #: Tool names the model is offered, in the config's order. Read from the environment's own
    #: `task.tools`; the runner turns these into JSON tool schemas.
    tool_names: tuple[str, ...]
    #: Commands the upstream harness refuses without executing. Pre-commit Hook's `run_step`
    #: routes through `validate_and_get_command`, which enforces `BANNED_COMMANDS`;
    #: Revert-or-Refactor's does not, and neither may be assumed of the other (see the adapters).
    banned_commands: tuple[str, ...]
    #: Directories prepended to the agent shell's `PATH`, supplying the pinned interpreters and
    #: checkers the container's Dockerfile would have installed.
    path_shims: tuple[Path, ...] = ()
    #: Optional per-turn objective progress telemetry, measured by us rather than self-reported.
    #: Returns a number, or None when it cannot be measured. `remaining_mypy_errors` is the
    #: Pre-commit Hook instance; this environment counts failing tests.
    progress_metric: Callable[[Path], int | None] | None = None
    #: Name of the progress metric, for the record. Empty when there is none.
    progress_metric_name: str = ""

    def prompts(self) -> dict[str, str]:
        """System and user prompt, read from the environment's own frozen config."""
        cfg = self.config()
        p = cfg.get("prompts", {})
        if not p.get("system_prompt") or not p.get("user_prompt"):
            raise ValueError(f"{self.config_path} is missing a system or user prompt")
        return {"system_prompt": p["system_prompt"], "user_prompt": p["user_prompt"]}

    def config(self) -> dict[str, Any]:
        """The environment's own frozen config, parsed. Never re-authored in this repo."""
        import yaml

        if self.config_path.suffix in (".yaml", ".yml"):
            return dict(yaml.safe_load(self.config_path.read_text()))
        return dict(json.loads(self.config_path.read_text()))

    def __post_init__(self) -> None:
        # FD-36 / C1.8: the field is required and must not be empty. A grader whose blind spots
        # are undeclared gets used as if it had none.
        if not self.grader_limits:
            raise ValueError(f"{self.env_key}: grader_limits must not be empty")
        if not self.tool_names:
            raise ValueError(f"{self.env_key}: tool_names must not be empty")


def git(workspace: Path, *args: str) -> None:
    """Run a git command in the workspace, failing loudly."""
    subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)


def upstream_scorer(env_key: str, fn_name: str = "score_variant") -> Callable[..., Any]:
    """Import a scoring function from the upstream environment.

    Imported lazily and by path, because the upstream tree is a sibling checkout rather than an
    installed package, and because importing every environment's scorer at startup would drag in
    dependencies that only one of them needs.
    """
    import importlib.util

    path = ENVS_ROOT / "environments" / env_key / "score.py"
    if not path.exists():
        raise FileNotFoundError(f"{env_key} ships no score.py at {path}")
    spec = importlib.util.spec_from_file_location(f"_upstream_{env_key}_score", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, fn_name)  # type: ignore[no-any-return]


def upstream_tool_schemas(env_key: str, names: Sequence[str]) -> list[dict[str, Any]]:
    """Load tool JSON schemas from the environment's own `tools.py`.

    Read from upstream rather than transcribed, for the same reason prompts are: a tool's name
    and description are part of the rendered prompt, so a re-authored one is a new prompt variant
    nobody froze (T41).

    Read by **AST**, not by import. `tools.py` imports `agent_interp_envs.tool_calling`, which
    imports the OpenAI and Anthropic SDKs at module scope; installing two provider clients into
    the harness environment to read a dict literal would be a real dependency for a nominal need,
    and `agent-interp-envs` is a sibling checkout rather than an installed package. Every tool
    definition in these files is a module-level literal, so `ast.literal_eval` reads them exactly,
    with no import side effects. If upstream ever computes one, this raises rather than guesses.
    """
    env_path = ENVS_ROOT / "environments" / env_key
    path = env_path / "tools.py"
    if not path.exists():
        raise FileNotFoundError(f"{env_key} ships no tools.py at {path}")

    literals = _module_literals(path)
    # `TOOL_REGISTRY` maps tool name -> the module-level constant holding its schema. The
    # registry itself is a dict of Names, not literals, so resolve it from the AST separately.
    registry = _tool_registry_names(path)
    imported = set(_imported_names(path))
    for missing_name in imported - literals.keys():
        # e.g. EXECUTE_COMMAND_TOOL, defined in `agent_interp_envs.tool_calling`.
        source = ENVS_ROOT / "src" / "agent_interp_envs" / "tool_calling.py"
        if source.exists():
            literals.update(
                {k: v for k, v in _module_literals(source).items() if k == missing_name}
            )

    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"{env_key}: unknown tool(s) {unknown}; have {sorted(registry)}")
    schemas = []
    for name in names:
        const = registry[name]
        if const not in literals:
            raise ValueError(
                f"{env_key}: tool {name!r} maps to {const!r}, which is not a module-level "
                "literal in tools.py — read it by hand rather than letting this guess."
            )
        schemas.append(literals[const])
    return schemas


def _module_literals(path: Path) -> dict[str, Any]:
    """Every module-level `NAME = <literal>` assignment in a Python file."""
    import ast

    out: dict[str, Any] = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return out


def _tool_registry_names(path: Path) -> dict[str, str]:
    """`TOOL_REGISTRY`'s name -> constant-identifier mapping, read from the AST."""
    import ast

    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Name) and target.id == "TOOL_REGISTRY"):
            continue
        if not isinstance(node.value, ast.Dict):
            break
        mapping: dict[str, str] = {}
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            if isinstance(key, ast.Constant) and isinstance(value, ast.Name):
                mapping[str(key.value)] = value.id
        return mapping
    raise ValueError(f"{path} has no literal TOOL_REGISTRY mapping")


def _imported_names(path: Path) -> list[str]:
    import ast

    names: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom):
            names.extend(alias.asname or alias.name for alias in node.names)
    return names


_REGISTRY: dict[str, Callable[[], EnvironmentSpec]] = {}


def register(key: str) -> Callable[[Callable[[], EnvironmentSpec]], Callable[[], EnvironmentSpec]]:
    def deco(fn: Callable[[], EnvironmentSpec]) -> Callable[[], EnvironmentSpec]:
        _REGISTRY[key] = fn
        return fn

    return deco


def get_environment(key: str) -> EnvironmentSpec:
    if key not in _REGISTRY:
        _load_all()
    if key not in _REGISTRY:
        raise KeyError(f"unknown environment {key!r}; have {sorted(_REGISTRY)}")
    return _REGISTRY[key]()


def list_environments() -> list[str]:
    _load_all()
    return sorted(_REGISTRY)


def _load_all() -> None:
    import importlib

    for mod in ("precommit_hook", "revert_or_refactor"):
        importlib.import_module(f"{__package__}.{mod}")


def write_provenance(workspace: Path, spec: EnvironmentSpec) -> None:
    """Record which environment built this workspace, next to the workspace itself."""
    (workspace / ".env_provenance.json").write_text(
        json.dumps(
            {
                "env_key": spec.env_key,
                "config": str(spec.config_path),
                "shortcuts": list(spec.shortcuts),
                "grader_limits": list(spec.grader_limits),
            },
            indent=2,
        )
    )
