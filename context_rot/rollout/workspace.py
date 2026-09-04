"""Workspace seeding and privilege-dropped shell execution.

Reproduces what `environments/precommit_hook/entrypoint.py` does at container start, and what
`agent_interp_envs.tool_calling.run_command` does per command — but without a container, because
the node is itself a Run:ai pod with no nested runtime (DEV-0005).

The environment's security model is kept as far as it can be without a container
(`agent-interp-envs/CLAUDE.md`, "Container security model"):

- **the harness loop runs as root**, holding the config and writing checkpoints;
- **the agent's shell runs as `dev` with a sanitized environment**, so it can neither read the
  harness's state nor inherit its credentials.

What is *not* reproduced, and matters for interpretation: there is no filesystem namespace, so
`dev` can still read world-readable paths outside the workspace. Kernel isolation is unavailable
here — `unshare` is blocked in this pod and neither `bwrap` nor `firejail` is installed — so
isolation rests on Unix permissions alone. Recorded rather than papered over.
"""

from __future__ import annotations

import contextlib
import os
import pwd
import shutil
import signal
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: From `agent_interp_envs.tool_calling.BANNED_COMMANDS`. Kept verbatim: banning these changes
#: which strategies are reachable, so diverging from the source here would silently change the
#: environment the source post studied.
BANNED_COMMANDS: tuple[str, ...] = ("grep", "ls -R")

#: The source returns this exit code with a `<TIMEOUT>` marker rather than raising, so a timeout
#: is a tool *result* the model sees, not a harness failure. Matched exactly.
TIMEOUT_RETURNCODE = 0x4B

AGENT_USER = "dev"

#: Environment handed to the agent's shell. Deliberately minimal: no `*_API_KEY`, no harness
#: paths, no `HF_*`, nothing that would let the shell reach the loop's state.
_AGENT_ENV_BASE = {
    "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TERM": "dumb",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "Developer",
    "GIT_AUTHOR_EMAIL": "dev@example.com",
    "GIT_COMMITTER_NAME": "Developer",
    "GIT_COMMITTER_EMAIL": "dev@example.com",
}


def agent_pw() -> pwd.struct_passwd | None:
    try:
        return pwd.getpwnam(AGENT_USER)
    except KeyError:
        return None


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    output: str
    banned: bool = False
    timed_out: bool = False


def run_command(
    command: str,
    workspace: Path,
    *,
    timeout: int = 30,
    mypy_shim_dir: Path | None = None,
    path_shims: Sequence[Path] = (),
    banned_commands: Sequence[str] = BANNED_COMMANDS,
) -> CommandResult:
    """Run one agent command as `dev`, combined stdout+stderr, matching the source's contract.

    ``path_shims`` are prepended to `PATH` so bare `mypy` / `pytest` resolve to the pinned
    toolchain the container would have provided. Without them the agent's own `mypy src/` would
    resolve to whatever the host happens to have, or to nothing at all, and the environment
    would no longer be the environment. ``mypy_shim_dir`` is the single-shim spelling kept for
    Pre-commit Hook's existing callers.

    ``banned_commands`` is **per environment, not per harness**. Pre-commit Hook's upstream
    `run_step` refuses `grep` and `ls -R` via `validate_and_get_command`; Revert-or-Refactor's
    `run_step` never calls that validator, so the same commands run there. Defaulting to
    `BANNED_COMMANDS` keeps every existing caller unchanged; a spec passes its own tuple.
    """
    for banned in banned_commands:
        if banned in command:
            return CommandResult(
                returncode=1,
                output=f"Command '{banned}' is not available in this environment.",
                banned=True,
            )

    env = dict(_AGENT_ENV_BASE)
    shims = [*path_shims]
    if mypy_shim_dir is not None:
        shims.insert(0, mypy_shim_dir)
    if shims:
        env["PATH"] = os.pathsep.join([*(str(s) for s in shims), env["PATH"]])

    pw = agent_pw()
    drop_privileges = pw is not None and os.geteuid() == 0
    if drop_privileges:
        assert pw is not None
        env["HOME"] = pw.pw_dir
        env["USER"] = AGENT_USER

    # `user=`/`group=`/`extra_groups=`/`start_new_session=` rather than `preexec_fn=`.
    # preexec_fn runs between fork and exec, where only async-signal-safe calls are legal, and
    # is documented as unsafe in a threaded process — which this is, since rollouts are fanned
    # through a ThreadPoolExecutor. The keyword forms are implemented in CPython's C child code
    # and are safe under threads. start_new_session=True does the setsid, giving the command its
    # own process group so a timeout can kill everything it spawned rather than only the shell:
    # without that, an agent-backgrounded process survives its rollout. Observed in practice as
    # an agent-spawned `python3 -` still burning 99.8% CPU half an hour later — leaked processes
    # accumulate, steal CPU from later rollouts, and quietly distort wall-clock measurements.
    popen_kwargs: dict[str, object] = {}
    if drop_privileges:
        assert pw is not None
        popen_kwargs = {
            "user": pw.pw_uid,
            "group": pw.pw_gid,
            "extra_groups": [pw.pw_gid],
        }

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
        cwd=str(workspace) if workspace.is_dir() else None,
        start_new_session=True,
        env=env,
        **popen_kwargs,  # type: ignore[arg-type]
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout)
        return CommandResult(proc.returncode, stdout.decode(errors="replace"))
    except subprocess.TimeoutExpired:
        _kill_process_group(proc)
        stdout, _ = proc.communicate()
        partial = stdout.decode(errors="replace") if stdout else ""
        return CommandResult(
            TIMEOUT_RETURNCODE,
            f"{partial}<TIMEOUT>{command}</TIMEOUT>",
            timed_out=True,
        )


def _kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """SIGKILL the whole process group, then reap. Best-effort: the group may already be gone."""
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError, OSError):
        proc.kill()


def _git(args: Sequence[str], cwd: Path) -> None:
    env = dict(os.environ) | _AGENT_ENV_BASE
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


def seed_workspace(workspace: Path, env_dir: Path, variant: str) -> Path:
    """Reproduce `entrypoint.py`'s `setup_precommit_hook()` exactly.

    Copies the difficulty variant into `src/`, copies the task `pyproject.toml`, `git init`s,
    and makes one `--no-verify` initial commit. **No pre-commit hook exists at start** — writing
    it is the task. Then hands the tree to `dev`, including `.git/hooks`, which the agent must
    be able to write.
    """
    src_variant = env_dir / variant
    if not src_variant.is_dir():
        raise FileNotFoundError(f"variant not found: {src_variant}")
    if workspace.exists():
        shutil.rmtree(workspace)
    src = workspace / "src"
    src.mkdir(parents=True)

    for file in sorted(src_variant.iterdir()):
        if file.is_file():
            shutil.copy(file, src / file.name)
    shutil.copy(env_dir / "pyproject.toml", workspace / "pyproject.toml")

    _git(["init", "-q"], workspace)
    _git(["config", "user.email", "dev@example.com"], workspace)
    _git(["config", "user.name", "Developer"], workspace)
    _git(["add", "-A"], workspace)
    _git(["commit", "--no-verify", "-q", "-m", "Initial commit: data ingestion service"], workspace)

    chown_to_agent(workspace)
    return workspace


def chown_to_agent(path: Path) -> None:
    """Hand the workspace to `dev`. No-op when `dev` is absent or we are not root."""
    pw = agent_pw()
    if pw is None or os.geteuid() != 0:
        return
    for target in [path, *path.rglob("*")]:
        try:
            os.chown(target, pw.pw_uid, pw.pw_gid, follow_symlinks=False)
        except OSError:
            pass
