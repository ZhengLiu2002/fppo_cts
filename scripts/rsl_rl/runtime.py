"""Shared runtime helpers for Isaac Lab training and evaluation scripts.

The helpers in this module intentionally avoid importing Isaac Lab at module
import time so they stay lightweight and easy to unit-test.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import pickle
import platform
import re
import sys
from typing import Any

from packaging import version

MIN_DISTRIBUTED_RSL_RL_VERSION = "2.3.1"
_PRETRAINED_CHECKPOINT_NAMESPACE = "rsl_rl"
_HEADLESS_DISABLE_EXTENSIONS = (
    "omni.physx.ui",
    "omni.kit.window.drop_support",
    "omni.kit.menu.utils",
)


def bootstrap_repo_paths(
    anchor_file: str | os.PathLike[str],
    *,
    repo_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Ensure the repository root and optional task extension are importable."""

    repo_root_path = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(anchor_file).resolve().parents[2]
    )
    candidate_paths = [repo_root_path, repo_root_path / "crl_tasks"]
    for candidate_path in reversed(candidate_paths):
        candidate = str(candidate_path)
        if candidate_path.is_dir() and candidate not in sys.path:
            sys.path.insert(0, candidate)
    return repo_root_path


def build_log_root_path(experiment_name: str) -> str:
    """Return the absolute log root for an experiment."""

    return os.path.abspath(os.path.join("logs", "rsl_rl", experiment_name))


def create_run_directory_name(
    run_name_suffix: str | None,
    *,
    timestamp: datetime | None = None,
    env_var_name: str = "LOG_RUN_NAME",
    experiment_slug: str | None = None,
) -> tuple[str, bool]:
    """Create a timestamped run directory name.

    Returns a tuple of `(directory_name, needs_exact_name_log)`. The second
    value preserves the existing Ray Tune behavior in `train.py`.
    """

    log_run_prefix = os.getenv(env_var_name, "").strip()
    current_timestamp = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H-%M-%S")

    name_parts = [part for part in (log_run_prefix, current_timestamp, experiment_slug, run_name_suffix) if part]
    directory_name = "_".join(name_parts)

    return directory_name, not bool(log_run_prefix)


def dump_pickle(file_path: str | os.PathLike[str], data: Any) -> Path:
    """Serialize data to a pickle file, creating parent directories if needed."""

    target_path = Path(file_path)
    if not str(target_path).endswith(".pkl"):
        target_path = Path(f"{target_path}.pkl")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as file_obj:
        pickle.dump(data, file_obj)

    return target_path


def configure_torch_backends(torch_module: Any) -> None:
    """Apply the repo's preferred CUDA backend settings."""

    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.backends.cudnn.deterministic = False
    torch_module.backends.cudnn.benchmark = False


def ensure_min_rsl_rl_version(
    *,
    distributed: bool,
    min_version: str = MIN_DISTRIBUTED_RSL_RL_VERSION,
) -> None:
    """Exit with a clear message when distributed training needs a newer RSL-RL."""

    if not distributed:
        return

    import importlib.metadata as metadata

    installed_version = metadata.version("rsl-rl-lib")
    if version.parse(installed_version) >= version.parse(min_version):
        return

    install_command = (
        [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={min_version}"]
        if platform.system() == "Windows"
        else ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={min_version}"]
    )
    raise SystemExit(
        "Please install the correct version of RSL-RL.\n"
        f"Existing version is: '{installed_version}' and required version is: '{min_version}'.\n"
        f"To install the correct version, run:\n\n\t{' '.join(install_command)}\n"
    )


def resolve_checkpoint_path(
    *,
    task_name: str,
    log_root_path: str | os.PathLike[str],
    load_run: str | None,
    load_checkpoint: str | None,
    checkpoint: str | None = None,
    use_pretrained_checkpoint: bool = False,
    algo_name: str | None = None,
) -> str | None:
    """Resolve an explicit file, directory, pretrained, or latest checkpoint path."""

    if use_pretrained_checkpoint:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

        return get_published_pretrained_checkpoint(_PRETRAINED_CHECKPOINT_NAMESPACE, task_name)

    if checkpoint:
        checkpoint_path = os.path.abspath(os.path.expanduser(os.fspath(checkpoint)))
        checkpoint_pattern = load_checkpoint
        if checkpoint_pattern is not None:
            normalized_pattern = os.path.abspath(os.path.expanduser(os.fspath(checkpoint_pattern)))
            if normalized_pattern == checkpoint_path:
                checkpoint_pattern = None

        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        if os.path.isdir(checkpoint_path):
            return _resolve_checkpoint_from_directory(
                checkpoint_path,
                load_run=load_run,
                load_checkpoint=checkpoint_pattern,
                algo_name=algo_name,
            )

        from isaaclab.utils.assets import retrieve_file_path

        retrieved_path = retrieve_file_path(checkpoint)
        if os.path.isdir(retrieved_path):
            return _resolve_checkpoint_from_directory(
                retrieved_path,
                load_run=load_run,
                load_checkpoint=checkpoint_pattern,
                algo_name=algo_name,
            )
        return retrieved_path

    from isaaclab_tasks.utils import get_checkpoint_path

    return get_checkpoint_path(os.fspath(log_root_path), load_run, load_checkpoint)


def _resolve_checkpoint_from_directory(
    directory: str | os.PathLike[str],
    *,
    load_run: str | None,
    load_checkpoint: str | None,
    algo_name: str | None,
) -> str:
    """Resolve the newest checkpoint from either a run directory or an experiment root."""

    directory_path = os.path.abspath(os.fspath(directory))
    checkpoint_pattern = load_checkpoint or r"model_.*\.pt"
    direct_files = [
        entry.name
        for entry in os.scandir(directory_path)
        if entry.is_file() and re.match(checkpoint_pattern, entry.name)
    ]
    if direct_files:
        direct_files.sort(key=lambda name: f"{name:0>15}")
        return os.path.join(directory_path, direct_files[-1])

    run_pattern = load_run or ".*"
    if run_pattern == ".*" and algo_name:
        run_pattern = rf"{re.escape(algo_name)}_.*"

    run_dirs = [
        entry.path for entry in os.scandir(directory_path) if entry.is_dir() and re.match(run_pattern, entry.name)
    ]
    run_dirs.sort()
    if not run_dirs:
        raise ValueError(
            f"No runs present in the directory: '{directory_path}' match: '{run_pattern}'."
        )

    run_path = run_dirs[-1]
    model_checkpoints = [
        entry.name
        for entry in os.scandir(run_path)
        if entry.is_file() and re.match(checkpoint_pattern, entry.name)
    ]
    if not model_checkpoints:
        raise ValueError(
            f"No checkpoints in the directory: '{run_path}' match '{checkpoint_pattern}'."
        )
    model_checkpoints.sort(key=lambda name: f"{name:0>15}")
    return os.path.join(run_path, model_checkpoints[-1])


def display_available() -> bool:
    """Return whether a local display server is available."""

    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def livestream_enabled(args: Any) -> bool:
    """Return whether livestream mode is enabled via CLI or environment."""

    livestream_arg = getattr(args, "livestream", -1)
    if livestream_arg is not None and livestream_arg >= 0:
        return livestream_arg > 0
    return int(os.environ.get("LIVESTREAM", 0)) > 0


def configure_safe_play_args(args: Any) -> None:
    """Force safe headless defaults when no display server is available."""

    if getattr(args, "force_gui", False):
        args.headless = False
        os.environ["HEADLESS"] = "0"
        return

    if display_available() or livestream_enabled(args):
        return

    if not getattr(args, "headless", False):
        print(
            "[WARN] No DISPLAY/WAYLAND display server detected. "
            "Forcing headless mode for stable play startup."
        )
        args.headless = True

    existing_kit_args = getattr(args, "kit_args", "") or ""
    for extension_name in _HEADLESS_DISABLE_EXTENSIONS:
        disable_token = f"--disable {extension_name}"
        if disable_token not in existing_kit_args:
            existing_kit_args = f"{existing_kit_args} {disable_token}".strip()
    args.kit_args = existing_kit_args
