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
) -> str | None:
    """Resolve an explicit, pretrained, or latest checkpoint path."""

    if use_pretrained_checkpoint:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

        return get_published_pretrained_checkpoint(_PRETRAINED_CHECKPOINT_NAMESPACE, task_name)

    if checkpoint:
        from isaaclab.utils.assets import retrieve_file_path

        return retrieve_file_path(checkpoint)

    from isaaclab_tasks.utils import get_checkpoint_path

    return get_checkpoint_path(os.fspath(log_root_path), load_run, load_checkpoint)


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
