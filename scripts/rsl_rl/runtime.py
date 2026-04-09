"""Shared runtime helpers for Isaac Lab training and evaluation scripts.

The helpers in this module intentionally avoid importing Isaac Lab at module
import time so they stay lightweight and easy to unit-test.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import platform
import random
import re
import subprocess
import sys
from typing import Any, Mapping

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


def _json_safe(value: Any) -> Any:
    """Convert common config/runtime objects into JSON-serializable values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def write_json_artifact(
    file_path: str | os.PathLike[str],
    data: Mapping[str, Any] | dict[str, Any],
) -> Path:
    """Write a JSON artifact, creating parent directories if needed."""

    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(_json_safe(dict(data)), indent=2, sort_keys=True), encoding="utf-8")
    return target_path


def dump_pickle(file_path: str | os.PathLike[str], data: Any) -> Path:
    """Serialize data to a pickle file, creating parent directories if needed."""

    target_path = Path(file_path)
    if not str(target_path).endswith(".pkl"):
        target_path = Path(f"{target_path}.pkl")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as file_obj:
        pickle.dump(data, file_obj)

    return target_path


def capture_rng_state(torch_module: Any) -> dict[str, Any]:
    """Capture Python, NumPy, and torch RNG state for exact resume support."""

    state: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        pass

    try:
        state["torch"] = torch_module.get_rng_state()
        if torch_module.cuda.is_available():
            state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    except Exception:
        pass

    return state


def restore_rng_state(torch_module: Any, state: dict[str, Any] | None) -> None:
    """Restore previously captured RNG state when resuming training."""

    if not state:
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np

            np.random.set_state(numpy_state)
        except Exception:
            pass

    torch_state = state.get("torch")
    if torch_state is not None:
        try:
            torch_module.set_rng_state(torch_state)
        except Exception:
            pass

    torch_cuda_state = state.get("torch_cuda")
    if torch_cuda_state is not None and torch_module.cuda.is_available():
        try:
            torch_module.cuda.set_rng_state_all(torch_cuda_state)
        except Exception:
            pass


def configure_torch_backends(torch_module: Any) -> None:
    """Apply the repo's preferred CUDA backend settings."""

    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.backends.cudnn.deterministic = False
    torch_module.backends.cudnn.benchmark = False


def collect_git_metadata(repo_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Collect a lightweight git snapshot for run manifests."""

    root_path = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()

    def _run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root_path,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip()

    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status = _run_git("status", "--short")
    if commit is None:
        return {"available": False, "repo_root": str(root_path)}

    return {
        "available": True,
        "repo_root": str(root_path),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def iter_task_variant_candidates(task_name: str, *, variant: str) -> list[str]:
    """Return candidate task ids for a requested task variant.

    Example:
        ``Isaac-Galileo-CRL-Teacher-v0`` + ``variant="eval"`` yields
        ``Isaac-Galileo-CRL-Teacher-Eval-v0`` before falling back to the original
        task name.
    """

    normalized_variant = variant.strip().lower()
    variant_token = normalized_variant.capitalize()
    candidates = [task_name]

    if f"-{variant_token}-" not in task_name:
        if "-Play-" in task_name:
            candidates.insert(0, task_name.replace("-Play-", f"-{variant_token}-"))
        elif task_name.endswith("-v0"):
            candidates.insert(0, task_name.replace("-v0", f"-{variant_token}-v0"))

    if normalized_variant == "play" and "-Eval-" in task_name:
        candidates.insert(0, task_name.replace("-Eval-", "-Play-"))
    elif normalized_variant == "eval" and "-Play-" not in task_name and "-Eval-" not in task_name:
        if task_name.endswith("-v0"):
            candidates.insert(0, task_name.replace("-v0", "-Eval-v0"))

    deduped: list[str] = []
    for candidate in candidates:
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def resolve_task_variant(
    task_name: str,
    *,
    variant: str,
    registered_tasks: set[str] | list[str] | tuple[str, ...],
) -> str:
    """Resolve the best matching task id for a target variant."""

    registered = set(registered_tasks)
    for candidate in iter_task_variant_candidates(task_name, variant=variant):
        if candidate in registered:
            return candidate
    return task_name


def build_evaluation_output_path(
    log_dir: str | os.PathLike[str],
    task_name: str,
    checkpoint_path: str | os.PathLike[str],
    *,
    summary_tag: str | None = None,
    filename: str = "summary.json",
) -> Path:
    """Create a stable evaluation artifact path under the run directory."""

    task_slug = re.sub(r"[^a-zA-Z0-9]+", "_", task_name).strip("_").lower() or "task"
    checkpoint_slug = Path(checkpoint_path).stem
    output_dir = Path(log_dir) / "evaluations" / task_slug / checkpoint_slug
    if summary_tag:
        tag_slug = re.sub(r"[^a-zA-Z0-9]+", "_", summary_tag).strip("_").lower() or "tag"
        output_dir = output_dir / tag_slug
    return output_dir / filename


def build_run_manifest(
    *,
    stage: str,
    task_name: str,
    log_dir: str | os.PathLike[str],
    agent_cfg: Any,
    env_cfg: Any | None = None,
    args: Any | None = None,
    preset: Any | None = None,
    training_type: str | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    repo_root: str | os.PathLike[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, JSON-safe manifest describing a train/eval run."""

    algorithm_cfg = getattr(agent_cfg, "algorithm", None)
    scene_cfg = getattr(env_cfg, "scene", None) if env_cfg is not None else None

    manifest: dict[str, Any] = {
        "stage": stage,
        "task_name": task_name,
        "log_dir": str(Path(log_dir).resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "training_type": training_type,
        "checkpoint_path": (
            str(Path(checkpoint_path).resolve()) if checkpoint_path is not None else None
        ),
        "seed": getattr(agent_cfg, "seed", None),
        "device": getattr(agent_cfg, "device", None),
        "experiment_name": getattr(agent_cfg, "experiment_name", None),
        "run_name": getattr(agent_cfg, "run_name", None),
        "max_iterations": getattr(agent_cfg, "max_iterations", None),
        "num_steps_per_env": getattr(agent_cfg, "num_steps_per_env", None),
        "num_envs": getattr(scene_cfg, "num_envs", None),
        "algorithm": {
            "class_name": getattr(algorithm_cfg, "class_name", None),
        },
        "git": collect_git_metadata(repo_root),
    }
    if algorithm_cfg is not None and hasattr(algorithm_cfg, "to_dict"):
        algorithm_dict = algorithm_cfg.to_dict()
        manifest["algorithm"]["cfg_keys"] = sorted(str(key) for key in algorithm_dict.keys())

    if preset is not None:
        manifest["preset"] = {
            "name": getattr(preset, "name", None),
            "slug": getattr(preset, "slug", None),
            "path": str(getattr(preset, "path", "")),
            "source_chain": [str(path) for path in getattr(preset, "source_chain", ())],
            "meta": getattr(preset, "meta", {}),
        }

    if args is not None:
        manifest["cli_args"] = {key: _json_safe(value) for key, value in vars(args).items()}

    if extra:
        manifest["extra"] = _json_safe(extra)

    return _json_safe(manifest)


def write_run_manifest(
    log_dir: str | os.PathLike[str],
    manifest: Mapping[str, Any] | dict[str, Any],
    *,
    filename: str = "run_manifest.json",
) -> Path:
    """Persist a run manifest alongside params and evaluation artifacts."""

    return write_json_artifact(Path(log_dir) / "params" / filename, dict(manifest))


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
