from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - only needed for TOML presets
        tomllib = None


_SUPPORTED_SUFFIXES = (".json", ".toml")


class ExperimentPresetError(ValueError):
    """Raised when an experiment preset cannot be resolved or applied."""


@dataclass(frozen=True)
class ExperimentPreset:
    """Resolved experiment preset with metadata and merged overrides."""

    name: str
    slug: str
    path: Path
    data: dict[str, Any]
    source_chain: tuple[Path, ...]

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self.data.get("meta", {}))

    @property
    def env_overrides(self) -> dict[str, Any]:
        return dict(self.data.get("env", {}))

    @property
    def agent_overrides(self) -> dict[str, Any]:
        return dict(self.data.get("agent", {}))


def repo_root() -> Path:
    """Return the repository root inferred from this module location."""

    return Path(__file__).resolve().parents[2]


def experiments_root(root: str | Path | None = None) -> Path:
    """Return the root directory that stores experiment presets."""

    base_root = Path(root).resolve() if root is not None else repo_root()
    return base_root / "experiments"


def available_experiment_presets(root: str | Path | None = None) -> list[dict[str, str]]:
    """Return available preset names, paths, and short descriptions."""

    presets_dir = experiments_root(root)
    if not presets_dir.exists():
        return []

    entries: list[dict[str, str]] = []
    for path in sorted(presets_dir.rglob("*")):
        if not path.is_file() or path.suffix not in _SUPPORTED_SUFFIXES:
            continue
        description = ""
        try:
            raw_data = _read_preset_file(path)
            meta = raw_data.get("meta", {}) if isinstance(raw_data, Mapping) else {}
            description = str(meta.get("description", "")).strip()
        except Exception:
            description = ""
        entries.append(
            {
                "name": _display_name_for_path(path, presets_dir, repo_root()),
                "path": str(path.resolve()),
                "description": description,
            }
        )
    return entries


def load_experiment_preset(
    *,
    selection: str | None = None,
    file_path: str | None = None,
    root: str | Path | None = None,
) -> ExperimentPreset | None:
    """Load and resolve an experiment preset from `experiments/` or an explicit file."""

    if not selection and not file_path:
        return None
    if selection and file_path:
        raise ExperimentPresetError("Please use either `--exp` or `--exp-file`, not both.")

    base_root = Path(root).resolve() if root is not None else repo_root()
    resolved_path = resolve_experiment_preset_path(selection or file_path or "", root=base_root)
    merged_data, source_chain = _load_preset_chain(resolved_path, root=base_root, stack=())
    presets_dir = experiments_root(base_root)
    name = _display_name_for_path(resolved_path, presets_dir, base_root)
    return ExperimentPreset(
        name=name,
        slug=_slugify(name),
        path=resolved_path,
        data=merged_data,
        source_chain=tuple(source_chain),
    )


def resolve_experiment_preset_path(selection: str, *, root: str | Path | None = None) -> Path:
    """Resolve a preset selection string to a concrete file path."""

    base_root = Path(root).resolve() if root is not None else repo_root()
    presets_dir = experiments_root(base_root)
    raw_path = Path(selection).expanduser()

    candidate_bases: list[Path] = []
    if raw_path.is_absolute():
        candidate_bases.append(raw_path)
    else:
        candidate_bases.extend(
            [
                presets_dir / raw_path,
                base_root / raw_path,
                Path.cwd() / raw_path,
            ]
        )

    for candidate in candidate_bases:
        resolved = _resolve_with_supported_suffix(candidate)
        if resolved is not None:
            return resolved

    suffix_hint = ", ".join(_SUPPORTED_SUFFIXES)
    raise ExperimentPresetError(
        f"Unable to find experiment preset '{selection}'. "
        f"Expected a file under `{presets_dir}` or an explicit path with one of: {suffix_hint}."
    )


def apply_experiment_preset(*, env_cfg: Any, agent_cfg: Any, preset: ExperimentPreset) -> None:
    """Apply resolved env/agent overrides to the live config objects."""

    if preset.agent_overrides:
        apply_overrides(agent_cfg, preset.agent_overrides, prefix="agent")
    if preset.env_overrides:
        apply_overrides(env_cfg, preset.env_overrides, prefix="env")


def apply_overrides(target: Any, overrides: Mapping[str, Any], *, prefix: str = "") -> None:
    """Recursively apply nested overrides to dict-like or attribute-based configs."""

    for key, value in overrides.items():
        key_path = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(target, dict):
            current_value = target.get(key)
            if isinstance(value, Mapping):
                if current_value is None:
                    target[key] = _clone_plain_data(value)
                elif isinstance(current_value, Mapping) or hasattr(current_value, "__dict__"):
                    apply_overrides(current_value, value, prefix=key_path)
                else:
                    raise ExperimentPresetError(
                        f"Cannot apply nested overrides to non-nested key `{key_path}`."
                    )
            else:
                target[key] = _coerce_override_value(current_value, value)
            continue

        if not hasattr(target, key):
            raise ExperimentPresetError(f"Unknown config key `{key_path}` in experiment preset.")

        current_value = getattr(target, key)
        if isinstance(value, Mapping):
            if current_value is None:
                raise ExperimentPresetError(
                    f"Cannot apply nested overrides to `None` at `{key_path}`."
                )
            if isinstance(current_value, Mapping) or hasattr(current_value, "__dict__"):
                apply_overrides(current_value, value, prefix=key_path)
            else:
                raise ExperimentPresetError(
                    f"Cannot apply nested overrides to non-nested key `{key_path}`."
                )
        else:
            setattr(target, key, _coerce_override_value(current_value, value))


def write_experiment_metadata(
    log_dir: str | Path,
    preset: ExperimentPreset,
    *,
    args: Any | None = None,
) -> Path:
    """Persist resolved preset information alongside training artifacts."""

    metadata = {
        "preset_name": preset.name,
        "preset_slug": preset.slug,
        "preset_path": str(preset.path),
        "source_chain": [str(path) for path in preset.source_chain],
        "meta": preset.meta,
        "overrides": {
            "env": preset.env_overrides,
            "agent": preset.agent_overrides,
        },
    }
    if args is not None:
        metadata["cli_args"] = {key: value for key, value in vars(args).items()}

    metadata_path = Path(log_dir) / "params" / "experiment.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _load_preset_chain(
    preset_path: Path,
    *,
    root: Path,
    stack: tuple[Path, ...],
) -> tuple[dict[str, Any], list[Path]]:
    resolved_path = preset_path.resolve()
    if resolved_path in stack:
        cycle = " -> ".join(str(path) for path in (*stack, resolved_path))
        raise ExperimentPresetError(f"Circular preset inheritance detected: {cycle}")

    raw_data = _read_preset_file(resolved_path)
    if not isinstance(raw_data, Mapping):
        raise ExperimentPresetError(f"Preset file `{resolved_path}` must contain a top-level object.")

    current_data = dict(raw_data)
    extends_value = current_data.pop("extends", [])
    if isinstance(extends_value, str):
        parent_specs = [extends_value]
    elif isinstance(extends_value, list):
        parent_specs = extends_value
    else:
        raise ExperimentPresetError(
            f"Preset `{resolved_path}` uses an invalid `extends` field: {extends_value!r}."
        )

    merged_data: dict[str, Any] = {}
    source_chain: list[Path] = []
    for parent_spec in parent_specs:
        parent_path = resolve_experiment_preset_path(parent_spec, root=root)
        parent_data, parent_sources = _load_preset_chain(
            parent_path,
            root=root,
            stack=(*stack, resolved_path),
        )
        merged_data = _deep_merge_dicts(merged_data, parent_data)
        source_chain.extend(parent_sources)

    merged_data = _deep_merge_dicts(merged_data, current_data)
    source_chain.append(resolved_path)
    return merged_data, _dedupe_paths(source_chain)


def _read_preset_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    if path.suffix == ".toml":
        if tomllib is None:
            raise ExperimentPresetError(
                f"Preset `{path}` is TOML, but neither `tomllib` nor `tomli` is available."
            )
        return tomllib.loads(text)
    raise ExperimentPresetError(f"Unsupported preset file extension: `{path.suffix}`.")


def _resolve_with_supported_suffix(candidate: Path) -> Path | None:
    if candidate.is_file() and candidate.suffix in _SUPPORTED_SUFFIXES:
        return candidate.resolve()
    if candidate.suffix:
        return None
    for suffix in _SUPPORTED_SUFFIXES:
        with_suffix = candidate.with_suffix(suffix)
        if with_suffix.is_file():
            return with_suffix.resolve()
    return None


def _display_name_for_path(path: Path, presets_dir: Path, base_root: Path) -> str:
    for base in (presets_dir, base_root):
        try:
            return path.resolve().relative_to(base.resolve()).with_suffix("").as_posix()
        except ValueError:
            continue
    return path.stem


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "preset"


def _clone_plain_data(data: Any) -> Any:
    if isinstance(data, Mapping):
        return {key: _clone_plain_data(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_clone_plain_data(item) for item in data]
    return data


def _coerce_override_value(current_value: Any, new_value: Any) -> Any:
    if isinstance(current_value, tuple) and isinstance(new_value, list):
        return tuple(new_value)
    if isinstance(current_value, set) and isinstance(new_value, list):
        return set(new_value)
    if isinstance(current_value, Path) and isinstance(new_value, str):
        return type(current_value)(new_value)
    return new_value


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {key: _clone_plain_data(value) for key, value in base.items()}
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = _clone_plain_data(value)
    return merged


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)
    return unique_paths


__all__ = [
    "ExperimentPreset",
    "ExperimentPresetError",
    "apply_experiment_preset",
    "apply_overrides",
    "available_experiment_presets",
    "experiments_root",
    "load_experiment_preset",
    "repo_root",
    "resolve_experiment_preset_path",
    "write_experiment_metadata",
]
