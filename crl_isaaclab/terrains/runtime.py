from __future__ import annotations

import numpy as np


def resolve_env_terrain_names(terrain) -> np.ndarray | None:
    """Resolve per-environment terrain names from the active terrain generator."""
    if terrain is None:
        return None

    terrain_gen = getattr(terrain, "terrain_generator_class", None)
    if terrain_gen is None:
        terrain_gen = getattr(terrain, "terrain_generator", None)
    if terrain_gen is None:
        return None

    terrain_names = getattr(terrain_gen, "terrain_names", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    if terrain_names is None or terrain_levels is None or terrain_types is None:
        return None

    try:
        levels_np = np.asarray(terrain_levels.detach().cpu().numpy(), dtype=np.int64)
    except Exception:
        levels_np = np.asarray(terrain_levels, dtype=np.int64)
    try:
        types_np = np.asarray(terrain_types.detach().cpu().numpy(), dtype=np.int64)
    except Exception:
        types_np = np.asarray(terrain_types, dtype=np.int64)

    if levels_np.size == 0 or types_np.size == 0:
        return None

    names = np.asarray(terrain_names)
    if names.ndim == 3 and names.shape[-1] == 1:
        names = names[..., 0]
    if names.ndim != 2:
        return None

    max_level = int(levels_np.max())
    max_type = int(types_np.max())
    if max_level >= names.shape[0] or max_type >= names.shape[1]:
        return None

    return names[levels_np, types_np].astype(str)
