"""Helpers for Galileo terrain profile mutations."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping


def restrict_terrain_generator_to_named_subterrains(
    terrain_generator,
    terrain_names: Iterable[str],
    *,
    proportions: Mapping[str, float] | None = None,
) -> None:
    """Keep only selected sub-terrains while preserving their public names."""

    selected_names = tuple(terrain_names)
    missing_names = [
        name for name in selected_names if name not in getattr(terrain_generator, "sub_terrains", {})
    ]
    if missing_names:
        available = ", ".join(sorted(getattr(terrain_generator, "sub_terrains", {}).keys()))
        missing = ", ".join(missing_names)
        raise ValueError(f"Unknown Galileo terrain(s): {missing}. Available: {available}")

    retained_subterrains = {}
    for name in selected_names:
        terrain_cfg = copy.deepcopy(terrain_generator.sub_terrains[name])
        terrain_cfg.proportion = float(proportions[name]) if proportions and name in proportions else 1.0
        retained_subterrains[name] = terrain_cfg

    terrain_generator.sub_terrains = retained_subterrains
