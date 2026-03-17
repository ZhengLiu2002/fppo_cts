# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.terrains.terrain_importer import TerrainImporter
import numpy as np

if TYPE_CHECKING:
    from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg


class CRLTerrainImporter(TerrainImporter):
    """Terrain importer that retains the runtime terrain generator instance."""

    terrain_prim_paths: list[str]

    terrain_origins: torch.Tensor | None

    env_origins: torch.Tensor

    @staticmethod
    def _build_curriculum_terrain_names(terrain_generator_cfg) -> np.ndarray | None:
        sub_terrains = getattr(terrain_generator_cfg, "sub_terrains", None)
        num_rows = getattr(terrain_generator_cfg, "num_rows", None)
        num_cols = getattr(terrain_generator_cfg, "num_cols", None)
        if not isinstance(sub_terrains, dict) or not sub_terrains or num_rows is None or num_cols is None:
            return None

        proportions = np.asarray(
            [float(getattr(sub_cfg, "proportion", 0.0)) for sub_cfg in sub_terrains.values()],
            dtype=np.float64,
        )
        total = float(proportions.sum())
        if total <= 0.0:
            return None
        proportions /= total
        cumsum = np.cumsum(proportions)
        terrain_names = np.empty((int(num_rows), int(num_cols)), dtype=object)
        sub_terrain_names = list(sub_terrains.keys())

        for col in range(int(num_cols)):
            threshold = col / float(num_cols) + 0.001
            matches = np.where(threshold < cumsum)[0]
            sub_index = int(matches.min()) if matches.size > 0 else len(sub_terrain_names) - 1
            terrain_names[:, col] = sub_terrain_names[sub_index]

        return terrain_names

    def __init__(self, cfg: TerrainImporterCfg):
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError(
                    "Input terrain type is 'generator' but no value provided for 'terrain_generator'."
                )
            terrain_generator_cls = self.cfg.terrain_generator.class_type
            self._terrain_generator_class = terrain_generator_cls(
                cfg=self.cfg.terrain_generator, device=self.device
            )
            if getattr(self._terrain_generator_class, "terrain_names", None) is None and getattr(
                self.cfg.terrain_generator, "curriculum", False
            ):
                self._terrain_generator_class.terrain_names = self._build_curriculum_terrain_names(
                    self.cfg.terrain_generator
                )
            self.import_mesh("terrain", self._terrain_generator_class.terrain_mesh)
            if self.cfg.use_terrain_origins:
                self.configure_env_origins(self._terrain_generator_class.terrain_origins)
            else:
                self.configure_env_origins()
            self._terrain_flat_patches = self._terrain_generator_class.flat_patches
        else:
            raise TypeError(f"CRL Terrain type only support generator, not {self.cfg.terrain_type}")
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    @property
    def terrain_generator_class(self):
        return self._terrain_generator_class

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        """Compute the origins of the environments in a grid based on configured spacing."""
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        # create a grid of origins
        num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
        num_cols = np.ceil(num_envs / num_rows)
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self.device),
            torch.arange(num_cols, device=self.device),
            indexing="ij",
        )
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        env_origins[:, 2] = 0.0
        return env_origins
