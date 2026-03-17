from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from prettytable import PrettyTable
from isaaclab.managers import RewardManager
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg

if TYPE_CHECKING:
    from crl_isaaclab.envs import CRLManagerBasedRLEnv


class CRLRewardManager(RewardManager):
    _env: CRLManagerBasedRLEnv
    _manager_setting_names = {"only_positive_rewards"}
    _manager_display_name = "Reward"

    def __init__(self, cfg: object, env: CRLManagerBasedRLEnv):
        if isinstance(cfg, dict):
            self.only_positive_rewards = bool(cfg.get("only_positive_rewards", False))
        else:
            self.only_positive_rewards = bool(getattr(cfg, "only_positive_rewards", False))
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Return a compact table that hides disabled terms."""
        display_terms = list(self._iter_nonzero_terms())
        msg = (
            f"<{self._manager_display_name}Manager> contains {len(display_terms)} active terms "
            "(weight != 0).\n"
        )

        table = PrettyTable()
        table.title = f"Active {self._manager_display_name} Terms"
        table.field_names = ["Index", "Name", "Weight"]
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        for display_index, (name, term_cfg, _) in enumerate(display_terms):
            table.add_row([display_index, name, term_cfg.weight])
        msg += table.get_string()
        msg += "\n"
        return msg

    def compute(self, dt: float) -> torch.Tensor:
        """
        Same to Legged Gym
        """
        # reset computation
        self._reward_buf[:] = 0.0
        # iterate over all the reward terms
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            # skip if weight is zero (kind of a micro-optimization)
            if term_cfg.weight == 0.0:
                self._step_reward[:, term_idx] = 0.0
                continue
            # compute term's value
            value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
            # update total reward
            self._reward_buf += value
            # update episodic sum
            self._episode_sums[name] += value
            # Update current reward for this step.
            self._step_reward[:, term_idx] = value / dt
        if self.only_positive_rewards:
            self._reward_buf.clamp_(min=0.0)
        return self._reward_buf

    def reset(self, env_ids=None) -> dict[str, torch.Tensor]:
        """Return episodic term logs while hiding disabled terms from logger output."""
        if env_ids is None:
            env_ids = slice(None)

        extras: dict[str, torch.Tensor] = {}
        for name, _, _ in self._iter_nonzero_terms():
            episodic_sum_avg = torch.mean(self._episode_sums[name][env_ids])
            extras[f"Episode_{self._manager_display_name}/{name}"] = (
                episodic_sum_avg / self._env.max_episode_length_s
            )

        for term_name in self._episode_sums.keys():
            self._episode_sums[term_name][env_ids] = 0.0

        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)

        return extras

    def get_active_iterable_terms(self, env_idx: int):
        """Return only non-zero terms for live displays."""
        terms = []
        for name, _, term_idx in self._iter_nonzero_terms():
            terms.append((name, [self._step_reward[env_idx, term_idx].cpu().item()]))
        return terms

    def _prepare_terms(self):
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_name in self._manager_setting_names:
                continue
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, RewardTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type RewardTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            if not isinstance(term_cfg.weight, (float, int)):
                raise TypeError(
                    f"Weight for the term '{term_name}' is not of type float or int."
                    f" Received: '{type(term_cfg.weight)}'."
                )
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)

    def _iter_nonzero_terms(self):
        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            try:
                weight = float(term_cfg.weight)
            except Exception:
                weight = 0.0
            if weight == 0.0:
                continue
            yield name, term_cfg, term_idx


class CRLCostManager(CRLRewardManager):
    """Cost manager variant with cost-specific logging labels."""

    _manager_display_name = "Cost"
