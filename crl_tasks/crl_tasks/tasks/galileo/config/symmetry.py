from __future__ import annotations

from typing import Literal

import torch

RunnerRole = Literal["teacher", "student"]
_GALILEO_AUGMENTATION_ENTRY = "galileo_teacher_left_right_augmentation"

_TEACHER_NUM_PROP = 48
_TEACHER_NUM_SCAN = 132
_TEACHER_NUM_PRIV_EXPLICIT = 5
_TEACHER_OBS_DIM = _TEACHER_NUM_PROP + _TEACHER_NUM_SCAN + _TEACHER_NUM_PRIV_EXPLICIT
_HEIGHT_SCAN_SHAPE = (11, 12)

_JOINT_PERM = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
_JOINT_SIGN = (-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def _batch_sign(values: torch.Tensor, pattern: tuple[float, ...]) -> torch.Tensor:
    return values.new_tensor(pattern).unsqueeze(0)


def _batch_index(values: torch.Tensor, indices: tuple[int, ...]) -> torch.Tensor:
    return torch.as_tensor(indices, device=values.device, dtype=torch.long)


def _mirror_joint_block(values: torch.Tensor, *, is_std: bool = False) -> torch.Tensor:
    mirrored = values.index_select(1, _batch_index(values, _JOINT_PERM))
    if is_std:
        return mirrored
    return mirrored * _batch_sign(values, _JOINT_SIGN)


def _mirror_height_scan(values: torch.Tensor) -> torch.Tensor:
    num_y, num_x = _HEIGHT_SCAN_SHAPE
    if values.shape[1] != num_y * num_x:
        raise ValueError(
            f"Expected Galileo height scan dim {num_y * num_x}, got {values.shape[1]}."
        )
    return values.reshape(values.shape[0], num_y, num_x).flip(1).reshape(values.shape[0], -1)


def _mirror_teacher_obs(obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim != 2:
        raise ValueError(f"Expected 2D observation batch, got shape {tuple(obs.shape)}.")
    if obs.shape[1] != _TEACHER_OBS_DIM:
        raise ValueError(
            f"Expected Galileo teacher observation dim {_TEACHER_OBS_DIM}, got {obs.shape[1]}."
        )

    mirrored = obs.clone()

    mirrored[:, 0:3] = obs[:, 0:3] * _batch_sign(obs[:, 0:3], (1.0, -1.0, 1.0))
    mirrored[:, 3:6] = obs[:, 3:6] * _batch_sign(obs[:, 3:6], (-1.0, 1.0, -1.0))
    mirrored[:, 6:9] = obs[:, 6:9] * _batch_sign(obs[:, 6:9], (1.0, -1.0, 1.0))
    mirrored[:, 9:21] = _mirror_joint_block(obs[:, 9:21])
    mirrored[:, 21:33] = _mirror_joint_block(obs[:, 21:33])
    mirrored[:, 33:45] = _mirror_joint_block(obs[:, 33:45])
    mirrored[:, 45:48] = obs[:, 45:48] * _batch_sign(obs[:, 45:48], (1.0, -1.0, -1.0))
    mirrored[:, 48:180] = _mirror_height_scan(obs[:, 48:180])
    mirrored[:, 180:183] = obs[:, 180:183] * _batch_sign(obs[:, 180:183], (1.0, -1.0, 1.0))
    mirrored[:, 183:185] = obs[:, 183:185]

    return mirrored


def _mirror_teacher_actions(actions: torch.Tensor, *, is_std: bool = False) -> torch.Tensor:
    if actions.ndim != 2:
        raise ValueError(f"Expected 2D action batch, got shape {tuple(actions.shape)}.")
    if actions.shape[1] != len(_JOINT_PERM):
        raise ValueError(f"Expected Galileo action dim {len(_JOINT_PERM)}, got {actions.shape[1]}.")
    return _mirror_joint_block(actions, is_std=is_std)


def galileo_teacher_left_right_augmentation(
    obs: torch.Tensor | None,
    actions: torch.Tensor | None,
    env=None,
    obs_type: str = "policy",
    is_std: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Duplicate a Galileo teacher batch with its left-right mirrored variant.

    The returned tensors always keep the original samples first, followed by the
    mirrored samples. This matches the PPO symmetry augmentation convention.
    """
    del env, obs_type

    obs_aug = None
    act_aug = None

    if obs is not None:
        mirrored_obs = _mirror_teacher_obs(obs)
        obs_aug = torch.cat((obs, mirrored_obs), dim=0)

    if actions is not None:
        mirrored_actions = _mirror_teacher_actions(actions, is_std=is_std)
        act_aug = torch.cat((actions, mirrored_actions), dim=0)

    return obs_aug, act_aug


def build_symmetry_cfg(role: RunnerRole, params: dict) -> dict | None:
    enabled = bool(params.get("enabled", False))
    if not enabled:
        return None
    if role != "teacher":
        raise ValueError("Galileo symmetry augmentation is currently implemented only for teacher training.")

    cfg = {
        key: value
        for key, value in params.items()
        if key not in {"enabled", "use_mirror_loss", "mirror_loss_coeff"}
    }
    cfg.setdefault("use_data_augmentation", True)
    cfg["data_augmentation_func"] = _GALILEO_AUGMENTATION_ENTRY
    return cfg


__all__ = [
    "build_symmetry_cfg",
    "galileo_teacher_left_right_augmentation",
]
