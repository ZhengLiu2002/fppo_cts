from __future__ import annotations

import copy
from typing import Optional, Callable, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .feature_extractors.state_encoder import *
from rsl_rl.utils import resolve_nn_activation

# Simple registry to avoid eval-based class lookup.
ACTOR_REGISTRY: Dict[str, Callable] = {}


def _normalize_scan_shape(shape) -> tuple[int, int] | None:
    if shape is None:
        return None
    if len(shape) != 2:
        raise ValueError(f"Scan shape must have exactly two dimensions, got {shape}.")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Scan shape must be positive, got {shape}.")
    return height, width


class CoordConvHeightScanEncoder(nn.Module):
    """Encode a flattened height-scan grid with CoordConv before MLP fusion."""

    def __init__(
        self,
        scan_shape: tuple[int, int],
        output_dim: int,
        activation: nn.Module,
        hidden_channels: tuple[int, int] | None = None,
        add_radius_channel: bool = True,
    ) -> None:
        super().__init__()
        self.scan_shape = _normalize_scan_shape(scan_shape)
        if self.scan_shape is None:
            raise ValueError("CoordConvHeightScanEncoder requires a concrete scan shape.")
        self.scan_dim = int(self.scan_shape[0] * self.scan_shape[1])
        self.output_dim = int(output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"CoordConv output_dim must be positive, got {output_dim}.")

        channels = hidden_channels or (32, 64)
        hidden0 = max(int(channels[0]), 8)
        hidden1 = max(int(channels[1]), hidden0)
        pooled_h = min(4, self.scan_shape[0])
        pooled_w = min(4, self.scan_shape[1])

        coord_channels = 2 + int(add_radius_channel)
        self.add_radius_channel = bool(add_radius_channel)
        self.conv = nn.Sequential(
            nn.Conv2d(1 + coord_channels, hidden0, kernel_size=3, padding=1),
            copy.deepcopy(activation),
            nn.Conv2d(hidden0, hidden1, kernel_size=3, padding=1),
            copy.deepcopy(activation),
            nn.Conv2d(hidden1, hidden1, kernel_size=3, padding=1),
            copy.deepcopy(activation),
            nn.AdaptiveAvgPool2d((pooled_h, pooled_w)),
            nn.Flatten(),
            nn.Linear(hidden1 * pooled_h * pooled_w, self.output_dim),
            nn.Tanh(),
        )

        ys = torch.linspace(-1.0, 1.0, self.scan_shape[0], dtype=torch.float32)
        xs = torch.linspace(-1.0, 1.0, self.scan_shape[1], dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coord_tensors = [xx, yy]
        if self.add_radius_channel:
            coord_tensors.append(torch.sqrt(torch.clamp(xx.square() + yy.square(), min=0.0)))
        coord_map = torch.stack(coord_tensors, dim=0)
        self.register_buffer("_coord_map", coord_map, persistent=False)

    def forward(self, scan: torch.Tensor) -> torch.Tensor:
        if scan.ndim != 2:
            raise ValueError(
                f"CoordConvHeightScanEncoder expects a 2D tensor [batch, scan_dim], got {scan.shape}."
            )
        if scan.shape[1] != self.scan_dim:
            raise ValueError(
                f"CoordConvHeightScanEncoder expected scan_dim={self.scan_dim}, got {scan.shape[1]}."
            )
        batch_size = scan.shape[0]
        scan_grid = scan.view(batch_size, 1, self.scan_shape[0], self.scan_shape[1])
        coord_map = self._coord_map.to(device=scan_grid.device, dtype=scan_grid.dtype).expand(
            batch_size, -1, -1, -1
        )
        return self.conv(torch.cat([scan_grid, coord_map], dim=1))


def _build_scan_encoder_module(
    scan_dim: int,
    encoder_dims: list[int] | None,
    activation: nn.Module,
    *,
    encoder_type: str = "mlp",
    scan_grid_shape: tuple[int, int] | None = None,
) -> tuple[nn.Module, int]:
    if scan_dim <= 0:
        return nn.Identity(), 0
    if encoder_dims is None or len(encoder_dims) == 0:
        return nn.Identity(), scan_dim

    encoder_type = str(encoder_type).strip().lower()
    if encoder_type == "coord_conv":
        scan_shape = _normalize_scan_shape(scan_grid_shape)
        if scan_shape is None:
            raise ValueError("CoordConv scan encoder requires `scan_grid_shape` to be set.")
        if scan_shape[0] * scan_shape[1] != scan_dim:
            raise ValueError(
                f"CoordConv scan shape {scan_shape} does not match scan_dim={scan_dim}."
            )
        hidden_channels = None
        if len(encoder_dims) >= 3:
            hidden_channels = (int(encoder_dims[0]), int(encoder_dims[1]))
        elif len(encoder_dims) == 2:
            hidden_channels = (int(encoder_dims[0]), int(encoder_dims[0]))
        output_dim = int(encoder_dims[-1])
        return (
            CoordConvHeightScanEncoder(
                scan_shape=scan_shape,
                output_dim=output_dim,
                activation=activation,
                hidden_channels=hidden_channels,
            ),
            output_dim,
        )
    if encoder_type != "mlp":
        raise ValueError(f"Unsupported scan encoder type: {encoder_type}.")

    scan_encoder = []
    scan_encoder.append(nn.Linear(scan_dim, encoder_dims[0]))
    scan_encoder.append(copy.deepcopy(activation))
    for layer_index in range(len(encoder_dims) - 1):
        if layer_index == len(encoder_dims) - 2:
            scan_encoder.append(nn.Linear(encoder_dims[layer_index], encoder_dims[layer_index + 1]))
            scan_encoder.append(nn.Tanh())
        else:
            scan_encoder.append(nn.Linear(encoder_dims[layer_index], encoder_dims[layer_index + 1]))
            scan_encoder.append(copy.deepcopy(activation))
    return nn.Sequential(*scan_encoder), int(encoder_dims[-1])


def register_actor(name: str):
    """注册 Actor 子类，避免使用 eval 动态查找。"""

    def decorator(cls):
        ACTOR_REGISTRY[name] = cls
        return cls

    return decorator


@register_actor("Actor")
class Actor(nn.Module):
    """基础 Actor：融合本体观测、激光、特权隐变量与历史编码。"""

    def __init__(
        self,
        num_actions,
        scan_encoder_dims,
        actor_hidden_dims,
        priv_encoder_dims,
        activation,
        tanh_encoder_output=False,
        **kwargs,
    ) -> None:
        super().__init__()
        # prop -> scan -> priv_explicit -> priv_latent -> hist
        # actor input: prop -> scan -> priv_explicit -> latent
        self.num_prop = num_prop = kwargs.pop("num_prop")
        self.num_scan = num_scan = kwargs.pop("num_scan")
        self.num_hist = num_hist = kwargs.pop("num_hist")
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent = kwargs.pop("num_priv_latent")
        self.num_priv_explicit = num_priv_explicit = kwargs.pop("num_priv_explicit")
        self.normalize_latent = bool(kwargs.pop("normalize_latent", False))
        history_latent_dim = int(kwargs.pop("history_latent_dim", num_priv_latent) or 0)
        self.history_latent_dim = history_latent_dim if history_latent_dim > 0 else num_priv_latent
        self.history_reconstruction_dim = int(kwargs.pop("history_reconstruction_dim", 0) or 0)
        self.velocity_estimation_dim = int(kwargs.pop("velocity_estimation_dim", 0) or 0)
        self.velocity_estimator_channel_size = int(
            kwargs.pop("velocity_estimator_channel_size", 0) or 0
        )
        self.scan_encoder_type = str(kwargs.pop("scan_encoder_type", "mlp")).strip().lower()
        self.scan_grid_shape = _normalize_scan_shape(kwargs.pop("scan_grid_shape", None))
        self.priv_terrain_scan_start = int(kwargs.pop("priv_terrain_scan_start", 0) or 0)
        self.priv_terrain_scan_dim = int(kwargs.pop("priv_terrain_scan_dim", 0) or 0)
        self.priv_terrain_scan_shape = _normalize_scan_shape(
            kwargs.pop("priv_terrain_scan_shape", None)
        )
        default_priv_terrain_dim = self.history_latent_dim if self.history_latent_dim > 0 else 0
        self.priv_terrain_encoder_output_dim = int(
            kwargs.pop("priv_terrain_encoder_output_dim", default_priv_terrain_dim)
            or default_priv_terrain_dim
        )
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        # Raw observation layout: current proprio + optional scan / explicit priv + optional raw priv latent + history.
        self.in_features = (
            num_prop + num_scan + num_priv_latent + num_priv_explicit + num_prop * num_hist
        )
        self.student_in_features = num_prop + num_scan + num_prop * num_hist
        self.priv_terrain_encoder: nn.Module | None = None
        self._priv_encoder_input_dim = num_priv_latent

        if self.priv_terrain_scan_dim > 0:
            if self.priv_terrain_scan_shape is None:
                raise ValueError(
                    "Privileged terrain CoordConv requires `priv_terrain_scan_shape` to be set."
                )
            if (
                self.priv_terrain_scan_shape[0] * self.priv_terrain_scan_shape[1]
                != self.priv_terrain_scan_dim
            ):
                raise ValueError(
                    "Privileged terrain scan shape does not match the configured scan dimension."
                )
            if not (
                0 <= self.priv_terrain_scan_start
                and self.priv_terrain_scan_start + self.priv_terrain_scan_dim <= num_priv_latent
            ):
                raise ValueError(
                    "Privileged terrain scan slice lies outside the privileged latent block."
                )
            if self.priv_terrain_encoder_output_dim <= 0:
                raise ValueError(
                    "Privileged terrain CoordConv requires a positive output dimension."
                )
            self.priv_terrain_encoder = CoordConvHeightScanEncoder(
                scan_shape=self.priv_terrain_scan_shape,
                output_dim=self.priv_terrain_encoder_output_dim,
                activation=activation,
            )
            self._priv_encoder_input_dim = (
                num_priv_latent - self.priv_terrain_scan_dim + self.priv_terrain_encoder_output_dim
            )

        if len(priv_encoder_dims) > 0 and self._priv_encoder_input_dim > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(
                nn.Linear(self._priv_encoder_input_dim, priv_encoder_dims[0])
            )
            priv_encoder_layers.append(copy.deepcopy(activation))
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(
                    nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1])
                )
                priv_encoder_layers.append(copy.deepcopy(activation))
            if priv_encoder_dims[-1] != self.history_latent_dim:
                priv_encoder_layers.append(
                    nn.Linear(priv_encoder_dims[-1], self.history_latent_dim)
                )
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
        elif (
            self._priv_encoder_input_dim > 0
            and self._priv_encoder_input_dim != self.history_latent_dim
        ):
            self.priv_encoder = nn.Sequential(
                nn.Linear(self._priv_encoder_input_dim, self.history_latent_dim),
                copy.deepcopy(activation),
            )
        else:
            self.priv_encoder = nn.Identity()
        latent_dim = self.history_latent_dim

        state_history_encoder_cfg = kwargs.pop("state_history_encoder")
        if num_hist > 0 and latent_dim > 0:
            state_histroy_encoder_class = eval(state_history_encoder_cfg.pop("class_name"))
            self.history_encoder: StateHistoryEncoder = state_histroy_encoder_class(
                activation,
                num_prop,
                num_hist,
                latent_dim,
                state_history_encoder_cfg.pop("channel_size"),
            )
        else:
            self.history_encoder = nn.Identity()
        if num_hist > 0 and self.velocity_estimation_dim > 0:
            velocity_channel_size = max(
                self.velocity_estimator_channel_size,
                self.velocity_estimation_dim,
                32,
            )
            self.velocity_estimator = TCNVelocityEstimator(
                activation,
                num_prop,
                num_hist,
                self.velocity_estimation_dim,
                velocity_channel_size,
            )
        else:
            self.velocity_estimator = None
        self._hist_latent_dim = latent_dim
        if self.if_scan_encode:
            self.scan_encoder, self.scan_encoder_output_dim = _build_scan_encoder_module(
                num_scan,
                scan_encoder_dims,
                activation,
                encoder_type=self.scan_encoder_type,
                scan_grid_shape=self.scan_grid_shape,
            )
        else:
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        # 主干 MLP：拼接 (prop + scan_latent + priv_explicit + priv_latent)
        actor_layers = []
        actor_layers.append(
            nn.Linear(
                num_prop
                + self.scan_encoder_output_dim
                + num_priv_explicit
                + self.velocity_estimation_dim
                + latent_dim,
                actor_hidden_dims[0],
            )
        )
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)
        if latent_dim > 0 and self.history_reconstruction_dim > 0:
            recon_hidden_dim = max(latent_dim * 2, 64)
            self.history_reconstruction_head = nn.Sequential(
                nn.Linear(latent_dim, recon_hidden_dim),
                activation,
                nn.Linear(recon_hidden_dim, self.history_reconstruction_dim),
            )
        else:
            self.history_reconstruction_head = None

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.normalize_latent or latent.shape[-1] <= 0:
            return latent
        return F.normalize(latent, p=2, dim=-1, eps=1.0e-6)

    def _expand_student_observation(self, obs: torch.Tensor) -> torch.Tensor:
        if (
            self.num_hist <= 0
            or self.in_features <= self.student_in_features
            or obs.shape[1] != self.student_in_features
        ):
            return obs
        middle_dim = self.in_features - self.student_in_features
        prop = obs[:, : self.num_prop + self.num_scan]
        history = obs[:, self.num_prop + self.num_scan :]
        zeros = obs.new_zeros((obs.shape[0], middle_dim))
        return torch.cat([prop, zeros, history], dim=1)

    def _ensure_full_observation(self, obs: torch.Tensor, hist_encoding: bool) -> torch.Tensor:
        if hist_encoding:
            return self._expand_student_observation(obs)
        return obs

    def _compose_backbone_input(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        scandots_latent: Optional[torch.Tensor] = None,
        velocity_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs = self._ensure_full_observation(
            obs, hist_encoding=latent.shape[-1] == self._hist_latent_dim
        )
        if self.if_scan_encode:
            obs_scan = obs[:, self.num_prop : self.num_prop + self.num_scan]
            if scandots_latent is None:
                scan_latent = self.scan_encoder(obs_scan)
            else:
                scan_latent = scandots_latent
            obs_prop_scan = torch.cat([obs[:, : self.num_prop], scan_latent], dim=1)
        else:
            obs_prop_scan = obs[:, : self.num_prop + self.num_scan]
        obs_priv_explicit = obs[
            :,
            self.num_prop + self.num_scan : self.num_prop + self.num_scan + self.num_priv_explicit,
        ]
        if velocity_feature is None:
            velocity_feature = obs.new_zeros((obs.shape[0], self.velocity_estimation_dim))
        return torch.cat([obs_prop_scan, obs_priv_explicit, velocity_feature, latent], dim=1)

    def forward_with_latent(
        self,
        obs: torch.Tensor,
        latent: torch.Tensor,
        scandots_latent: Optional[torch.Tensor] = None,
        velocity_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        backbone_input = self._compose_backbone_input(
            obs,
            latent,
            scandots_latent,
            velocity_feature=velocity_feature,
        )
        return self.actor_backbone(backbone_input)

    def forward(self, obs, hist_encoding: bool, scandots_latent: Optional[torch.Tensor] = None):
        """前向推理：
        - 可选对激光编码（或复用外部提供的 scandots_latent）
        - 可选使用历史编码器，否则只编码特权隐式
        """
        obs = self._ensure_full_observation(obs, hist_encoding=hist_encoding)
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        velocity_feature = self.resolve_velocity_feature(obs, hist_encoding=hist_encoding)
        return self.forward_with_latent(
            obs,
            latent,
            scandots_latent,
            velocity_feature=velocity_feature,
        )

    def infer_priv_latent(self, obs):
        """仅编码特权隐式变量。"""
        obs = self._ensure_full_observation(obs, hist_encoding=False)
        if self.num_priv_latent <= 0 or self._hist_latent_dim <= 0:
            return obs.new_zeros((obs.shape[0], self._hist_latent_dim))
        priv = obs[
            :,
            self.num_prop
            + self.num_scan
            + self.num_priv_explicit : self.num_prop
            + self.num_scan
            + self.num_priv_explicit
            + self.num_priv_latent,
        ]
        if self.priv_terrain_encoder is not None:
            scan_start = self.priv_terrain_scan_start
            scan_end = scan_start + self.priv_terrain_scan_dim
            terrain_scan = priv[:, scan_start:scan_end]
            terrain_latent = self.priv_terrain_encoder(terrain_scan)
            priv = torch.cat([priv[:, :scan_start], terrain_latent, priv[:, scan_end:]], dim=1)
        return self._normalize_latent(self.priv_encoder(priv))

    def infer_hist_latent(self, obs):
        """对历史本体序列做 1D 卷积编码。"""
        obs = self._ensure_full_observation(obs, hist_encoding=True)
        if self.num_hist <= 0 or self._hist_latent_dim <= 0:
            return self.infer_priv_latent(obs)
        hist = obs[:, -self.num_hist * self.num_prop :]
        latent = self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
        return self._normalize_latent(latent)

    def infer_scandots_latent(self, obs):
        """仅对激光点做编码。"""
        scan = obs[:, self.num_prop : self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

    def infer_velocity_estimate(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._ensure_full_observation(obs, hist_encoding=True)
        if (
            self.velocity_estimation_dim <= 0
            or self.velocity_estimator is None
            or self.num_hist <= 0
        ):
            return obs.new_zeros((obs.shape[0], self.velocity_estimation_dim))
        hist = obs[:, -self.num_hist * self.num_prop :]
        return self.velocity_estimator(hist.view(-1, self.num_hist, self.num_prop))

    def _extract_velocity_components(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.velocity_estimation_dim <= 0:
            return tensor.new_zeros((tensor.shape[0], 0))
        if self.velocity_estimation_dim == 3 and tensor.shape[1] >= 6:
            # Regress planar velocity plus yaw rate: vx, vy, wz.
            return torch.cat([tensor[:, 0:2], tensor[:, 5:6]], dim=1)
        return tensor[:, : self.velocity_estimation_dim]

    def extract_teacher_velocity(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self._ensure_full_observation(obs, hist_encoding=False)
        if self.velocity_estimation_dim <= 0:
            return obs.new_zeros((obs.shape[0], 0))
        start = self.num_prop + self.num_scan + self.num_priv_explicit
        required_priv_dim = 6 if self.velocity_estimation_dim == 3 else self.velocity_estimation_dim
        end = start + required_priv_dim
        if self.num_priv_latent >= required_priv_dim and obs.shape[1] >= end:
            return self._extract_velocity_components(obs[:, start:end])
        return self.infer_velocity_estimate(obs)

    def resolve_velocity_feature(self, obs: torch.Tensor, hist_encoding: bool) -> torch.Tensor:
        if hist_encoding:
            return self.infer_velocity_estimate(obs)
        return self.extract_teacher_velocity(obs)

    def reconstruct_privileged_from_history(self, obs):
        if self.history_reconstruction_head is None:
            return None
        latent = self.infer_hist_latent(obs)
        return self.history_reconstruction_head(latent)


class ActorCriticRMA(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        cost_critic_hidden_dims=None,
        num_cost_heads: int = 1,
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super(ActorCriticRMA, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims = kwargs["priv_encoder_dims"]
        scan_encoder_dims = kwargs["scan_encoder_dims"]
        activation = resolve_nn_activation(activation)
        actor_cfg = kwargs.pop("actor")
        for key in (
            "scan_encoder_type",
            "scan_grid_shape",
            "priv_terrain_scan_start",
            "priv_terrain_scan_dim",
            "priv_terrain_scan_shape",
            "priv_terrain_encoder_output_dim",
        ):
            if key in kwargs and key not in actor_cfg:
                actor_cfg[key] = kwargs[key]
        actor_class_name = actor_cfg.pop("class_name")
        if actor_class_name not in ACTOR_REGISTRY:
            raise KeyError(f"Actor class '{actor_class_name}' not registered.")
        actor_class = ACTOR_REGISTRY[actor_class_name]
        self.actor: Actor = actor_class(
            num_actions,
            scan_encoder_dims,
            actor_hidden_dims,
            priv_encoder_dims,
            activation,
            tanh_encoder_output=kwargs["tanh_encoder_output"],
            **actor_cfg,
        )
        self._encode_scan_for_critic = bool(kwargs.get("encode_scan_for_critic", False))
        self._num_prop = int(kwargs.get("num_prop", 0))
        self._num_scan = int(kwargs.get("num_scan", 0))
        self._num_priv_explicit = int(kwargs.get("num_priv_explicit", 0))
        self._num_priv_latent = int(kwargs.get("num_priv_latent", 0))
        self._history_latent_dim = int(kwargs.get("history_latent_dim", 0) or 0)
        self._history_reconstruction_mode = str(
            kwargs.get("history_reconstruction_mode", "hidden_privileged")
        )
        self._critic_num_prop = int(kwargs.get("critic_num_prop", self._num_prop) or self._num_prop)
        self._critic_num_scan = int(kwargs.get("critic_num_scan", self._num_scan) or self._num_scan)
        self._critic_num_priv_explicit = int(
            kwargs.get("critic_num_priv_explicit", self._num_priv_explicit)
            or self._num_priv_explicit
        )
        self._critic_num_priv_latent = int(
            kwargs.get("critic_num_priv_latent", self._num_priv_latent) or self._num_priv_latent
        )
        self._critic_num_hist = int(kwargs.get("critic_num_hist", 0) or 0)
        self._critic_use_latent = bool(kwargs.get("critic_use_latent", False))
        critic_scan_encoder_dims = kwargs.get("critic_scan_encoder_dims", None)
        self._scan_encoder_type = str(kwargs.get("scan_encoder_type", "mlp")).strip().lower()
        self._scan_grid_shape = _normalize_scan_shape(kwargs.get("scan_grid_shape", None))
        self._critic_scan_encoder = None
        self._critic_scan_encoder_output_dim = self._critic_num_scan
        if self._encode_scan_for_critic and self._critic_num_scan > 0 and critic_scan_encoder_dims:
            self._critic_scan_encoder, self._critic_scan_encoder_output_dim = (
                _build_scan_encoder_module(
                    self._critic_num_scan,
                    critic_scan_encoder_dims,
                    activation,
                    encoder_type=self._scan_encoder_type,
                    scan_grid_shape=self._scan_grid_shape,
                )
            )
        elif self._encode_scan_for_critic and self._critic_num_scan > 0:
            self._critic_scan_encoder_output_dim = self.actor.scan_encoder_output_dim
        self._reconstruction_missing_prop = max(self._critic_num_prop - self._num_prop, 0)
        self._full_history_reconstruction_dim = (
            self._reconstruction_missing_prop
            + self._critic_num_scan
            + self._critic_num_priv_explicit
            + self._critic_num_priv_latent
        )
        if self._history_reconstruction_mode == "hidden_privileged":
            self._history_reconstruction_dim = self._full_history_reconstruction_dim
        elif self._history_reconstruction_mode == "base_lin_vel":
            self._history_reconstruction_dim = min(3, self._critic_num_prop)
        else:
            raise ValueError(
                "Unsupported history reconstruction mode: " f"{self._history_reconstruction_mode}."
            )
        actor_reconstruction_dim = int(getattr(self.actor, "history_reconstruction_dim", 0) or 0)
        if (
            actor_reconstruction_dim > 0
            and self._history_reconstruction_dim > 0
            and actor_reconstruction_dim != self._history_reconstruction_dim
        ):
            raise ValueError(
                "Actor reconstruction head dimension does not match target dimension: "
                f"{actor_reconstruction_dim} != {self._history_reconstruction_dim}."
            )

        # Critic：直接接收 num_critic_obs（可能含特权信息）
        if self._encode_scan_for_critic and self._critic_num_scan > 0:
            mlp_input_dim_c = (
                num_critic_obs - self._critic_num_scan + self._critic_scan_encoder_output_dim
            )
        else:
            mlp_input_dim_c = num_critic_obs
        if self._critic_use_latent and self._history_latent_dim > 0:
            mlp_input_dim_c += self._history_latent_dim
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1])
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Cost critic (decoupled from reward critic)
        self.num_cost_heads = max(int(num_cost_heads), 1)
        cost_critic_hidden_dims = cost_critic_hidden_dims or critic_hidden_dims
        cost_critic_layers = []
        cost_critic_layers.append(nn.Linear(mlp_input_dim_c, cost_critic_hidden_dims[0]))
        cost_critic_layers.append(activation)
        for layer_index in range(len(cost_critic_hidden_dims)):
            if layer_index == len(cost_critic_hidden_dims) - 1:
                cost_critic_layers.append(
                    nn.Linear(cost_critic_hidden_dims[layer_index], self.num_cost_heads)
                )
            else:
                cost_critic_layers.append(
                    nn.Linear(
                        cost_critic_hidden_dims[layer_index],
                        cost_critic_hidden_dims[layer_index + 1],
                    )
                )
                cost_critic_layers.append(activation)
        self.cost_critic = nn.Sequential(*cost_critic_layers)

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        print(f"Cost Critic MLP: {self.cost_critic}")

        self.noise_std_type = noise_std_type
        # Keep exploration recoverable: avoid hard min clamp on std that can zero out gradients.
        self._std_floor = float(kwargs.get("std_floor", 2.0e-3))
        self._std_ceiling = float(kwargs.get("std_ceiling", 2.0))
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

        self.distribution = None
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, hist_encoding):
        """用最新 actor 输出均值并配合可学习方差构造高斯分布。"""
        mean = self.actor(observations, hist_encoding)
        self.update_distribution_from_mean(mean)

    def update_distribution_from_mean(self, mean: torch.Tensor) -> None:
        # sanitize mean to prevent NaN/Inf blowing up Normal
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        if self.noise_std_type == "scalar":
            std = mean * 0.0 + F.softplus(self.std)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )
        # Avoid dead-zone at lower bound: add a floor instead of hard min clamp.
        std = torch.nan_to_num(std, nan=1.0, posinf=self._std_ceiling, neginf=1.0)
        std = torch.clamp(std, max=self._std_ceiling) + self._std_floor
        self.distribution = Normal(mean, std)

    def update_distribution_with_latent(
        self,
        observations: torch.Tensor,
        latent: torch.Tensor,
        scandots_latent: Optional[torch.Tensor] = None,
        velocity_feature: Optional[torch.Tensor] = None,
    ) -> None:
        mean = self.actor.forward_with_latent(
            observations,
            latent,
            scandots_latent,
            velocity_feature=velocity_feature,
        )
        self.update_distribution_from_mean(mean)

    def _resolve_actor_mode_latent(
        self,
        observations: torch.Tensor,
        actor_mode: str,
        *,
        detach_student_encoder: bool = False,
    ) -> torch.Tensor:
        if actor_mode == "teacher":
            return self.actor.infer_priv_latent(observations)
        if actor_mode == "student":
            latent = self.actor.infer_hist_latent(observations)
            return latent.detach() if detach_student_encoder else latent
        raise ValueError(f"Unsupported actor mode: {actor_mode}")

    def actor_mode_latent(
        self,
        observations: torch.Tensor,
        actor_mode: str,
        *,
        detach_student_encoder: bool = False,
    ) -> torch.Tensor:
        return self._resolve_actor_mode_latent(
            observations,
            actor_mode,
            detach_student_encoder=detach_student_encoder,
        )

    def actor_mode_velocity_feature(
        self,
        observations: torch.Tensor,
        actor_mode: str,
    ) -> torch.Tensor:
        if actor_mode == "teacher":
            return self.actor.extract_teacher_velocity(observations)
        if actor_mode == "student":
            return self.actor.infer_velocity_estimate(observations)
        raise ValueError(f"Unsupported actor mode: {actor_mode}")

    def act_by_mode(
        self,
        observations: torch.Tensor,
        actor_mode: str,
        *,
        detach_student_encoder: bool = False,
        deterministic: bool = False,
    ) -> torch.Tensor:
        latent = self._resolve_actor_mode_latent(
            observations,
            actor_mode,
            detach_student_encoder=detach_student_encoder,
        )
        velocity_feature = self.actor_mode_velocity_feature(observations, actor_mode)
        self.update_distribution_with_latent(
            observations,
            latent,
            velocity_feature=velocity_feature,
        )
        if deterministic:
            return self.distribution.mean
        return self.distribution.sample()

    def act(self, observations, hist_encoding=False, **kwargs):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, hist_encoding=False, scandots_latent=None, **kwargs):
        actions_mean = self.actor(observations, hist_encoding, scandots_latent)
        return actions_mean

    def act_student_inference(self, observations, zero_privileged: bool = True):
        """Deployment helper for student policy: zero privileged terms and force history encoding."""
        obs = observations
        if (
            obs.shape[1] == self.actor.in_features
            and zero_privileged
            and (self._num_priv_explicit + self._num_priv_latent) > 0
        ):
            obs = observations.clone()
            start = self._num_prop + self._num_scan
            end = start + self._num_priv_explicit + self._num_priv_latent
            obs[:, start:end] = 0.0
        return self.actor(obs, hist_encoding=True)

    def _encode_critic_obs(self, critic_observations: torch.Tensor) -> torch.Tensor:
        if not self._encode_scan_for_critic or self._critic_num_scan <= 0:
            return critic_observations
        obs_scan = critic_observations[
            :, self._critic_num_prop : self._critic_num_prop + self._critic_num_scan
        ]
        if self._critic_scan_encoder is not None:
            scan_latent = self._critic_scan_encoder(obs_scan)
        else:
            scan_latent = self.actor.scan_encoder(obs_scan)
        return torch.cat(
            [
                critic_observations[:, : self._critic_num_prop],
                scan_latent,
                critic_observations[:, self._critic_num_prop + self._critic_num_scan :],
            ],
            dim=1,
        )

    def evaluate(
        self,
        critic_observations,
        observations: torch.Tensor | None = None,
        actor_mode: str | None = None,
        latent: torch.Tensor | None = None,
        detach_student_encoder: bool = False,
        **kwargs,
    ):
        critic_input = self._encode_critic_obs(critic_observations)
        if self._critic_use_latent and self._history_latent_dim > 0:
            if latent is None:
                if observations is None or actor_mode is None:
                    raise ValueError(
                        "Critic latent conditioning requires observations and actor_mode."
                    )
                latent = self._resolve_actor_mode_latent(
                    observations,
                    actor_mode,
                    detach_student_encoder=detach_student_encoder,
                )
            critic_input = torch.cat([critic_input, latent], dim=-1)
        value = self.critic(critic_input)
        return value

    def evaluate_cost(
        self,
        critic_observations,
        observations: torch.Tensor | None = None,
        actor_mode: str | None = None,
        latent: torch.Tensor | None = None,
        detach_student_encoder: bool = False,
        **kwargs,
    ):
        critic_input = self._encode_critic_obs(critic_observations)
        if self._critic_use_latent and self._history_latent_dim > 0:
            if latent is None:
                if observations is None or actor_mode is None:
                    raise ValueError(
                        "Critic latent conditioning requires observations and actor_mode."
                    )
                latent = self._resolve_actor_mode_latent(
                    observations,
                    actor_mode,
                    detach_student_encoder=detach_student_encoder,
                )
            critic_input = torch.cat([critic_input, latent], dim=-1)
        cost_value = self.cost_critic(critic_input)
        return cost_value

    def teacher_latent(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor.infer_priv_latent(observations)

    def student_latent(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor.infer_hist_latent(observations)

    def latent_alignment_targets(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.student_latent(observations), self.teacher_latent(observations).detach()

    def student_encoder_parameters(self):
        history_encoder = getattr(self.actor, "history_encoder", None)
        if history_encoder is None or isinstance(history_encoder, nn.Identity):
            return []
        return list(history_encoder.parameters())

    def predict_velocity_estimate(self, observations: torch.Tensor) -> torch.Tensor | None:
        velocity = self.actor.infer_velocity_estimate(observations)
        if velocity.shape[-1] <= 0:
            return None
        return velocity

    def extract_velocity_estimation_target(
        self, critic_observations: torch.Tensor
    ) -> torch.Tensor | None:
        velocity_dim = int(getattr(self.actor, "velocity_estimation_dim", 0) or 0)
        required_prop_dim = 6 if velocity_dim == 3 else velocity_dim
        if velocity_dim <= 0 or self._critic_num_prop < required_prop_dim:
            return None
        return self.actor._extract_velocity_components(critic_observations[:, :required_prop_dim])

    def velocity_estimation(
        self, observations: torch.Tensor, critic_observations: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        prediction = self.predict_velocity_estimate(observations)
        target = self.extract_velocity_estimation_target(critic_observations)
        if prediction is None or target is None:
            return None, None
        return prediction, target

    def predict_history_reconstruction(self, observations: torch.Tensor) -> torch.Tensor | None:
        return self.actor.reconstruct_privileged_from_history(observations)

    def extract_history_reconstruction_target(
        self, critic_observations: torch.Tensor
    ) -> torch.Tensor | None:
        if self._history_reconstruction_mode == "base_lin_vel":
            if self._critic_num_prop < 3:
                return None
            return critic_observations[:, :3]
        target_terms: list[torch.Tensor] = []
        if self._reconstruction_missing_prop > 0:
            target_terms.append(critic_observations[:, : self._reconstruction_missing_prop])
        cursor = self._critic_num_prop
        if self._critic_num_scan > 0:
            target_terms.append(critic_observations[:, cursor : cursor + self._critic_num_scan])
            cursor += self._critic_num_scan
        if self._critic_num_priv_explicit > 0:
            target_terms.append(
                critic_observations[:, cursor : cursor + self._critic_num_priv_explicit]
            )
            cursor += self._critic_num_priv_explicit
        if self._critic_num_priv_latent > 0:
            target_terms.append(
                critic_observations[:, cursor : cursor + self._critic_num_priv_latent]
            )
        if not target_terms:
            return None
        return torch.cat(target_terms, dim=1)

    def history_reconstruction(
        self, observations: torch.Tensor, critic_observations: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        prediction = self.predict_history_reconstruction(observations)
        target = self.extract_history_reconstruction_target(critic_observations)
        if prediction is None or target is None:
            return None, None
        return prediction, target

    def load_state_dict(self, state_dict, strict=True):
        incompatible = super().load_state_dict(state_dict, strict=False)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith("actor.history_reconstruction_head")
            and not key.startswith("actor.velocity_estimator")
        ]
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith("actor.history_reconstruction_head")
            and not key.startswith("actor.velocity_estimator")
        ]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "State dict mismatch for ActorCriticRMA. "
                f"Missing keys: {missing}. Unexpected keys: {unexpected}."
            )
        return True
