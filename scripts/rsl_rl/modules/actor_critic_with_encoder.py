from __future__ import annotations
from typing import Optional, Callable, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .feature_extractors.state_encoder import *
from rsl_rl.utils import resolve_nn_activation

# Simple registry to avoid eval-based class lookup.
ACTOR_REGISTRY: Dict[str, Callable] = {}


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
        history_latent_dim = int(kwargs.pop("history_latent_dim", num_priv_latent) or 0)
        self.history_latent_dim = history_latent_dim if history_latent_dim > 0 else num_priv_latent
        self.history_reconstruction_dim = int(kwargs.pop("history_reconstruction_dim", 0) or 0)
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        # Raw observation layout: current proprio + optional scan / explicit priv + optional raw priv latent + history.
        self.in_features = (
            num_prop + num_scan + num_priv_latent + num_priv_explicit + num_prop * num_hist
        )

        if len(priv_encoder_dims) > 0 and num_priv_latent > 0:
            priv_encoder_layers = []
            priv_encoder_layers.append(nn.Linear(num_priv_latent, priv_encoder_dims[0]))
            priv_encoder_layers.append(activation)
            for l in range(len(priv_encoder_dims) - 1):
                priv_encoder_layers.append(
                    nn.Linear(priv_encoder_dims[l], priv_encoder_dims[l + 1])
                )
                priv_encoder_layers.append(activation)
            if priv_encoder_dims[-1] != self.history_latent_dim:
                priv_encoder_layers.append(nn.Linear(priv_encoder_dims[-1], self.history_latent_dim))
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
        elif num_priv_latent > 0 and num_priv_latent != self.history_latent_dim:
            self.priv_encoder = nn.Sequential(
                nn.Linear(num_priv_latent, self.history_latent_dim),
                activation,
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
        self._hist_latent_dim = latent_dim
        if self.if_scan_encode:
            scan_encoder = []
            scan_encoder.append(nn.Linear(num_scan, scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(scan_encoder_dims) - 1):
                if l == len(scan_encoder_dims) - 2:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(nn.Linear(scan_encoder_dims[l], scan_encoder_dims[l + 1]))
                    scan_encoder.append(activation)
            self.scan_encoder = nn.Sequential(*scan_encoder)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
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

    def forward(self, obs, hist_encoding: bool, scandots_latent: Optional[torch.Tensor] = None):
        """前向推理：
        - 可选对激光编码（或复用外部提供的 scandots_latent）
        - 可选使用历史编码器，否则只编码特权隐式
        """
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
        if hist_encoding:
            latent = self.infer_hist_latent(obs)
        else:
            latent = self.infer_priv_latent(obs)
        backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        backbone_output = self.actor_backbone(backbone_input)
        return backbone_output

    def infer_priv_latent(self, obs):
        """仅编码特权隐式变量。"""
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
        return self.priv_encoder(priv)

    def infer_hist_latent(self, obs):
        """对历史本体序列做 1D 卷积编码。"""
        if self.num_hist <= 0 or self._hist_latent_dim <= 0:
            return self.infer_priv_latent(obs)
        hist = obs[:, -self.num_hist * self.num_prop :]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))

    def infer_scandots_latent(self, obs):
        """仅对激光点做编码。"""
        scan = obs[:, self.num_prop : self.num_prop + self.num_scan]
        return self.scan_encoder(scan)

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
        critic_scan_encoder_dims = kwargs.get("critic_scan_encoder_dims", None)
        self._critic_scan_encoder = None
        self._critic_scan_encoder_output_dim = self._critic_num_scan
        if self._encode_scan_for_critic and self._critic_num_scan > 0 and critic_scan_encoder_dims:
            scan_encoder = []
            scan_encoder.append(nn.Linear(self._critic_num_scan, critic_scan_encoder_dims[0]))
            scan_encoder.append(activation)
            for l in range(len(critic_scan_encoder_dims) - 1):
                if l == len(critic_scan_encoder_dims) - 2:
                    scan_encoder.append(
                        nn.Linear(critic_scan_encoder_dims[l], critic_scan_encoder_dims[l + 1])
                    )
                    scan_encoder.append(nn.Tanh())
                else:
                    scan_encoder.append(
                        nn.Linear(critic_scan_encoder_dims[l], critic_scan_encoder_dims[l + 1])
                    )
                    scan_encoder.append(activation)
            self._critic_scan_encoder = nn.Sequential(*scan_encoder)
            self._critic_scan_encoder_output_dim = critic_scan_encoder_dims[-1]
        elif self._encode_scan_for_critic and self._critic_num_scan > 0:
            self._critic_scan_encoder_output_dim = self.actor.scan_encoder_output_dim
        self._reconstruction_missing_prop = max(self._critic_num_prop - self._num_prop, 0)
        self._history_reconstruction_dim = (
            self._reconstruction_missing_prop
            + self._critic_num_scan
            + self._critic_num_priv_explicit
            + self._critic_num_priv_latent
        )

        # Critic：直接接收 num_critic_obs（可能含特权信息）
        if self._encode_scan_for_critic and self._critic_num_scan > 0:
            mlp_input_dim_c = (
                num_critic_obs - self._critic_num_scan + self._critic_scan_encoder_output_dim
            )
        else:
            mlp_input_dim_c = num_critic_obs
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
        if zero_privileged and (self._num_priv_explicit + self._num_priv_latent) > 0:
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

    def evaluate(self, critic_observations, **kwargs):
        critic_input = self._encode_critic_obs(critic_observations)
        value = self.critic(critic_input)
        return value

    def evaluate_cost(self, critic_observations, **kwargs):
        critic_input = self._encode_critic_obs(critic_observations)
        cost_value = self.cost_critic(critic_input)
        return cost_value

    def predict_history_reconstruction(self, observations: torch.Tensor) -> torch.Tensor | None:
        return self.actor.reconstruct_privileged_from_history(observations)

    def extract_history_reconstruction_target(
        self, critic_observations: torch.Tensor
    ) -> torch.Tensor | None:
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
        ]
        unexpected = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith("actor.history_reconstruction_head")
        ]
        if strict and (missing or unexpected):
            raise RuntimeError(
                "State dict mismatch for ActorCriticRMA. "
                f"Missing keys: {missing}. Unexpected keys: {unexpected}."
            )
        return True
