# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from scripts.rsl_rl.storage.rollout_storage import RolloutStorage


class _DaggerReplayBuffer:
    """Fixed-size replay of student-visited states labeled by the teacher."""

    def __init__(
        self,
        capacity: int,
        obs_shape,
        teacher_obs_shape,
        actions_shape,
        *,
        storage_device: str = "cpu",
    ) -> None:
        self.capacity = max(int(capacity), 0)
        self.device = torch.device(storage_device)
        self.observations = None
        self.teacher_observations = None
        self.teacher_actions = None
        self._obs_shape = tuple(obs_shape)
        self._teacher_obs_shape = tuple(teacher_obs_shape)
        self._actions_shape = tuple(actions_shape)
        if self.capacity > 0:
            self.observations = torch.zeros(self.capacity, *self._obs_shape, device=self.device)
            self.teacher_observations = torch.zeros(
                self.capacity, *self._teacher_obs_shape, device=self.device
            )
            self.teacher_actions = torch.zeros(self.capacity, *self._actions_shape, device=self.device)
        self.size = 0
        self.ptr = 0

    def __len__(self) -> int:
        return int(self.size)

    @property
    def fill_ratio(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return float(self.size) / float(self.capacity)

    def add(
        self,
        observations: torch.Tensor,
        teacher_observations: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> None:
        if self.capacity <= 0 or self.observations is None:
            return

        obs = observations.detach().to(self.device)
        teacher_obs = teacher_observations.detach().to(self.device)
        expert_actions = teacher_actions.detach().to(self.device)
        batch = obs.shape[0]
        if batch <= 0:
            return

        if batch >= self.capacity:
            obs = obs[-self.capacity :]
            teacher_obs = teacher_obs[-self.capacity :]
            expert_actions = expert_actions[-self.capacity :]
            batch = self.capacity

        first = min(self.capacity - self.ptr, batch)
        second = batch - first

        self.observations[self.ptr : self.ptr + first].copy_(obs[:first])
        self.teacher_observations[self.ptr : self.ptr + first].copy_(teacher_obs[:first])
        self.teacher_actions[self.ptr : self.ptr + first].copy_(expert_actions[:first])

        if second > 0:
            self.observations[:second].copy_(obs[first:])
            self.teacher_observations[:second].copy_(teacher_obs[first:])
            self.teacher_actions[:second].copy_(expert_actions[first:])

        self.ptr = (self.ptr + batch) % self.capacity
        self.size = min(self.size + batch, self.capacity)

    def iter_batches(
        self,
        *,
        batch_size: int,
        num_epochs: int,
        target_device: str,
    ):
        if self.size <= 0 or self.observations is None:
            return

        effective_batch_size = max(1, min(int(batch_size), self.size))
        target = torch.device(target_device)

        for _ in range(max(int(num_epochs), 1)):
            indices = torch.randperm(self.size, device=self.device)
            for start in range(0, self.size, effective_batch_size):
                batch_idx = indices[start : start + effective_batch_size]
                yield (
                    self.observations[batch_idx].to(target),
                    self.teacher_observations[batch_idx].to(target),
                    self.teacher_actions[batch_idx].to(target),
                )


class DAgger:
    """DAgger-style student training with a frozen teacher as the label source."""

    policy: StudentTeacher | StudentTeacherRecurrent
    """The student teacher model."""

    def __init__(
        self,
        policy,
        teacher_policy=None,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        loss_type="mse",
        reconstruction_loss_coef: float = 0.0,
        dagger_buffer_size: int = 524_288,
        dagger_batch_size: int = 16_384,
        dagger_min_buffer_size: int = 131_072,
        dagger_batches_per_update: int = 32,
        teacher_action_ratio_start: float = 0.0,
        teacher_action_ratio_end: float = 0.0,
        teacher_action_ratio_decay_steps: int = 1,
        deterministic_rollout: bool = True,
        max_grad_norm: float = 1.0,
        device="cpu",
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
    ):
        # device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # DAgger components
        self.policy = policy
        self.policy.to(self.device)
        self.teacher_policy = teacher_policy
        if self.teacher_policy is not None:
            self.teacher_policy.to(self.device)
            self.teacher_policy.eval()
            for param in self.teacher_policy.parameters():
                param.requires_grad_(False)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self._student_module().parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # DAgger parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.reconstruction_loss_coef = float(reconstruction_loss_coef)
        self.dagger_buffer_size = max(int(dagger_buffer_size), 0)
        self.dagger_batch_size = max(int(dagger_batch_size), 1)
        self.dagger_min_buffer_size = max(int(dagger_min_buffer_size), 1)
        self.dagger_batches_per_update = max(int(dagger_batches_per_update), 0)
        if self.dagger_buffer_size > 0:
            self.dagger_min_buffer_size = min(self.dagger_min_buffer_size, self.dagger_buffer_size)
        self.teacher_action_ratio_start = float(teacher_action_ratio_start)
        self.teacher_action_ratio_end = float(teacher_action_ratio_end)
        self.teacher_action_ratio_decay_steps = max(int(teacher_action_ratio_decay_steps), 1)
        self.deterministic_rollout = bool(deterministic_rollout)
        self.max_grad_norm = float(max_grad_norm)
        self.replay_buffer: _DaggerReplayBuffer | None = None
        self._last_teacher_action_ratio = 0.0
        self.num_rollouts = 0

        # initialize the loss function
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def _student_module(self):
        return getattr(self.policy, "student", self.policy)

    def _teacher_module(self):
        if self.teacher_policy is not None:
            return self.teacher_policy
        return getattr(self.policy, "teacher", None)

    def _use_history_encoding(self) -> bool:
        actor = getattr(self.policy, "actor", None)
        return bool(getattr(actor, "num_hist", 0) > 0)

    @staticmethod
    def _sanitize_tensor(
        tensor: torch.Tensor,
        *,
        nan: float = 0.0,
        posinf: float = 1.0e6,
        neginf: float = 0.0,
        clamp: float | None = 1.0e6,
    ) -> torch.Tensor:
        sanitized = torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)
        if clamp is not None:
            min_value = 0.0 if neginf >= 0.0 else -float(clamp)
            sanitized = torch.clamp(sanitized, min=min_value, max=float(clamp))
        return sanitized

    def _teacher_action_ratio(self) -> float:
        if self.teacher_action_ratio_start == self.teacher_action_ratio_end:
            return self.teacher_action_ratio_end
        progress = min(
            float(self.num_rollouts) / float(self.teacher_action_ratio_decay_steps), 1.0
        )
        return self.teacher_action_ratio_start + progress * (
            self.teacher_action_ratio_end - self.teacher_action_ratio_start
        )

    def _policy_act(self, obs):
        try:
            return self.policy.act(obs, hist_encoding=self._use_history_encoding())
        except TypeError:
            return self.policy.act(obs)

    def _policy_act_inference(self, obs):
        try:
            return self.policy.act_inference(obs, hist_encoding=self._use_history_encoding())
        except TypeError:
            return self.policy.act_inference(obs)

    def _teacher_act_inference(self, teacher_obs):
        teacher_module = self._teacher_module()
        if teacher_module is None:
            raise AttributeError(
                "DAgger requires a teacher policy, but none was attached. "
                "Provide a teacher checkpoint through the runner's DAgger path."
            )
        if hasattr(teacher_module, "act_inference"):
            try:
                return teacher_module.act_inference(teacher_obs, hist_encoding=False)
            except TypeError:
                return teacher_module.act_inference(teacher_obs)
        if hasattr(teacher_module, "act"):
            try:
                return teacher_module.act(teacher_obs, hist_encoding=False)
            except TypeError:
                return teacher_module.act(teacher_obs)
        raise AttributeError(
            f"Teacher policy of type {type(teacher_module).__name__} does not support inference."
        )

    def _reset_student_policy(self, dones=None, hidden_states=None):
        if hidden_states is not None:
            try:
                self.policy.reset(hidden_states=hidden_states)
                return
            except TypeError:
                pass
        if dones is not None:
            self.policy.reset(dones)
        else:
            self.policy.reset()

    def _detach_student_hidden_states(self, dones=None):
        if not hasattr(self.policy, "detach_hidden_states"):
            return
        if dones is not None:
            self.policy.detach_hidden_states(dones)
        else:
            self.policy.detach_hidden_states()

    def _get_student_hidden_states(self):
        if hasattr(self.policy, "get_hidden_states"):
            return self.policy.get_hidden_states()
        return None

    def _rollout_student_actions(self, obs: torch.Tensor) -> torch.Tensor:
        if self.deterministic_rollout:
            return self._policy_act_inference(obs)
        return self._policy_act(obs)

    def _history_reconstruction_loss(
        self,
        obs: torch.Tensor,
        privileged_obs: torch.Tensor,
    ) -> torch.Tensor:
        if self.reconstruction_loss_coef <= 0.0 or not hasattr(self.policy, "history_reconstruction"):
            return torch.zeros((), device=obs.device)
        prediction, target = self.policy.history_reconstruction(obs, privileged_obs)
        if prediction is None or target is None:
            return torch.zeros((), device=obs.device)
        reconstruction_loss = nn.functional.mse_loss(prediction, target.detach())
        reconstruction_loss = self._sanitize_tensor(
            reconstruction_loss, nan=0.0, posinf=1.0e6, neginf=0.0, clamp=1.0e6
        )
        return self.reconstruction_loss_coef * reconstruction_loss

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_obs_shape,
        teacher_obs_shape,
        actions_shape,
    ):
        # create rollout storage
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            self.device,
        )
        if self.dagger_buffer_size > 0:
            self.replay_buffer = _DaggerReplayBuffer(
                self.dagger_buffer_size,
                student_obs_shape,
                teacher_obs_shape,
                actions_shape,
            )

    def act(self, obs, teacher_obs, hist_encoding=False):
        student_actions = self._rollout_student_actions(obs).detach()
        with torch.no_grad():
            teacher_actions = self._teacher_act_inference(teacher_obs).detach()
        teacher_action_ratio = max(0.0, min(1.0, self._teacher_action_ratio()))
        rollout_actions = student_actions
        if teacher_action_ratio >= 1.0:
            rollout_actions = teacher_actions
        elif teacher_action_ratio > 0.0:
            teacher_mask = (
                torch.rand(student_actions.shape[0], 1, device=student_actions.device)
                < teacher_action_ratio
            )
            rollout_actions = torch.where(teacher_mask, teacher_actions, student_actions)

        self._last_teacher_action_ratio = teacher_action_ratio
        self.transition.actions = rollout_actions.detach()
        self.transition.privileged_actions = teacher_actions
        # record the observations
        self.transition.observations = obs
        self.transition.privileged_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, infos, costs=None, cost_terms=None):
        # record the rewards and dones
        self.transition.rewards = rewards
        self.transition.dones = dones
        # record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self._reset_student_policy(dones=dones)

    def _append_rollout_to_replay(self) -> int:
        if self.replay_buffer is None or self.storage is None or self.storage.step <= 0:
            return 0
        steps = self.storage.step
        samples_added = steps * self.storage.num_envs
        teacher_observations = (
            self.storage.privileged_observations[:steps]
            if self.storage.privileged_observations is not None
            else self.storage.observations[:steps]
        )
        self.replay_buffer.add(
            self.storage.observations[:steps].flatten(0, 1),
            teacher_observations.flatten(0, 1),
            self.storage.privileged_actions[:steps].flatten(0, 1),
        )
        return int(samples_added)

    def _build_loss_dict(
        self,
        *,
        behavior: float,
        reconstruction: float,
        buffer_size: float,
        buffer_fill: float,
        samples_added: float,
        update_due: bool,
        updated: bool,
        num_batches: float,
        warmup: bool,
    ) -> dict[str, float]:
        return {
            "behavior": behavior,
            "reconstruction": reconstruction,
            "dagger_buffer_size": buffer_size,
            "dagger_buffer_fill": buffer_fill,
            "dagger_samples_added": samples_added,
            "dagger_update_due": float(update_due),
            "dagger_updated": float(updated),
            "dagger_num_batches": num_batches,
            "dagger_batch_budget": float(self.dagger_batches_per_update),
            "dagger_warmup": float(warmup),
            "dagger_policy_updates": float(self.num_updates),
            "teacher_action_ratio": self._last_teacher_action_ratio,
        }

    def update(self, do_policy_update: bool = True):
        samples_added = float(self._append_rollout_to_replay())
        if samples_added > 0:
            self.num_rollouts += 1
        self.storage.clear()
        buffer_size = float(len(self.replay_buffer)) if self.replay_buffer is not None else 0.0
        buffer_fill = self.replay_buffer.fill_ratio if self.replay_buffer is not None else 0.0

        if not do_policy_update:
            return self._build_loss_dict(
                behavior=0.0,
                reconstruction=0.0,
                buffer_size=buffer_size,
                buffer_fill=buffer_fill,
                samples_added=samples_added,
                update_due=False,
                updated=False,
                num_batches=0.0,
                warmup=False,
            )

        if self.replay_buffer is None or len(self.replay_buffer) < self.dagger_min_buffer_size:
            return self._build_loss_dict(
                behavior=0.0,
                reconstruction=0.0,
                buffer_size=buffer_size,
                buffer_fill=buffer_fill,
                samples_added=samples_added,
                update_due=True,
                updated=False,
                num_batches=0.0,
                warmup=True,
            )

        mean_behavior_loss = 0.0
        mean_reconstruction_loss = 0.0
        num_batches = 0

        for obs_batch, teacher_obs_batch, teacher_actions_batch in self.replay_buffer.iter_batches(
            batch_size=self.dagger_batch_size,
            num_epochs=self.num_learning_epochs,
            target_device=self.device,
        ):
            actions = self._policy_act_inference(obs_batch)
            behavior_loss = self._sanitize_tensor(
                self.loss_fn(actions, teacher_actions_batch),
                nan=0.0,
                posinf=1.0e6,
                neginf=0.0,
                clamp=1.0e6,
            )
            reconstruction_loss = self._history_reconstruction_loss(obs_batch, teacher_obs_batch)
            loss = self._sanitize_tensor(
                behavior_loss + reconstruction_loss,
                nan=0.0,
                posinf=1.0e6,
                neginf=0.0,
                clamp=1.0e6,
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            if self.max_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(self._student_module().parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_behavior_loss += behavior_loss.item()
            mean_reconstruction_loss += reconstruction_loss.item()
            num_batches += 1
            if self.dagger_batches_per_update > 0 and num_batches >= self.dagger_batches_per_update:
                break

        if num_batches > 0:
            mean_behavior_loss /= num_batches
            mean_reconstruction_loss /= num_batches

        self.num_updates += 1
        return self._build_loss_dict(
            behavior=mean_behavior_loss,
            reconstruction=mean_reconstruction_loss,
            buffer_size=buffer_size,
            buffer_fill=buffer_fill,
            samples_added=samples_added,
            update_due=True,
            updated=True,
            num_batches=float(num_batches),
            warmup=False,
        )

    def load_teacher_state_dict(self, state_dict: dict) -> None:
        teacher_module = self._teacher_module()
        if teacher_module is None:
            raise AttributeError(
                "Cannot load a teacher checkpoint because the DAgger algorithm has no "
                "teacher policy attached."
            )
        try:
            teacher_module.load_state_dict(state_dict)
        except RuntimeError as exc:
            if "cost_critic" not in str(exc):
                raise
            teacher_state = teacher_module.state_dict()
            patched_state = dict(state_dict)
            patched = False
            for key, src_tensor in list(state_dict.items()):
                if not (key.startswith("cost_critic.") and key.endswith(".weight")):
                    continue
                if key not in teacher_state:
                    continue
                tgt_tensor = teacher_state[key]
                if src_tensor.ndim != 2 or tgt_tensor.ndim != 2:
                    continue
                if src_tensor.shape[1] != tgt_tensor.shape[1]:
                    continue
                if src_tensor.shape[0] == tgt_tensor.shape[0]:
                    continue

                bias_key = key.replace(".weight", ".bias")
                if bias_key not in state_dict or bias_key not in teacher_state:
                    continue
                src_bias = state_dict[bias_key]
                tgt_bias = teacher_state[bias_key]
                if src_bias.ndim != 1 or tgt_bias.ndim != 1:
                    continue

                if src_tensor.shape[0] == 1:
                    new_weight = src_tensor.expand(tgt_tensor.shape[0], -1).clone()
                else:
                    new_weight = tgt_tensor.clone()
                    rows = min(src_tensor.shape[0], tgt_tensor.shape[0])
                    new_weight[:rows, :] = src_tensor[:rows, :]
                if src_bias.shape[0] == 1:
                    new_bias = src_bias.expand(tgt_bias.shape[0]).clone()
                else:
                    new_bias = tgt_bias.clone()
                    rows = min(src_bias.shape[0], tgt_bias.shape[0])
                    new_bias[:rows] = src_bias[:rows]

                patched_state[key] = new_weight
                patched_state[bias_key] = new_bias
                patched = True
            if not patched:
                raise
            teacher_module.load_state_dict(patched_state)
        teacher_module.eval()
        for param in teacher_module.parameters():
            param.requires_grad_(False)

    """
    Helper functions
    """

    def broadcast_parameters(self):
        """Broadcast model parameters to all GPUs."""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        grads = [
            param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None
        ]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
