# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# rsl-rl
from rsl_rl.modules import StudentTeacher, StudentTeacherRecurrent
from scripts.rsl_rl.storage.rollout_storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student model to mimic a teacher model."""

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

        # distillation components
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
        self.last_hidden_states = None

        # distillation parameters
        self.num_learning_epochs = num_learning_epochs
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate
        self.reconstruction_loss_coef = float(reconstruction_loss_coef)

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
                "Distillation requires a teacher policy, but none was attached. "
                "Provide a teacher checkpoint through the runner's distillation path."
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
        return self.reconstruction_loss_coef * nn.functional.mse_loss(prediction, target.detach())

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

    def act(self, obs, teacher_obs, hist_encoding=False):
        # compute the actions
        self.transition.actions = self._policy_act(obs).detach()
        with torch.no_grad():
            self.transition.privileged_actions = self._teacher_act_inference(teacher_obs).detach()
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

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_reconstruction_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self._reset_student_policy(hidden_states=self.last_hidden_states)
            self._detach_student_hidden_states()
            for obs, privileged_obs, _, privileged_actions, dones in self.storage.generator():

                # inference the student for gradient computation
                actions = self._policy_act_inference(obs)

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)
                reconstruction_loss = self._history_reconstruction_loss(obs, privileged_obs)

                # total loss
                loss = loss + behavior_loss + reconstruction_loss
                mean_behavior_loss += behavior_loss.item()
                mean_reconstruction_loss += reconstruction_loss.item()
                cnt += 1

                # gradient step
                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    self.optimizer.step()
                    self._detach_student_hidden_states()
                    loss = 0

                # reset dones
                done_mask = dones.view(-1)
                self._reset_student_policy(dones=done_mask)
                self._detach_student_hidden_states(done_mask)

        mean_behavior_loss /= cnt
        mean_reconstruction_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self._get_student_hidden_states()
        self._detach_student_hidden_states()

        # construct the loss dictionary
        loss_dict = {
            "behavior": mean_behavior_loss,
            "reconstruction": mean_reconstruction_loss,
        }

        return loss_dict

    def load_teacher_state_dict(self, state_dict: dict) -> None:
        teacher_module = self._teacher_module()
        if teacher_module is None:
            raise AttributeError(
                "Cannot load a teacher checkpoint because the distillation algorithm has no "
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
