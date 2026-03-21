# Student History Memo

## Context

Current priority is to finish training a high-quality **teacher** policy first.
The student-history work is deferred so it does not distract from teacher reward / gait / FPPO stability debugging.

This memo records the intended student proprio-history design so we can resume it later without forgetting the details.

## Goal

Enable a **student actor** with:
- blind proprioceptive policy
- **20-step proprio history**
- **1D temporal convolution encoder** for history
- asymmetric critic kept privileged

## Intended design

### Student actor history settings
- `actor_num_hist = 20`
- `actor_num_priv_latent = 32`
- use the existing `StateHistoryEncoder`
- history input should be **proprio only**

### Proprio history contents
The history sequence should contain:
- `base_ang_vel`
- `projected_gravity`
- `joint_pos_rel`
- `joint_vel_rel`
- `last_action`
- `generated_commands(base_velocity)`

Do **not** include:
- scan / terrain observation
- privileged teacher-only information

### Encoder choice
Use the existing 1D conv history encoder in:
- `scripts/rsl_rl/modules/feature_extractors/state_encoder.py`

For `tsteps=20`, the code path already exists and is preferred over adding an RNN first.

## Why this is useful

Student control is currently blind and feedforward by default.
A 20-step proprio history should help with:
- actuator delay
- contact timing / gait phase inference
- partial observability in blind locomotion
- better recovery under proprio-only control

## Integration points to revisit later

### Observation/config side
Need the student policy observation to append:
- a latent placeholder block (32 dims)
- a 20-step proprio history block

Relevant files:
- `crl_tasks/crl_tasks/tasks/galileo/config/defaults.py`
- `crl_tasks/crl_tasks/tasks/galileo/config/mdp_cfg.py`
- `crl_isaaclab/envs/mdp/observations.py`

### Policy / training side
Need history encoding to be used consistently in:
- rollout action sampling
- PPO / FPPO update-time `policy.act(...)`
- inference / play / export

Relevant files:
- `scripts/rsl_rl/algorithms/ppo.py`
- `scripts/rsl_rl/algorithms/fppo.py`
- `scripts/rsl_rl/modules/on_policy_runner_with_extractor.py`
- `scripts/rsl_rl/exporter.py`

### Distillation warning
When we return to student training, also verify the **distillation** path uses history encoding.
Relevant file:
- `scripts/rsl_rl/algorithms/dagger.py`

This is important because our actual workflow is:
1. train teacher
2. distill / train student from teacher

So student history must not only work in student RL mode; it must also be connected in the distillation pipeline.

## Recommended future order

After teacher is good:
1. enable student proprio history
2. verify observation dimensions match policy dimensions
3. verify rollout/update/inference all use history encoding
4. patch distillation path if needed
5. run student ablation:
   - no history
   - 10-step history
   - 20-step history

## Expected success criteria

Compared with blind feedforward student, history-enabled student should show:
- better tracking
- fewer falls
- improved gait consistency
- stronger robustness to delay and unseen terrain variations
