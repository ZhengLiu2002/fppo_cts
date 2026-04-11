# Project Structure

This repo follows a task-centric layout similar to production RL frameworks (tasks → configs → agents). The goal is to keep task configuration modular and easy to extend, while keeping the training scripts and IsaacLab extensions stable.

## Top-Level

- `crl_isaaclab/`: Custom IsaacLab extensions (envs, terrains, actuators, MDP terms).
- `crl_tasks/`: Task registration and task-specific configuration package.
- `scripts/rsl_rl/`: Training, evaluation, play, and export scripts.
- `tests/`: Functional tests and sanity checks.
- `tools/sim/`: Isaac Sim / GUI-only demo scripts kept out of automated tests.
- `docs/`: Structure and style guidance.

## Task Configuration Layout

Each task lives under `crl_tasks/crl_tasks/tasks/<task_name>/` and exposes:

- `config/defaults.py`: Shared defaults (robot USD, scene defaults, terrain presets, algorithm knobs).
- `config/mdp_cfg.py`: MDP terms (observations, rewards, terminations, curriculum, events).
- `config/scene_cfg.py`: CTS scene configuration shared by train/eval/play.
- `config/cts_env_cfg.py`: CTS train/eval/play environment configuration.
- `config/costs_cfg.py`: CMDP constraint/cost definitions.
- `config/agents/`: Agent configs for CTS benchmark algorithms.
- `config/assets/`: Task-specific USD assets used in configs/tests.

Gym registrations live at `crl_tasks/crl_tasks/tasks/<task_name>/__init__.py`.

## Algorithms

Algorithm registry and validation live in `scripts/rsl_rl/algorithms/registry.py`.
See `docs/ALGORITHMS.md` for the extension workflow.
