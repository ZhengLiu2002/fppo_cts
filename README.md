# IsaacLab Galileo CTS

面向 Galileo 四足机器人复杂地形全向盲走任务的 CTS-only 约束强化学习训练框架。

仓库只保留一条主线：

- 单一任务：`Isaac-Galileo-CTS-v0`
- 单一评测入口：`Isaac-Galileo-CTS-Eval-v0`
- 单一回放 / 导出 / 遥控入口：`Isaac-Galileo-CTS-Play-v0`
- 单一研究目标：在统一 CTS 任务上对比 CRL 算法，重点验证 `FPPO`

## 核心位置

- Gym 注册：[crl_tasks/crl_tasks/tasks/galileo/__init__.py](/home/lz/Project/IsaacLab/fppo_ts/crl_tasks/crl_tasks/tasks/galileo/__init__.py)
- CTS 环境：[crl_tasks/crl_tasks/tasks/galileo/config/cts_env_cfg.py](/home/lz/Project/IsaacLab/fppo_ts/crl_tasks/crl_tasks/tasks/galileo/config/cts_env_cfg.py)
- CTS scene：[crl_tasks/crl_tasks/tasks/galileo/config/scene_cfg.py](/home/lz/Project/IsaacLab/fppo_ts/crl_tasks/crl_tasks/tasks/galileo/config/scene_cfg.py)
- MDP 配置：[crl_tasks/crl_tasks/tasks/galileo/config/mdp_cfg.py](/home/lz/Project/IsaacLab/fppo_ts/crl_tasks/crl_tasks/tasks/galileo/config/mdp_cfg.py)
- CTS benchmark runner：[crl_tasks/crl_tasks/tasks/galileo/config/agents/rsl_cts_cfg.py](/home/lz/Project/IsaacLab/fppo_ts/crl_tasks/crl_tasks/tasks/galileo/config/agents/rsl_cts_cfg.py)
- 算法注册表：[scripts/rsl_rl/algorithms/registry.py](/home/lz/Project/IsaacLab/fppo_ts/scripts/rsl_rl/algorithms/registry.py)
- 推理 / 导出 / 遥控：[scripts/rsl_rl/play.py](/home/lz/Project/IsaacLab/fppo_ts/scripts/rsl_rl/play.py) [scripts/rsl_rl/play_keyboard.py](/home/lz/Project/IsaacLab/fppo_ts/scripts/rsl_rl/play_keyboard.py) [scripts/rsl_rl/exporter.py](/home/lz/Project/IsaacLab/fppo_ts/scripts/rsl_rl/exporter.py)

## 安装

```bash
cd /home/lz/Project/IsaacLab/fppo_ts
pip install -e .
pip install -e ".[dev]"

cd /home/lz/Project/IsaacLab/fppo_ts/crl_tasks
pip install --no-build-isolation -e .
```

## 环境准备

```bash
conda activate isaaclab
cd /home/lz/Project/IsaacLab/fppo_ts
```

## 训练

最小 smoke test：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --run_name smoke \
  --headless
```

FPPO 对比命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/benchmark/cts_main \
  --num_envs 4096 \
  --run_name fppo_main \
  --headless \
  --logger wandb \
  --log_project_name galileo_cts
```

PPO 对比命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo ppo \
  --exp galileo/benchmark/cts_main \
  --num_envs 4096 \
  --run_name ppo_main \
  --headless \
  --logger wandb \
  --log_project_name galileo_cts
```

说明：

- `CTS` 是统一任务基座，`--algo` 是 CRL 对比轴
- 当前默认算法由 [defaults.py](/home/lz/Project/IsaacLab/fppo_ts/crl_tasks/crl_tasks/tasks/galileo/config/defaults.py) 指定，默认是 `fppo`
- 支持的主要算法：`ppo`、`fppo`、`np3o`、`ppo_lagrange`、`cpo`、`pcpo`、`focops`
- 训练结果默认写入 `logs/rsl_rl/<experiment_name>/`
- 每次训练的最终配置会落到：
  - `params/env.yaml`
  - `params/agent.yaml`
  - `params/experiment.json`

多卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --distributed \
  --num_envs 3000 \
  --exp galileo/benchmark/cts_main \
  --run_name fppo_4gpu \
  --device cuda:0
```

## 评估

FPPO 评估示例：

```bash
python scripts/rsl_rl/evaluation.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --max_episodes 256 \
  --headless \
  --checkpoint 
```

评估会输出 `summary.json`，包含 reward、episode length、aggregate cost 与 per-term cost 统计。

PPO 评估示例：

```bash
python scripts/rsl_rl/evaluation.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo ppo \
  --max_episodes 256 \
  --headless \
  --checkpoint 
```

## 回放与遥控

GUI 回放：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CTS-Play-v0 \
  --checkpoint <checkpoint.pt> \
  --num_envs 16 \
  --force_gui
```

键盘遥控：

FPPO 键盘遥控示例：

```bash
python scripts/rsl_rl/play_keyboard.py \
  --task Isaac-Galileo-CTS-Play-v0 \
  --algo fppo \
  --force_gui \
  --checkpoint 
```

PPO 键盘遥控示例：

```bash
python scripts/rsl_rl/play_keyboard.py \
  --task Isaac-Galileo-CTS-Play-v0 \
  --algo ppo \
  --force_gui \
  --checkpoint 
```

常用控制：

- `W/S` 或方向键 `Up/Down`：前进 / 后退
- `A/D`：左右横移
- `Q/E` 或方向键 `Left/Right`：左 / 右转
- `Space`：清空当前目标速度
- `R`：重置机器人
- `Esc`：退出

## 导出

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CTS-Play-v0 \
  --num_envs 1 \
  --export_only \
  --headless \
  --checkpoint 
```

默认输出：

- `exported_policy/policy.onnx`
- `exported_policy/policy.yaml`

导出接口面向最终 blind policy，核心输入按 `policy.yaml` 为准。

PPO 导出示例：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CTS-Play-v0 \
  --algo ppo \
  --checkpoint <checkpoint.pt> \
  --num_envs 1 \
  --export_only \
  --headless
```

## 实验 preset

查看可用 preset：

```bash
python scripts/rsl_rl/train.py --list-exp
```

推荐起点：

- `galileo/fppo_smoke`
- `galileo/studies/algo_compare_cts_fair`
- `galileo/benchmark/cts_main`

## 开发

- 算法接入说明：`docs/ALGORITHMS.md`
- 实验工作流：`docs/EXPERIMENTS.md`
- 对比计划：`docs/ALGO_COMPARISON_PLAN.md`
- 目录结构：`docs/STRUCTURE.md`

建议提交前执行：

```bash
./scripts/run_tests.sh -q
python -m black scripts/rsl_rl tests
```
