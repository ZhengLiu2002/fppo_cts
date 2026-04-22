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
  --num_envs 2048 \
  --run_name fppo_main \
  --headless \
  --logger wandb \
  --max_iterations 20000 \
  --log_project_name galileo_cts
```

PPO 对比命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo ppo \
  --exp galileo/benchmark/cts_main \
  --num_envs 2048 \
  --run_name ppo_main \
  --headless \
  --logger wandb \
  --max_iterations 20000 \
  --log_project_name galileo_cts
```

FPPO 单独教师对比命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/teacher_only_upper_bound \
  --run_name teacher_only \
  --num_envs 2048 \
  --headless \
  --logger wandb \
  --max_iterations 20000 \
  --log_project_name galileo_cts
```

说明：

- `CTS` 是统一的 teacher-student 训练框架，`--algo` 只切换优化器 / 投影机制
- 在 `Isaac-Galileo-CTS-v0` 上切 `ppo`、`fppo`、`ppo_lagrange` 等时，teacher/student rollout、latent 对齐和学生 history encoder 监督保持一致
- runner 不再按算法名字写死 CTS 分支；算法如需额外 CTS 运行时行为，通过 [contracts.py](/home/lz/Project/IsaacLab/fppo_ts/scripts/rsl_rl/algorithms/contracts.py) 的契约声明
- 当前默认算法由 [defaults.py](/home/lz/Project/IsaacLab/fppo_ts/crl_tasks/crl_tasks/tasks/galileo/config/defaults.py) 指定，默认是 `fppo`
- 支持的主要算法：`ppo`、`fppo`、`np3o`、`ppo_lagrange`、`cpo`、`pcpo`、`focops`
- 兼容旧配置时，`--algo cts` 会退化为 `ppo` on CTS framework，但新的对比实验建议直接使用真实优化器名
- 训练结果默认写入 `logs/rsl_rl/<experiment_name>/`
- 每次训练的最终配置会落到：
  - `params/env.yaml`
  - `params/agent.yaml`
  - `params/experiment.json`
- 每次训练还会在 run 根目录生成 `policy.yaml`
  - 这份文件直接来自当次训练的 live 配置，是部署参数的唯一来源
  - 后续 `play.py --export_only` 只会复制这份 `policy.yaml` 到 `exported_policy/`

命令追加参数：

README 里的训练命令都可以在末尾继续追加 CLI 参数，临时覆盖 preset 或任务默认值。比如把训练轮数改成 `20000`：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/benchmark/cts_main \
  --run_name fppo_main \
  --headless \
  --max_iterations 20000
```

常见追加方式：

- `--max_iterations 20000`：覆盖 `agent.max_iterations`
- `--num_envs 4096`：覆盖 `env.scene.num_envs`
- `--seed 42`：覆盖随机种子
- 同时使用 `--exp` 时，CLI 追加参数优先级高于 preset

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
  --checkpoint logs/
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
conda activate isaaclab
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CTS-Play-v0 \
  --num_envs 1 \
  --export_only \
  --headless \
  --checkpoint logs/
```

默认输出：

- `exported_policy/policy.onnx`
- `exported_policy/policy.yaml`

导出接口面向最终 blind policy，核心输入按 `policy.yaml` 为准。
`exported_policy/policy.yaml` 由 run 根目录的 `policy.yaml` 直接复制得到，不再回读旧 checkpoint 邻接的冻结配置。

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
