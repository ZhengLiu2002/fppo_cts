# 实验版本管理与 Git 工作流

CTS-only 框架下，只需要管理两类版本：

- 代码版本：算法实现、奖励设计、观测布局、训练流程改动
- 实验版本：超参数、terrain、约束阈值、训练规模改动

## 规则

- 只改参数：新增 `experiments/` preset
- 改代码逻辑：新建 git 分支
- 同时跑多个代码版本：优先用 `git worktree`

## 训练脚本支持

- `--exp <name>`
- `--exp-file <path>`
- `--list-exp`

配置生效顺序：

1. CTS 任务默认配置
2. `--algo` 触发的算法 profile
3. preset 覆写
4. CLI 强优先级参数

每次训练会保存：

- `params/env.yaml`
- `params/agent.yaml`
- `params/experiment.json`

## 推荐 preset 结构

```text
experiments/
  galileo/
    base.json
    debug/
      fppo_smoke.json
      blind_walk.json
    curriculum/
      flat_speed.json
      flat_omni.json
      rough_foundation.json
      rough_progressive.json
      rough_mainline.json
    probes/
      terrain_row.json
      terrain_perception.json
    legacy/
      rough_metric_displacement.json
      rough_row_bootstrap.json
      terrain_hard.json
    variants/
      cost_limit_relaxed.json
    studies/
      algo_compare_cts.json
    benchmark/
      cts_main.json
```

## 常用命令

列出 preset：

```bash
python scripts/rsl_rl/train.py --list-exp
```

跑 smoke test：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/debug/fppo_smoke \
  --run_name smoke \
  --headless
```

跑最低难度盲走调试：

```bash
python scripts/rsl_rl/train_galileo_min_blind_walk.py
```

跑平地速度课程验证：

```bash
python scripts/rsl_rl/train_galileo_flat_speed_curriculum.py
```

跑平地全向课程验证：

```bash
python scripts/rsl_rl/train_galileo_flat_omni_curriculum.py
```

跑 rough terrain 课程验证：

```bash
python scripts/rsl_rl/train_galileo_terrain_curriculum.py
```

跑推荐 rough terrain 课程 v3：

```bash
python scripts/rsl_rl/train_galileo_terrain_curriculum_v3.py
```

跑 rough terrain v3.1 metric ablation：

```bash
python scripts/rsl_rl/train_galileo_terrain_curriculum_v31_displacement.py
```

跑 rough terrain v3.1 row bootstrap ablation：

```bash
python scripts/rsl_rl/train_galileo_terrain_curriculum_v31_row_bootstrap.py
```

跑 terrain row 信号诊断：

```bash
python scripts/rsl_rl/train_galileo_terrain_row_probe.py
```

跑 terrain perception 诊断：

```bash
python scripts/rsl_rl/train_galileo_terrain_perception_probe.py
```

跑正式 CTS benchmark：

```bash
python scripts/rsl_rl/train_galileo_cts_main.py
```

显式 preset 命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/debug/blind_walk \
  --run_name min_blind_walk \
  --headless
```

平地速度课程的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/curriculum/flat_speed \
  --run_name flat_speed_curriculum \
  --headless
```

平地全向课程的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/curriculum/flat_omni \
  --run_name flat_omni_curriculum \
  --headless
```

rough terrain 课程的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/curriculum/rough_foundation \
  --run_name terrain_curriculum \
  --headless
```

推荐 rough terrain 课程 v3 的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/curriculum/rough_progressive \
  --run_name terrain_curriculum_v3 \
  --headless
```

rough terrain v3.1 metric ablation 的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/legacy/rough_metric_displacement \
  --run_name terrain_curriculum_v31_displacement \
  --headless
```

rough terrain v3.1 row bootstrap ablation 的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/legacy/rough_row_bootstrap \
  --run_name terrain_curriculum_v31_row_bootstrap \
  --headless
```

terrain row 信号诊断的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/probes/terrain_row \
  --run_name terrain_row_probe \
  --headless
```

terrain perception 诊断的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/probes/terrain_perception \
  --run_name terrain_perception_probe \
  --headless
```

正式 CTS benchmark 的显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/benchmark/cts_main \
  --run_name cts_main \
  --headless
```

`galileo/benchmark/cts_main` 的最新版本参考了 `extreme_load` 的 blind-locomotion 课程形状：`terrain_levels` 仍然是主线，`lin_x / wz` 使用粗粒度 time-based level，`lin_y` 改为 terrain-specific command range，而不是单独挂一条 lateral curriculum。当前更推荐 fresh start，因为训练脚本还没有把模型和课程状态做成真正的一体恢复。

最新课程 run 的同条件键盘评测：

```bash
python scripts/rsl_rl/play_keyboard_galileo_flat.py \
  --checkpoint <checkpoint.pt>
```

```bash
python scripts/rsl_rl/play_keyboard_galileo_flat_omni.py \
  --checkpoint <checkpoint.pt>
```

```bash
python scripts/rsl_rl/play_keyboard_galileo_terrain.py \
  --checkpoint <checkpoint.pt>
```

这些 wrapper 会自动带上对应 `--exp`，并根据 checkpoint 所在 run 的 TensorBoard 事件恢复最新记录的命令课程状态，避免 play 还停留在初始窄指令范围。

跑公平对比：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo cpo \
  --exp galileo/benchmark/cts_main \
  --run_name cpo \
  --headless
```

## 什么时候用 preset

优先放进 preset 的改动：

- `num_envs`
- `max_iterations`
- `cost_limit`
- `step_size`
- terrain 难度
- 随机扰动范围
- curriculum 阈值
- reward / cost 系数

Galileo 课程与最近 run 诊断见 `docs/GALILEO_CURRICULUM.md`。推荐顺序是先跑 `galileo/debug/blind_walk` 稳定奖励、约束和算法参数，再跑 `galileo/curriculum/flat_speed` 和 `galileo/curriculum/flat_omni` 验证平地 gait 与全向指令能力，用 `galileo/probes/terrain_row` / `galileo/probes/terrain_perception` 拆分 row 信号与 blind perception 的责任，之后进入 `galileo/curriculum/rough_mainline` 做正式 rough curriculum，最后再跑 benchmark。`galileo/curriculum/rough_foundation` 和 `galileo/curriculum/rough_progressive` 继续保留作分阶段对照。

## 什么时候开分支

应走代码分支的改动：

- 新 reward / cost term
- 改 observation 维度
- 改 actor / critic 结构
- 改算法更新公式
- 改 rollout / storage / advantage 逻辑
- 改训练入口流程

## 推荐实际工作流

参数实验：

1. 保持主代码不动
2. 在 `experiments/galileo/` 新建 preset
3. 用 `--exp` 跑实验
4. 结果好就保留 preset，不好就删掉

代码实验：

1. 先打 baseline tag
2. 新建 `exp/...` 分支
3. 需要长期并行时用 `git worktree add`
4. 每完成一个小逻辑就 commit 一次
