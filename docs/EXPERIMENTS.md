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
    fppo_smoke.json
    cost_limit_relaxed.json
    terrain_hard.json
    studies/
      algo_compare_cts_fair.json
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
  --exp galileo/fppo_smoke \
  --run_name smoke \
  --headless
```

跑公平对比：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo cpo \
  --exp galileo/studies/algo_compare_cts_fair \
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
