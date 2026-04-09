# Experiment Presets

这个目录用于存放“参数版本”，而不是“代码分叉”。

- 纯参数试验：新增一个 preset 文件，不需要新建 git 分支。
- 代码逻辑试验：新建 git 分支，必要时配合 `git worktree`。
- 推荐把每个 preset 都写成“相对基线的差异”，避免复制整份大配置。

## Benchmark Presets

论文主实验建议直接从下面两份 preset 开始：

- `galileo/benchmark/teacher_main`
- `galileo/benchmark/student_main`

它们会继承公平对比 study preset，并固定主实验的 `experiment_name`、保存频率和 benchmark 元信息。

## 文件格式

- 支持 `json`
- 也支持 `toml`（需要环境里可用 `tomllib` 或 `tomli`）

## 约定结构

- `meta`：说明信息，不直接改配置
- `env`：覆盖环境配置
- `agent`：覆盖算法 / policy / runner 配置
- `extends`：可选，继承另一个 preset

示例：

```json
{
  "extends": "galileo/base",
  "meta": {
    "description": "Quick smoke test before a long run"
  },
  "env": {
    "scene": {
      "num_envs": 512
    }
  },
  "agent": {
    "max_iterations": 1000
  }
}
```

## 使用方式

列出所有 preset：

```bash
python scripts/rsl_rl/train.py --list-exp
```

训练时加载 preset：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --exp galileo/fppo_smoke \
  --run_name teacher \
  --headless
```

更详细的工作流和 Git 教程见 `docs/EXPERIMENTS.md`。
