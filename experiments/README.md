# Experiment Presets

这个目录用于存放“参数版本”，而不是“代码分叉”。

- 纯参数试验：新增一个 preset 文件，不需要新建 git 分支。
- 代码逻辑试验：新建 git 分支，必要时配合 `git worktree`。
- 推荐把每个 preset 都写成“相对基线的差异”，避免复制整份大配置。

## Preset Taxonomy

核心 preset：

- `galileo/debug/blind_walk`：最低难度盲走调试，先在纯平地、窄指令、无域随机化但保留确定性 reset 的设置下验证奖励、约束和算法参数。
- `galileo/debug/fppo_smoke`：最短链路 smoke test，用来确认配置和训练入口没有接坏。
- `galileo/curriculum/flat_speed`：平地速度课程验证，只打开 `lin_vel_x` 课程，检查机器人是否能在更宽的前向速度范围内形成基本 gait。
- `galileo/curriculum/flat_omni`：平地全向速度课程，在纯平地、无域随机化下依次放开 `vx / vy / wz`，先验证完整 teleop 指令能力。
- `galileo/curriculum/rough_foundation`：粗地形基础课程，训练从 flat-only terrain family 起步，再逐步解锁 rough terrain type；域随机化继续关闭，只单独验证 terrain 轴。
- `galileo/curriculum/rough_progressive`：推荐的 rough curriculum 中间阶段，先在 `random_rough` 上用 command-aligned row metric 完成 rough-row bootstrap，再逐步放开 omni 指令和更难 terrain family。
- `galileo/curriculum/rough_mainline`：当前 rough 主线课程，在已验证的 row-bootstrap 设置上给 `vy / wz` 一个可达的极小起始范围，让 omni gate 和 terrain-type gate 都能拿到真实 active samples。
- `galileo/probes/terrain_row`：terrain row 信号诊断，固定到单一 easy rough family，并把命令收成长时域前向行走，只检查 `terrain_levels_vel` 在 blind rough 任务里能否真正爬升。
- `galileo/probes/terrain_perception`：在 `galileo/probes/terrain_row` 的基础上，临时给 student actor 打开 `height_scan`，用来区分“课程信号失真”和“blind rough perception 难度过高”。
- `galileo/benchmark/cts_main`：正式 CTS benchmark。当前课程骨架参考了更简洁的开源 locomotion recipe，用来做最终长程训练与算法对比。
- `galileo/studies/algo_compare_cts`：历史入口名，当前通过 alias 兼容到 `galileo/benchmark/cts_main`。

非核心但保留的 preset：

- `galileo/legacy/rough_metric_displacement`：历史 ablation，只把 row metric 切回 `displacement`，用于判断 `command_projected` 是否过于保守。
- `galileo/legacy/rough_row_bootstrap`：历史 ablation，把 stage-0 `vx` 固定成 row-probe 风格的前向几何，用来判断宽 `lin_x` 是否拖慢 rough-row bootstrap。
- `galileo/legacy/terrain_hard`：更激进的 terrain 初始化采样，只建议做定向对照。
- `galileo/variants/cost_limit_relaxed`：放松 cost limit 的参数变体，用于专项 sweep。

它们统一基于 CTS blind-locomotion task，不再区分 teacher/student 两套实验树。正式 benchmark 前建议按 `galileo/debug/blind_walk -> galileo/curriculum/flat_speed -> galileo/curriculum/flat_omni -> galileo/probes/terrain_row / galileo/probes/terrain_perception -> galileo/curriculum/rough_mainline -> galileo/benchmark/cts_main` 的顺序推进，避免在基础行走未稳定时同时叠加地形、指令和域随机化难度。

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
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/debug/fppo_smoke \
  --run_name smoke \
  --headless
```

更详细的工作流和 Git 教程见 `docs/EXPERIMENTS.md`。
