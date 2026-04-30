# Galileo 课程设计与训练诊断

## 最新 Run 诊断

本地最新训练输出主要有五组：

- `logs/rsl_rl/galileo_flat_speed_curriculum/2026-04-26_00-03-45_galileo-flat-speed-curriculum_flat_speed_curriculum`
  - 当前最稳定的平地速度课程 run，训练到 `8318` iter。
  - 末尾 `Train/mean_reward=34.552`、`mean_episode_length=993.63`、`error_vel_xy=0.0768`、`error_vel_yaw=0.1036`、`body_contact=0.0010`、`cost_violation_rate=0.0`。
  - `lin_vel_x_command_threshold=1.0` 已经完全放开，说明 reward、约束和 FPPO 主体在最小平地 gait 场景里是成立的。
- `logs/rsl_rl/galileo_flat_omni_curriculum/2026-04-26_11-11-00_galileo-flat-omni-curriculum_flat_omni_curriculum`
  - 最新平地全向课程 run，当前训练到 `803` iter。
  - 末尾 `Train/mean_reward=29.087`、`mean_episode_length=1000.0`、`error_vel_xy=0.1788`、`error_vel_yaw=0.1945`、`body_contact=0.0085`、`cost_violation_rate=0.0`。
  - 这条 run 已经学会“稳定行走”，但 `lin_vel_y_command_threshold=0.0`、`ang_vel_z_command_threshold=0.0`，说明全向课程尚未真正解锁。
- `logs/rsl_rl/galileo_terrain_curriculum/2026-04-26_11-11-35_galileo-terrain-curriculum_terrain_curriculum`
  - 最新 rough terrain 课程 run，当前训练到 `797` iter。
  - 末尾 `Train/mean_reward=29.536`、`mean_episode_length=999.78`、`error_vel_xy=0.1669`、`error_vel_yaw=0.1874`、`body_contact=0.0077`、`cost_violation_rate=0.0`。
  - `lin_vel_x_command_threshold=1.0` 但 `lin_vel_y_command_threshold=0.0`、`ang_vel_z_command_threshold=0.0`、`terrain_type_progression=0.0`，说明 rough run 现在本质上还停留在“flat-only bootstrap”。
- 需要注意的配置问题：
  - `flat_omni_curriculum` 和 `terrain_curriculum` 原先把 lateral / yaw curriculum 的 `min_command_speed` 设成了 `0.06 / 0.12`，但启动指令范围只有 `vy=+-0.03`、`wz=+-0.04`，active ratio 会永久为 `0`，课程逻辑物理上无法推进。
  - 这个门槛现已收回到可达的 bootstrap 范围内，后续应重新起一条 `flat_omni` 和 `terrain` run 来验证课程能否真正解锁。

- `logs/rsl_rl/galileo_benchmark_cts_main/2026-04-25_22-00-11_galileo-benchmark-cts-main_fppo_main`
  - 当前最新正式训练 run，训练到 `662` iter。
  - 末尾 `Train/mean_reward=0.106`、`mean_episode_length=95.48`、`body_contact=0.468`、`bad_orientation=0.530`、`cost_violation_rate=0.562`。
  - `terrain_levels` 基本仍在 `0`，`lin_x_level=0.30`、`ang_z_level=0.30`，说明这次 run 还处在“基础步态没站稳，课程也还没真正推起来”的阶段。
- `logs/rsl_rl/galileo_benchmark_cts_main/2026-04-22_23-38-33_galileo-benchmark-cts-main_fppo_main`
  - 旧的长跑 baseline，训练到 `10731` iter。
  - 末尾 `mean_episode_length=412.85`、`terrain_levels=2.70`、`lin_x_level=0.69`、`ang_z_level=0.77`，同时 `cost_violation_rate=0.686`、`error_vel_xy=0.952`、`error_vel_yaw=0.402`。
  - 这说明旧课程会在基础步态和 tracking 还没稳住前，就把地形和指令同时推高。
- `logs/rsl_rl/galileo_min_blind_walk/2026-04-25_23-11-45_galileo-min-blind-walk_min_blind_walk`
  - 第一版最低课程 run。
  - 末尾 `mean_episode_length=1.0`、`body_contact=0.925`、`bad_orientation=0.712`。
  - 这次 run 应视为无效样本：不是奖励本身有结论，而是 preset 误把必要 reset 一起关掉了。
- `logs/rsl_rl/galileo_min_blind_walk/2026-04-25_23-22-57_galileo-min-blind-walk_min_blind_walk`
  - 修复 deterministic reset 后的 2-iter smoke run。
  - 末尾 `mean_episode_length=28.26`、`body_contact=0.0078`、`bad_orientation=0.0`、`cost_violation_rate=0.0`。
  - 这还不足以评价奖励优劣，但已经证明最低课程 preset 本身是可训练、可复位的。

结论：先用最低课程难度消除地形和域随机化影响，调出可靠的奖励、约束和算法参数；再按 `flat_speed -> flat_omni -> terrain -> full CTS` 的顺序逐轴放开难度。

## 课程轴

当前仍保留单阶段训练和同一套 CTS/FPPO 算法框架，不引入多阶段切换。

- 地形课程：训练初始从 flat-only terrain family 起步；只有基础全向 tracking 稳定后才解锁 rough terrain type，随后 `terrain_levels_vel` 再在已解锁 family 内按行走距离推进 row difficulty。
- 地形课程采用 stage-wise gate，而不是单个全局阈值：flat-only bootstrap 阶段不再要求 `terrain_levels` 先冲到固定均值，同时允许比 rough stage 略宽松的 tracking 门槛；后续 rough stage 再结合 row progress、tracking 误差和最小停留时间推进，并保留 stall fallback，避免 terrain family 被永久卡死。
- 指令课程：`lin_vel_x`、`lin_vel_y`、`ang_vel_z` 都从窄范围开始；只有追踪误差、活跃指令比例和最小推进间隔满足后才扩展范围。
- 域随机化课程：质量、质心、摩擦、reset 姿态/速度、关节扰动、执行器增益和 push 从确定性设置开始；只有基础指令能力和 terrain type 课程都稳定后才逐步放开到完整范围。

## 最低难度盲走

用途：只验证盲走本体能力，先调奖励、约束和算法参数，不让地形和域随机化干扰判断。

```bash
python scripts/rsl_rl/train_galileo_min_blind_walk.py
```

等价显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/debug/blind_walk \
  --run_name min_blind_walk \
  --headless
```

最低难度 preset 做了这些限制：

- 纯平地：只保留 `flat`，关闭 terrain curriculum。
- 窄指令：`vx=(-0.08, 0.18)`、`vy=(-0.03, 0.03)`、`wz=(-0.04, 0.04)`。
- 无域随机化：关闭质量、质心、摩擦、执行器增益和 push 随机化。
- 保留确定性 reset：基座位姿、基座速度和腿部关节在每次 reset 时回到固定初值，避免调参 run 因为“没真正复位”而失真。
- 训练入口不变：仍使用 `Isaac-Galileo-CTS-v0`、CTS observation/reward/cost、FPPO runner。

## 平地速度课程

用途：在保持平地、确定性动力学和固定 reset 的前提下，只打开 `lin_vel_x` 课程，验证机器人是否能随着前向速度范围扩大而形成基本 gait。

```bash
python scripts/rsl_rl/train_galileo_flat_speed_curriculum.py
```

等价显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/curriculum/flat_speed \
  --run_name flat_speed_curriculum \
  --headless
```

这个 preset 的特点：

- 地形仍然固定为 `flat`，不引入 terrain curriculum。
- 域随机化仍然关闭，避免把 gait 形成和鲁棒性学习混在一起。
- 只打开 `lin_vel_x_command_threshold`，并移除 terrain gate，让速度课程能在 flat 模式下真正推进。
- `lin_vel_y` 和 `ang_vel_z` 继续锁在最小范围，先单独回答“前向基本运动能力”。
- `lin_vel_x` 从 `(-0.08, 0.18)` 逐步扩展到 `(-0.4, 0.8)`，属于“中等速度”验证，不直接跳到正式训练全范围。

手动评估可以配合纯平地键盘遥控：

```bash
python scripts/rsl_rl/play_keyboard_galileo_flat.py \
  --checkpoint <checkpoint.pt>
```

这个 wrapper 会自动带上 `--exp galileo/curriculum/flat_speed`，并从 checkpoint 所在 run 的 TensorBoard 事件里恢复最新的 `lin_vel_x` 课程状态，再把平地键盘速度幅值对齐到当前 run，避免 checkpoint 正确但 play 命令分布还停留在初始窄范围。

## 平地全向课程

用途：在保持纯平地、确定性动力学和固定 reset 的前提下，把 `vx / vy / wz` 三轴指令课程都打开，先验证完整 teleop 命令能力，再进入 rough terrain。

```bash
python scripts/rsl_rl/train_galileo_flat_omni_curriculum.py
```

等价显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/curriculum/flat_omni \
  --run_name flat_omni_curriculum \
  --headless
```

这个 preset 的特点：

- 地形固定为 `flat`，不引入 terrain curriculum。
- 域随机化继续关闭，避免把 rough terrain 和动力学鲁棒性同时引入。
- `lin_vel_x / lin_vel_y / ang_vel_z` 三条课程都在 flat 条件下推进，先单独验证全向跟踪与站立。

手动评估建议使用全向平地 wrapper：

```bash
python scripts/rsl_rl/play_keyboard_galileo_flat_omni.py \
  --checkpoint <checkpoint.pt>
```

这个 wrapper 会自动带上 `--exp galileo/curriculum/flat_omni`，并恢复 checkpoint 所在 run 最新记录的 `vx / vy / wz` 课程状态。

## Rough Terrain 课程

用途：在 rough terrain 上保留确定性动力学和固定 reset，让训练从 flat-only terrain family 起步，再逐步解锁 rough terrain type，单独验证 terrain 轴。

```bash
python scripts/rsl_rl/train_galileo_terrain_curriculum.py
```

等价显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/curriculum/rough_foundation \
  --run_name terrain_curriculum \
  --headless
```

这个 preset 的特点：

- 起始时虽然场景已经是 rough generator，但 terrain-type curriculum 会先把所有 env 约束到 `flat` 列。
- `vx / vy / wz` 先在 flat-only 阶段完成大部分扩展，再逐步解锁 smooth rough、boxes、stairs 等 terrain family。
- `terrain_levels_vel` 仍负责 row difficulty 递进，但 terrain family unlock 改为 stage-wise threshold：flat bootstrap 阶段不依赖 terrain row 均值，后续 rough stage 只要求适度 row progress；若某个 stage 长时间满足 tracking 但仍卡住，会触发 stall fallback，避免课程死锁。
- 域随机化继续关闭，避免把 terrain 学习和鲁棒性学习混在一起。

手动评估建议使用 rough terrain wrapper：

```bash
python scripts/rsl_rl/play_keyboard_galileo_terrain.py \
  --checkpoint <checkpoint.pt>
```

这个 wrapper 会保留 preset 对应的 rough terrain generator，并恢复 checkpoint 所在 run 最新记录的命令课程状态。

## Terrain 诊断 Probe

当 `terrain_curriculum` 已经证明 reward、约束和 FPPO 本体没有明显炸掉，但 `terrain_levels` 长时间不抬升时，推荐先跑下面两条 probe，而不是立刻同时改 reward 和算法。

### Terrain Row Probe

用途：只回答一个问题: `terrain_levels_vel` 在 blind rough 任务里是不是还代表“真的学会更难 row 了”。

```bash
python scripts/rsl_rl/train_galileo_terrain_row_probe.py
```

等价显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/probes/terrain_row \
  --run_name terrain_row_probe \
  --headless
```

这个 preset 只做三件事：

- 把 terrain mix 收成单一 `random_rough` family，避免 terrain type 解锁干扰 row signal。
- 关掉 `terrain_type_progression` 和三条 command curriculum，只保留 `terrain_levels_vel`。
- 把命令改成 `10s` 重采样、低 standing ratio、纯前向行走，避免 omni 指令和净位移抵消把 row progress 假性压平。

判读：

- 如果这条 run 里 `terrain_levels` 很快抬升，主锅在当前 rough curriculum 的 row 信号设计，而不是 FPPO。
- 如果这条 run 里 `terrain_levels` 仍然抬不起来，问题就更像是 blind rough locomotion 本体、rough reward shape，或者 `terrain_levels_vel` 更新逻辑本身。

### Terrain Perception Probe

用途：在 `galileo/probes/terrain_row` 的基础上，单独测试 blind perception 是不是 rough terrain 的主瓶颈。

```bash
python scripts/rsl_rl/train_galileo_terrain_perception_probe.py
```

等价显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/probes/terrain_perception \
  --run_name terrain_perception_probe \
  --headless
```

这个 preset 仅在 `galileo/probes/terrain_row` 基础上额外给 student actor 打开 `height_scan`，其余 rough terrain、reward、约束和 FPPO 设置保持一致。

判读：

- `galileo/probes/terrain_row` 差，但 `galileo/probes/terrain_perception` 好：主锅偏向 blind rough perception。
- 两条 probe 都差：主锅偏向 row 更新逻辑、rough reward shape，或者地形本身太难。
- 两条 probe 都好：说明正式 rough curriculum 叠了太多门控，不是 locomotion 本体学不会。

## Rough Terrain Curriculum v3

probe 跑通以后，推荐不要直接继续沿用旧的 `galileo/curriculum/rough_foundation`，而是切到新的 `galileo/curriculum/rough_progressive`。

```bash
python scripts/rsl_rl/train_galileo_terrain_curriculum_v3.py
```

等价显式命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/curriculum/rough_progressive \
  --run_name terrain_curriculum_v3 \
  --headless
```

这版 preset 的重点是：

- `terrain_levels_vel` 不再只看 reset 时的原点净位移，而是按整段 episode 的 command-aligned planar progress 推进 row difficulty。
- terrain stage 改成 `random_rough -> random_rough(omni reopen) -> slopes -> boxes -> stairs`，先把 rough row 学稳，再继续叠几何难度。
- `vy / wz` 课程不再和 rough-row bootstrap 绑死在同一时刻，而是在 `terrain_levels` 达到稳定阈值后才开始放开。
- 旧 `terrain_curriculum` 保留作对照，不再作为默认推荐入口。

如果 `galileo/curriculum/rough_progressive` 的训练主体稳定，但 `terrain_levels` 仍然抬得很慢，下一步建议直接跑下面两条 legacy A/B，而不是回去先改 reward 或 FPPO：

1. `galileo/legacy/rough_metric_displacement`
   - 只把 row metric 从 `command_projected` 切回 `displacement`
   - 用来判断是不是 `command_projected` 太保守，导致 rough-row bootstrap 迟迟过不了 gate

2. `galileo/legacy/rough_row_bootstrap`
   - 在 `v31_displacement` 基础上，把 stage-0 `vx` 固定到 `0.25~0.75` 的前向区间
   - 用来判断是不是过早放宽 `lin_x` 几何，把 rough-row progress 稀释掉了

## Rough Terrain Curriculum v3.2

最新 `terrain_curriculum_v31_row_bootstrap_same_family_fix` 已经证明：在单一 `random_rough` family 内，保留同 family row、使用 `displacement` row metric，并固定 `vx=0.25~0.75` 后，`terrain_levels` 可以升到约 `6`。但这条 run 仍停在 `terrain_type_progression=0`、`lin_vel_y_command_threshold=0`、`ang_vel_z_command_threshold=0`，原因是 `vy / wz` 起始指令为 0，而 gate 又要求非零 active command sample。

下一步切到 v3.2：

```bash
python scripts/rsl_rl/train_galileo_terrain_curriculum_v32_omni_unlock.py \
  --logger wandb \
  --log_project_name galileo_cts \
  --run_name terrain_curriculum_v32_omni_unlock
```

这版 preset 的变化：

- 继续继承 `v31_row_bootstrap` 已验证的 rough-row bootstrap：`random_rough`、`displacement` row metric、`vx=0.25~0.75`、无域随机化。
- 给 `vy` 和 `wz` 一个可达的极小起始范围：`vy=(-0.03, 0.03)`、`wz=(-0.04, 0.04)`。
- 把 `vy / wz` gate 的 `min_command_speed` 收到起始范围内，让 active ratio 不再永久为 0。
- 保持 terrain family stage-wise 解锁：先看 `terrain_levels` 是否继续维持在高 row，再看 `vy / wz` 是否推进，最后看 `terrain_type_progression` 是否进入后续 stage。

判读重点：

- 好信号：`terrain_levels` 维持 `4+`，随后 `lin_vel_y_command_threshold` 和 `ang_vel_z_command_threshold` 从 `0` 开始上升，`terrain_type_progression` 至少推进到 `1`。
- 可接受风险：刚打开 `vy / wz` 时 `error_vel_xy`、`error_vel_yaw` 和 `body_contact` 短期变差，但 episode length 应保持在 `900+` 附近。
- 坏信号：`terrain_levels` 仍高，但 `vy / wz` 长时间为 `0`，说明 gate 仍过严；若 `vy / wz` 上升后 `body_contact` 明显失控，下一步应先收窄 `max_curriculum_lin_y / max_curriculum_ang_z` 或放慢 step，而不是立刻解锁更多 terrain family。

## 正式训练

建议顺序是：`最低难度盲走 -> 平地速度课程 -> 平地全向课程 -> terrain probes -> rough_mainline -> 正式 benchmark`。这样每一步只新增一个主要难度来源，更容易定位到底是 reward、command、terrain 还是 domain randomization 在拖训练。

正式 benchmark 命令：

```bash
python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CTS-v0 \
  --algo fppo \
  --exp galileo/benchmark/cts_main \
  --run_name fppo_main \
  --headless \
  --logger wandb \
  --max_iterations 20000 \
  --log_project_name galileo_cts
```

正式训练现在会吸收 v3.2 的 rough-row bootstrap 经验，但课程骨架进一步参考了 `extreme_load` 的 blind locomotion：起步仍使用 `vx=(0.25, 0.75)` 的前向 rough 行走区间和 `displacement` row metric，不过 `lin_y` 不再单独做一条 curriculum，而是交给 terrain-specific command range；`lin_x / wz` 则改成更粗粒度的 time-based level。这样课程主链路被收成 `terrain row -> coarse command widening -> terrain family -> delayed DR`，更像成熟开源实现的“少而硬”风格。

正式训练建议先关注五类指标：`Train/mean_episode_length`、`Episode_Termination/fallen`、`body_contact`、`Curriculum/terrain_type_progression`、`Curriculum/domain_randomization_scale`。若 episode length 未稳定增长，或者 terrain-type stage 长时间卡住，不建议再继续增加 gate；优先回看 terrain-specific command ranges 是否过宽。若 `domain_randomization_scale` 开始抬升后姿态明显变差，应先减慢域随机化课程，而不是继续提高 terrain 难度。

当前这版正式训练更推荐从 0 起跑，而不是从 `galileo/curriculum/rough_mainline` resume。原因不是 rough-row bootstrap 无用，而是训练脚本暂时还不能把模型和课程状态做成完全一致的恢复；如果只 warm-start 权重，反而容易把策略丢回不匹配的课程分布里。

这样做的好处是，正式训练阶段只需要回答最后两个问题：

- terrain family 到最终 stage 后，`domain_randomization_scale` 能不能开始上升；
- DR 接回后，`episode_length / body_contact / fallen` 会不会明显恶化。
