# Galileo 算法对比实验计划

这份文档把当前研究计划整理成可直接运行的 preset 和命令。

## 1. 总体结构

当前实验分成两条主线：

- **Teacher 对比实验**
- **Student 对比实验**

---

## 2. Teacher 对比实验

### 2.1 公平对比主组

使用同一套 **多奖励、少约束** Teacher 配置的算法：

- `ppo`
- `fppo`
- `np3o`
- `ppo_lagrange`
- `cpo`
- `pcpo`
- `focops`

对应 preset：

- `experiments/galileo/studies/algo_compare_teacher_fair.json`

这套 preset 的含义：

- 使用 Teacher 侧共享奖励项（所有算法共用同一套 gait-tuned reward 参数）
- 只保留三项共享约束：
  - `joint_pos_prob_constraint`
  - `joint_vel_prob_constraint`
  - `joint_torque_prob_constraint`
- 因此 PPO / FPPO / 其他对比算法的 reward / cost 参数保持一致，保证公平对比

### 2.2 NP3O 保留组

为了保留你原来那套“多约束、少奖励”的 NP3O 设置，单独保留：

- `experiments/galileo/studies/algo_compare_teacher_np3o_baseline.json`

它只给 `np3o` 用，方便你比较：

- `np3o` 在公平配置下的表现
- `np3o` 在原始 baseline 配置下的表现

### 2.3 Teacher 训练命令

公平对比主组模板：

```bash
LOG_RUN_NAME=<algo> python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo <algo> \
  --exp galileo/studies/algo_compare_teacher \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

Teacher: fppo

```bash
LOG_RUN_NAME=fppo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo fppo \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 4096 \
  --max_iterations 20000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

Teacher: ppo

```bash
LOG_RUN_NAME=ppo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo ppo \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

Teacher: pcpo

```bash
LOG_RUN_NAME=pcpo python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo pcpo \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

Teacher: np3o

```bash
LOG_RUN_NAME=np3o python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo np3o \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

NP3O baseline：

```bash
LOG_RUN_NAME=np3o_baseline python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Teacher-v0 \
  --algo np3o \
  --exp galileo/studies/algo_compare_teacher_np3o_baseline \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name teacher \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

### 2.4 Teacher 可视化命令

Teacher 可视化模板：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CRL-Teacher-Play-v0 \
  --algo <algo> \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 16 \
  --checkpoint <teacher_checkpoint>.pt \
  --force_gui
```

其中：

- `--task` 必须使用 `Isaac-Galileo-CRL-Teacher-Play-v0`
- `--checkpoint` 要填 Teacher 自己训练出来的模型
- `--algo` 要与 checkpoint 对应算法一致，例如 `fppo` / `ppo` / `np3o`
- 当前仓库没有 `distillation` 这个算法名；Student 蒸馏算法对应的是 `dagger`
- 如果你要看的是 Teacher，就不要使用 `Isaac-Galileo-CRL-Student-Play-v0`
- 普通 `play.py` 回放默认不会导出模型，所以 Teacher 可直接可视化

例如可视化 `fppo` Teacher：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CRL-Teacher-Play-v0 \
  --algo fppo \
  --exp galileo/studies/algo_compare_teacher_fair \
  --num_envs 16 \
  --force_gui \
  --checkpoint logs/rsl_rl/galileo_algo_compare_teacher_fair/

```

如果当前机器没有图形界面，可以去掉 `--force_gui`，改用 `--video --video_length 1000 --headless` 录制回放。

---

## 3. Student 对比实验

### 3.1 Student 公平对比 preset

对应 preset：

- `experiments/galileo/studies/algo_compare_student_fair.json`

适用算法：

- `ppo`
- `fppo`
- `np3o`
- `ppo_lagrange`
- `cpo`
- `pcpo`
- `focops`
- `dagger`

### 3.2 Student 为什么不需要额外 reward/cost 覆写

因为 Student 默认配置本身就是你要的形式：

- `rewards: StudentRewardsCfg = StudentRewardsCfg()`
- `costs: StudentCostsCfg = StudentCostsCfg()`

对应位置：

- `crl_tasks/crl_tasks/tasks/galileo/config/student_env_cfg.py`
- `crl_tasks/crl_tasks/tasks/galileo/config/mdp_cfg.py`

也就是说，Student 默认就是：

- 相对密集的 reward
- 只有三项 joint feasibility constraints

所以这个 preset 的主要作用是：

- 固定实验命名
- 给 student 算法对比统一归档
- 避免以后你自己手动写很多重复命令

### 3.3 Student 训练命令

Student RL 算法模板：

```bash
LOG_RUN_NAME=<algo> python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Student-v0 \
  --algo <algo> \
  --exp galileo/studies/algo_compare_student_fair \
  --num_envs 4096 \
  --max_iterations 50000 \
  --run_name student \
  --headless \
  --logger wandb \
  --log_project_name galileo_teacher
```

例如 `fppo`： (DAgger)

```bash
LOG_RUN_NAME=fppo_student_dagger python scripts/rsl_rl/train.py \
  --task Isaac-Galileo-CRL-Student-v0 \
  --algo dagger \
  --exp galileo/studies/algo_compare_student_distill_tuned \
  --num_envs 4096 \
  --max_iterations 15000 \
  --run_name student_dagger \
  --headless \
  --checkpoint logs/rsl_rl/galileo_algo_compare_teacher_fair/ \
  --logger wandb \
  --log_project_name galileo_teacher

```

### 3.4 Student 可视化命令

Student 可视化模板：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CRL-Student-Play-v0 \
  --algo <algo> \
  --exp <same_exp_as_training> \
  --num_envs 16 \
  --checkpoint <student_checkpoint>.pt \
  --force_gui
```

其中：

- 普通 Student RL 算法通常使用 `--exp galileo/studies/algo_compare_student_fair`
- DAgger Student 通常使用 `--algo dagger`
- DAgger Student 推荐使用 `--exp galileo/studies/algo_compare_student_distill_tuned`
- `--checkpoint` 要填 Student 自己训练出来的模型，而不是 Teacher checkpoint
- 这里的 `--checkpoint` 语义和训练时不同：`play.py` / `evaluation.py` / `demo.py` 中加载的是 Student policy

例如 `dagger` Student 可视化：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CRL-Student-Play-v0 \
  --algo dagger \
  --exp galileo/studies/algo_compare_student_distill_tuned \
  --num_envs 16 \
  --force_gui \
  --checkpoint logs/rsl_rl/galileo_algo_compare_student_fair
```

如果当前机器没有图形界面，可以去掉 `--force_gui`，改用 `--video --video_length 1000 --headless` 录制回放。

### 3.5 Student 模型导出命令

当前仓库没有单独的 export 入口；需要通过 `scripts/rsl_rl/play.py --export_only` 导出模型。

这里同样要求 `--checkpoint` 指向 Student 自己训练出的模型文件。

`--export_only` 在导出完成后会直接退出，不进入回放循环。

推荐这样运行：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CRL-Student-Play-v0 \
  --algo <algo> \
  --exp <same_exp_as_training> \
  --num_envs 1 \
  --checkpoint <student_checkpoint>.pt \
  --export_only \
  --headless
```

这条命令会在 checkpoint 所在目录下自动生成：

- `exported_policy/policy.onnx`
- `exported_policy/policy.yaml`

例如 `dagger` Student 导出：

```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Galileo-CRL-Student-Play-v0 \
  --algo dagger \
  --exp galileo/studies/algo_compare_student_distill_tuned \
  --num_envs 1 \
  --export_only \
  --headless \
  --checkpoint logs/rsl_rl/galileo_algo_compare_student_fair/
```

### 3.6 DAgger 说明

`dagger` 也可以复用这个 Student preset，但它和普通 RL 算法不同：

- 需要 teacher checkpoint
- 训练时需要配合 `--load_run` / `--checkpoint` 使用
- 如果你想专门改善 student 的指令跟踪，推荐使用
  `experiments/galileo/studies/algo_compare_student_distill_tuned.json`
  这个 tuned preset。它只改 student DAgger 相关参数，不影响 teacher preset。

因此更推荐把它看成 Student 侧的单独子实验。

---

## 4. 遥控可视化

### 4.1 教师

```bash
python scripts/rsl_rl/play_keyboard.py \
  --task Isaac-Galileo-CRL-Teacher-Play-v0 \
  --force_gui \
  --real-time \
  --debug-keys \
  --algo fppo \
  --exp galileo/studies/algo_compare_teacher_fair \
  --checkpoint logs/rsl_rl/galileo_algo_compare_teacher_fair/
```

### 4.2 学生

```bash
python scripts/rsl_rl/play_keyboard.py \
  --task Isaac-Galileo-CRL-Student-Play-v0 \
  --force_gui \
  --algo dagger \
  --exp galileo/studies/algo_compare_student_distill_tuned \
  --checkpoint logs/rsl_rl/galileo_algo_compare_student_fair/
```

如果想调台阶强度，可以加：

```bash
--step-height 0.18 --stair-steps 10
```

W/S 或 方向键上/下：前进/后退
A/D：左/右平移
Q/E 或 方向键左/右：左/右转
Space：清零当前指令
R：重置机器人
Esc：退出

## 4. 实验开始前怎么检查 preset 是否真的生效

训练时脚本会：

1. 先读取 task 默认配置
2. 再应用 `--algo`
3. 再应用 `--exp`
4. 最后把最终结果写入日志目录

训练后重点看：

- `logs/rsl_rl/<experiment>/<run>/params/env.yaml`
- `logs/rsl_rl/<experiment>/<run>/params/agent.yaml`
- `logs/rsl_rl/<experiment>/<run>/params/experiment.json`

其中 `env.yaml` 是本次训练真正生效的 reward/cost 配置。

---

## 5. 推荐执行顺序

### 第一步：先 smoke test

先把 `max_iterations` 改成 `1000`，每个算法快速跑一下，确认：

- 没有配置报错
- 日志目录命名正常
- `params/env.yaml` 和 `params/experiment.json` 正常生成

### 第二步：再跑正式实验

正式实验统一使用：

- 相同 task
- 相同 `num_envs`
- 相同 `max_iterations`
- 相同 wandb project
- 不同 `LOG_RUN_NAME`

### 第三步：补多 seed

建议对表现最好的几种算法再补 `3` 个 seed。

---

## 6. 组件消融实验建议

如果目标不是“证明某个 trick 都有帮助”，而是想找出真正必要的模块，建议按下面的顺序做：

1. 先用现有 WandB 曲线筛掉明显“没被触发”的 trick
2. 再做 `1 seed x 短程` 的快速消融筛查
3. 只有对结果敏感的项目，才做 `3 seeds x 正式训练`

### 6.1 最值得先做的消融表

| 优先级 | 组件 / trick                                       | 范围                   | 为什么值得做                                                               | 推荐消融方式                                                               | 现有 WandB 是否基本够用 | 主要判据                                                                                                |
| ------ | -------------------------------------------------- | ---------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------- |
| P0     | FPPO 投影修正本体                                  | Teacher / Student FPPO | 这是 FPPO 和 PPO 的核心区别，最先要证明                                    | 先做 `fppo vs ppo`；若想更纯粹，再补“predictor-only, no corrector”开关 | 部分够用                | `reward`、`cost_violation_rate`、`current_max_violation`、`accept_rate`、`active_constraints` |
| P0     | 自适应步长 `step_size_adaptive`                  | FPPO                   | 这是典型可能“复杂但未必必要”的模块                                       | `step_size_adaptive=False`                                               | 够用                    | `step_size`、`effective_step_ratio`、`accept_rate`、`infeasible_batch_rate` 是否明显变化        |
| P0     | 约束 curriculum `adaptive_constraint_curriculum` | FPPO                   | 如果几乎从不 tighten，就很可能是冗余逻辑                                   | 关闭 curriculum，并令起始/最终 limit 相同                                  | 够用                    | `curriculum_progress`、`curriculum_tighten_count`、`curriculum_gate_*`、reward/cost 曲线          |
| P0     | Teacher 对称增强 `symmetry_cfg`                  | Teacher                | Teacher 已启用左-右镜像增强，可能和 gait reward 存在功能重叠               | 关闭 symmetry augmentation                                                 | 基本够用                | `reward`、`error_vel_xy`、`prob_joint_*`、步态相关 cost/reward、最终动作观感                      |
| P0     | Student 重构辅助损失 `reconstruction_loss_coef`  | Student RL / DAgger    | 当前 student 默认带 reconstruction，很可能是“有损失下降但不提性能”的典型 | `reconstruction_loss_coef=0`                                             | 够用                    | `Loss/reconstruction` 下降是否真的带来 reward / tracking / behavior 改善                              |
| P0     | 奖励项裁剪 / 删除                                  | Teacher / Student      | 最容易藏冗余，尤其是权重很小的 shaping 项                                  | 把单个 reward 项权重置 0，逐项筛                                           | 基本够用                | 该 reward 项的 episodic 曲线量级、移除后 reward/cost/稳定性是否几乎不变                                 |
| P1     | Predictor KL 控制                                  | FPPO                   | 包括 predictor 自适应 lr 和 hard stop，可能过于保守                        | 放宽 `predictor_kl_hard_limit`，或关掉自适应 KL 调度                     | 够用                    | `predictor_kl`、`predictor_stop_rate`、`predictor_lr` 是否长期接近“未触发”                      |
| P1     | `use_clipped_surrogate`                          | FPPO                   | predictor 里再做 PPO clip 可能是重复稳健化                                 | `use_clipped_surrogate=False`                                            | 够用                    | `predictor_kl`、reward、cost violation 是否明显恶化                                                   |
| P1     | 多 cost-head critic                                | 所有 constrained 算法  | 当前会按 active cost term 自动推成多头，未必一定比单头更值                 | 显式固定 `num_cost_heads=1` 对比当前多头                                 | 不完全够用              | reward/cost 能看大方向，但最好再补每个 cost head 的拟合误差                                             |
| P1     | Student 的 constraint adapter                      | Student FPPO           | Student 侧会启用 cost 归一化/聚合，复杂度不低                              | 关闭 `constraint_adapter`                                                | 不完全够用              | 需要看 reward/cost，也建议补 raw vs normalized cost 曲线                                                |
| P1     | Student 动作延迟与动作历史                         | Student                | 真实部署有意义，但训练里可能和 history encoder / reconstruction 重叠       | 关闭 `use_delay`，或把 `history_length` 从 8 降到 1                    | 不够                    | 需要补 action smoothness / slip / 恢复能力评测                                                          |
| P1     | Student 域随机化与 push 扰动                       | Student                | 对鲁棒性可能有帮助，但对训练曲线不一定体现                                 | 分组关掉 mass/com/material/push                                            | 不够                    | 训练曲线不够，必须补鲁棒性评测                                                                          |
| P2     | 命令 curriculum 节流逻辑                           | Teacher / Student      | 可能只是“让训练更平滑”，也可能拖慢上限                                   | 关掉 gated progression 或放宽 gate                                         | 不完全够用              | 最好补 command level 曲线；只看 reward 容易误判                                                         |
| P2     | DAgger 的 teacher mixing schedule                  | Student DAgger         | 有可能 handoff 过程过长，造成冗余 teacher 依赖                             | 缩短或取消 `teacher_action_ratio` 衰减                                   | 部分够用                | `teacher_action_ratio`、behavior loss、student reward、最终 student 独立表现                          |

### 6.2 先验上可以降优先级的项

从当前代码看，下面几项不是最优先的消融对象：

- Teacher 侧 `constraint_adapter` 当前是关闭的，所以它不影响 Teacher 结果
- Student 的 `action_smoothness_l2` 现已从 reward 配置中移除
- Teacher / Student runner 里的 `empirical_normalization` 当前默认都是 `False`

这些项更适合放到“确认主结论以后再清理代码”的阶段处理。

---

## 7. 哪些 trick 可以直接用 WandB 曲线判断

下面这些模块，现有日志已经基本足够判断它们到底有没有在起作用。

### 7.1 FPPO 投影修正是否真的在工作

重点看：

- `Compare/FPPO/accept_rate`
- `Compare/FPPO/effective_step_size`
- `Compare/FPPO/effective_step_ratio`
- `Compare/FPPO/active_constraints`
- `Compare/FPPO/infeasible_batch_rate`
- `Compare/FPPO/current_max_violation`

判读方法：

- 如果 `active_constraints` 长期接近 `0`，`effective_step_ratio` 长期接近 `1`，`infeasible_batch_rate` 几乎为 `0`，说明 corrector 很少真正介入
- 如果 corrector 频繁介入，同时 `current_max_violation` 被压下去而 reward 没明显受损，说明它是有效模块
- 如果 corrector 频繁介入，但 reward 被压住、`accept_rate` 很低、`recovery_mode_rate` 很高，说明它可能过于保守

### 7.2 自适应步长是否有必要

重点看：

- `Compare/FPPO/base_step_size`
- `Compare/FPPO/effective_step_size`
- `Compare/FPPO/effective_step_ratio`
- `Compare/FPPO/accept_rate`

判读方法：

- 如果 `step_size` 很快收敛到一个稳定值，而且关掉自适应后几乎不影响 reward / cost，那么这部分可以简化
- 如果不同训练阶段对 `step_size` 的需求明显不同，并且自适应能减少拒绝率或 violation，那么应保留

### 7.3 约束 curriculum 是否真的触发过

重点看：

- `Compare/FPPO/curriculum_progress`
- `Compare/FPPO/curriculum_tighten_count`
- `Compare/FPPO/curriculum_gate_max_ratio`
- `Compare/FPPO/curriculum_gate_min_margin`
- `Compare/FPPO/curriculum_reward_ema`

判读方法：

- 如果 `curriculum_tighten_count` 长期为 `0`，那这套逻辑大概率只是“挂在那里没有起作用”
- 如果每次 tighten 后 reward 不掉、cost margin 更稳定，说明它在帮你做“先学会、再收紧”
- 如果 tighten 后训练明显反复，说明 curriculum 可能太激进

### 7.4 Predictor 的 KL 保护是不是多余

重点看：

- `Compare/FPPO/predictor_kl`
- `Compare/FPPO/predictor_stop_rate`
- `Compare/FPPO/predictor_lr`
- `Loss/learning_rate`

判读方法：

- 如果 `predictor_stop_rate` 基本总是 `0`，且 `predictor_kl` 一直远低于 hard limit，这个 hard stop 很可能是冗余保护
- 如果 `predictor_lr` 剧烈抖动，但 reward 没明显收益，KL 自适应也可能过度设计
- 如果一旦放宽 KL 保护，reward 提升更快但 violation 明显恶化，那它仍然有价值

### 7.5 Reward shaping 项有没有“真贡献”

重点看：

- `Episode_Reward/<term>`
- `Train/mean_reward`
- `Compare/Tracking/*`

判读方法：

- 某个 reward 项长期量级极小，或者几乎是一条平线，优先怀疑它是冗余项
- 某个 reward 项自己涨得很好，但总 reward、tracking、稳定性都没改善，说明它更像“自嗨代理目标”
- 特别建议先筛 Teacher 的 `joint_torques_l2`、`joint_acc_l2`、`hip_pos_l2`、`action_rate_l2`，以及 Student 的同类小权重项

### 7.6 约束项到底是谁在卡训练

重点看：

- `Episode_Cost/prob_joint_pos`
- `Episode_Cost/prob_joint_vel`
- `Episode_Cost/prob_joint_torque`
- `ConstraintDiag/raw_joint_pos_violation_frac`
- `ConstraintDiag/raw_joint_torque_ratio_mean`
- `ConstraintDiag/raw_joint_torque_ratio_max`

判读方法：

- 如果某一项长期远低于 limit，几乎没有成为 active constraint，它大概率不是当前阶段的关键约束
- 如果某一项几乎总是主导 violation，就优先围绕它做 reward / action / policy 结构上的消融

---

## 8. 哪些 trick 不能只靠现有 WandB，需要补日志或补实验

这些模块只看当前训练曲线，很容易误判。

### 8.1 Student 的动作延迟 / 动作历史

为什么现有曲线不够：

- 它更多影响的是控制平滑性、延迟鲁棒性、真实部署一致性
- 训练 reward 可能几乎不变，但落地行为差很多

建议新增曲线：

- `Policy/action_l2`
- `Policy/action_delta_l2`
- `Policy/action_delta_abs_mean`
- `Metrics/foot_slip`
- `Metrics/recovery_time_after_push`

### 8.2 Student 的 constraint adapter

为什么现有曲线不够：

- 你现在只能看到“聚合后的 cost”，但看不到 adapter 到底把哪些项放大/缩小了

建议新增曲线：

- `Cost/raw_term/<name>`
- `Cost/normalized_term/<name>`
- `Cost/adapter_scale/<name>`
- `Cost/adapter_agg_weight/<name>`

如果补完后发现所有 term 的 scale 很快都稳定到接近常数，而且关掉 adapter 性能不掉，就可以考虑删掉。

### 8.3 多 cost-head critic

为什么现有曲线不够：

- reward / total cost 可能相近，但单头 critic 也许已经能学会
- 需要看“每个约束的 value 拟合”是不是因为多头才更稳定

建议新增曲线：

- `Loss/cost_value_<name>`
- `Value/cost_explained_variance_<name>`

### 8.4 命令 curriculum

为什么现有曲线不够：

- reward 上升可能只是因为 curriculum 还没放开，模型并没有真正学到更强能力

建议新增曲线：

- `Curriculum/lin_x_level`
- `Curriculum/ang_z_level`
- `Curriculum/terrain_level_mean`
- `Curriculum/active_command_ratio`

如果 reward 很好，但 curriculum level 一直没推上去，这就是典型“训练看起来不错，实际能力没长上去”。

### 8.5 域随机化 / push 扰动

为什么现有曲线不够：

- 这类 trick 的价值通常体现在 OOD 鲁棒性，而不是同分布训练 reward

建议补评测，不要只看训练：

- 固定 friction / mass / COM 偏移评测
- 统一 push 强度的恢复评测
- 更难 terrain 区间的独立评测
- fall rate / timeout rate / tracking degradation

---

## 9. 如何系统地找出“没什么作用的冗余操作”

可以把冗余分成 4 类：

### 9.1 Dormant 冗余：模块根本没触发

典型特征：

- `predictor_stop_rate` 长期接近 `0`
- `curriculum_tighten_count` 长期不变
- `active_constraints` 长期接近 `0`

这类最容易删，因为它本来就几乎没有参与训练。

### 9.2 Proxy-only 冗余：只优化了自己的代理指标

典型特征：

- `Loss/reconstruction` 下降很漂亮，但 reward / tracking / cost 没改善
- 某 reward term 本身很好看，但最终行为没变

这类要特别小心，因为“loss 变小”不等于“算法更好”。

### 9.3 Overlap 冗余：两个 trick 在做同一件事

当前最值得怀疑的重叠对：

- `symmetry augmentation` 和 `trot/gait` 类 reward
- `reconstruction loss` 和 `history encoder`
- `constraint curriculum` 和 `predictor/corrector` 的保守更新
- `action delay/history` 和动作平滑类 penalty

对这类不要只做单独 ablation，建议补少量二阶组合：

- 去掉 A
- 去掉 B
- 同时去掉 A+B

如果单独去掉都没事，但同时去掉明显退化，说明它们是“互相替代”而不是“都没用”。

### 9.4 Eval-only 非冗余：训练曲线看不出来，但评测有价值

典型对象：

- 域随机化
- push 扰动
- 动作延迟

这类即使训练 reward 没提升，也不能直接删，必须看部署目标是不是需要鲁棒性。

---

## 10. 推荐的实际执行顺序

为了避免实验爆炸，建议按下面这个顺序来：

### 10.1 第 1 轮：只看已有 WandB

目标：

- 找出“基本没触发”的 trick
- 找出“自己的 proxy 很好看，但主指标没收益”的 trick

### 10.2 第 2 轮：做短程筛查

建议：

- 每个候选消融只跑 `3000 ~ 5000` iterations
- 只跑 `1` 个 seed
- 先筛 Teacher，再筛 Student

目标：

- 只保留那些一删就明显变差，或者一删反而更干净的模块

### 10.3 第 3 轮：做正式消融

建议：

- 对通过短程筛查的 4 到 6 个模块，做 `3 seeds`
- 使用正式训练长度
- 保留统一 task / num_envs / max_iterations / eval protocol

### 10.4 第 4 轮：补鲁棒性评测

这一步主要给下面几类模块：

- 域随机化
- push 扰动
- 动作延迟 / history
- student 相关辅助损失

如果某个 trick 对训练曲线没帮助，但对鲁棒性评测明显有帮助，它就不是冗余，而是“收益维度不同”。
