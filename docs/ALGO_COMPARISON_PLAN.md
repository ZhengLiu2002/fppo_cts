# Galileo CTS 实验计划

这份文档描述当前唯一推荐工作流：在统一 CTS 任务上比较 constrained RL 算法，用于 Galileo 四足机器人复杂地形全向盲走训练。

## 1. 实验对象

- 统一任务：`Isaac-Galileo-CTS-v0`
- 统一评测：`Isaac-Galileo-CTS-Eval-v0`
- 统一回放 / 导出：`Isaac-Galileo-CTS-Play-v0`
- 统一 actor：blind proprio + proprio history
- 统一 critic：privileged asymmetric critic

算法比较轴：

- `ppo`
- `fppo`
- `np3o`
- `ppo_lagrange`
- `cpo`
- `pcpo`
- `focops`

## 2. 对比前必须固定

- 相同 task
- 相同 `num_envs`
- 相同 `max_iterations`
- 相同 logger / project
- 相同 checkpoint 选择规则
- 相同 eval / play / export 链路

训练后必须核查：

- `params/env.yaml`
- `params/agent.yaml`
- `params/experiment.json`

## 3. 推荐执行顺序

1. 跑一次 `galileo/fppo_smoke` 验证链路
2. 跑 `galileo/studies/algo_compare_cts_fair` 做短程算法筛选
3. 为候选算法补 `galileo/benchmark/cts_main` 的长程训练
4. 用统一 checkpoint 规则补 3 到 5 个 seed
5. 对最佳设置做 GUI 回放、键盘遥控和导出检查

## 4. 主要指标

训练主指标：

- `Train/mean_reward`
- `Train/mean_episode_length`
- `Compare/Tracking/error_vel_xy`
- `Compare/Tracking/error_vel_yaw`

约束指标：

- `Cost/mean_cost_return`
- `Cost/cost_limit_margin`
- `Cost/cost_violation_rate`
- `Cost/current_max_violation`
- `ConstraintDiag/raw_joint_pos_violation_frac`
- `ConstraintDiag/raw_joint_torque_ratio_mean`

FPPO 诊断指标：

- `Compare/FPPO/accept_rate`
- `Compare/FPPO/effective_step_size`
- `Compare/FPPO/kl`
- `Compare/FPPO/current_max_violation`

## 5. 不能只看训练曲线的内容

- action delay / history
- domain randomization
- push disturbance
- 导出接口稳定性
- 真实遥控手感

这些需要额外配套：

- GUI 回放
- 键盘遥控
- 导出后接口检查

## 6. 现在不再维护的逻辑

仓库已经收敛为 CTS-only 主线，不再维护任何双线实验树或并行入口。
