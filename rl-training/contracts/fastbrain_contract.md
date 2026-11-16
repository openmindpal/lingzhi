# 快脑槽契约(fastbrain_contract)

> 唯一权威 = 灵智系统引擎契约 §1.5(随合作交付)。训练侧镜像。

## 运行形态(引擎机制——训练侧只需对齐 IO 与约束)

- 快脑 = 以任务节拍(rate_hz)连续决策的专家代理;引擎只承载机制(托管运行时/流式目标信箱/
  微码路由/安全围栏),快脑本体 = 声明行 + 权重(引擎内零快脑逻辑)。
- 肌肉与快脑并驱:快脑消费肌肉同源观测(逐位同源,含 goal 块),产出 goal 更新写入流式信箱,
  肌肉(reach_goal,goal_conditioned)逐拍执行。

## 权重 IO

| 项 | 规格 |
|---|---|
| 输入 | `[1, 101]` = 肌肉同源观测展平(97) + 当前 goal(4) |
| 输出 | `[1, 4]` = goal 更新 [goal_x, goal_y, goal_z, goal_f] |

## 声明行(包装载校验)

```toml
[[fastbrains.rows]]
name    = "tactician"
kind    = "onnx"                          # ∈ {onnx}
weights = "fastbrains/tactician.onnx"
outputs = { goal_stream = "reach_goal" }  # 交叉校验:须对应存在且 goal_conditioned 的策略行
rate_hz = 50.0                            # > 0
budget_ms = 60000                         # 回合预算硬帽——超时诚实收场
```

## 训练约束

| 项 | 约束 |
|---|---|
| 输出有限 | goal 任一分量非有限 = 派发 Err——导出端加 clamp 头保有限 |
| 输出域内 | 作业域包络内(基座系米 / f∈[0,1])——训练输出层域裁剪 |
| 防抖 | goal 变化率限制(训练奖励/输出差分惩罚) |
| 失效安全 | estop/联锁凌驾快脑;策略不得假设"快脑永在线"——肌肉策略须独立可用 |

## 训练方式(本平台参考实现)

① 监督蒸馏:解析专家(toward moving target, 限速/限域)生成 (obs+goal)→goal' 轨迹对,行为克隆;
② RL 外环(预留):最终任务奖励回传快脑;
③ 规则混合(预留)。
