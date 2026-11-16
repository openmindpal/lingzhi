# 观测布局契约(obs_layout)

> 唯一权威 = 灵智系统引擎契约 §2(随合作交付)。本文件 = 训练侧镜像,与实现不一致处以引擎为准。

## 布局(按序拼接,块间无间隔)

```
[ q(J) | dq(J)? | tau(J)? | imu(10)? | contact(6k)? | goal(4)? ]
```

| 块 | 维度 | 单位/语义 |
|---|---|---|
| q | J=29 | 关节位置 rad(按 joint_mapping.ordered 名投影) |
| dq | J | 关节速度 rad/s |
| tau | J | 关节力矩 N·m |
| imu | 10 | [quat_wxyz(4), gyro_xyz(3), accel_xyz(3)] |
| contact | 6k | k 路六维力/矩 [fx,fy,fz,tx,ty,tz](G1 双腕 k=2 → 12) |
| goal | 4 | [goal_x, goal_y, goal_z, goal_f](基座系 m / 归一 0~1) |

## 宽度合法集(装载期校验)

- `b = 1 + motor_fields 中 :dq/:tau 字段数`(G1 未声明 → b=3)
- 合法 obs_dim ∈ { b·J + 10 + 6k }(+4 goal)——G1: **97**(基础)/ **101**(+goal)/ **109**(+contact k=2)
- 运行期每拍 `obs.len() == obs_dim`,不符 = fail-fast(引擎);训练侧同判据(§3 契约逐位一致)

## 训练侧硬规则

1. **引擎不归一化**——单位 rad / rad·s⁻¹ / N·m / SI;归一化必须烘焙进 ONNX 计算图(export_onnx 承担)。
2. **禁特权信息(Sim2Real 第一杀手)**——观测装配器按本布局白名单实现:物体真值位姿、
   仿真上帝视角状态、特权标志一律不得进观测;真机无此输入 = 权重上机即瞎。
3. obs_contact 行训练必须含接触力噪声/偏置随机化(§10 引擎训练硬约束)。
4. goal 块域:xyz 基座系米(作业域包络),f 归一 0~1——课程采样须覆盖全包络。

### goal 块语义扩展——运动模式演化提案(正式化)

标准语义 = [目标x, 目标y, 目标z, 力/强度标量](基座系米/归一 0~1)。运动模式任务
在**宽度不变(恒 4 维)**前提下按行声明扩展语义;引擎训练指南 §2.2 对齐确认前,
dodge/intercept 权重不得上真机:

| 行任务 | goal 语义 |
|---|---|
| 标准(reach/reach_contact/grasp/bimanual/intercept) | [截击/目标x, y, z, 力标量] |
| dodge(闪避) | [避让方向单位x, 避让方向单位y, 威胁距离m, 紧急度0~1]——威胁向量逐拍注入 |
| inhand(手内) | goal[0]=目标旋转弧度(其余维未用) |

规则: 语义扩展不改宽度、不改其余块;训练侧实现 = rltrain/env_mj.py(_goal_obs),
引擎侧对齐确认后本提案转正式契约。
