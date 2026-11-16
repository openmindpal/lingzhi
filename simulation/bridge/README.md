



# 灵智系统仿真桥接器——契约（bridge/）

> 被验系统 = 灵智系统（引擎为灵智系统专有组件，不在本平台开源范围——保密声明见 RELEASE.md）。

| 项 | 契约 |
|---|---|
| 确定性 | 固定种子 → 同输入序列同仿真轨迹（回归可复现） |
| 时间加速 | headless 批量：仿真时间 > 实时（长时任务分钟级验证） |

## 文件分工

| 文件 | 职责 |
|---|---|
| mj_model.py | MJCF 装载/mj_step/关节层 PD 伺服/传感快照/外扰通道 |
| camera_stream.py | 渲染线程 @5Hz 产帧 → /frame.jpg（GL 单线程生命周期） |
| faults.py | 故障注入器：断流/冻结帧/急停/力超限——由 suites/fault_injection/*.yaml 驱动 |
| time_control.py | 时间加速 + 固定种子（确定性回归） |
| smoke_local.py | 本地冒烟 S1~S4（模型/包一致、闭环、快照契约、424B/664B 线打包） |
| smoke_biped.py | 双足冒烟 B1~B4（平衡 ORT 闭环扰动回归 + reach 语义精确/伺服有界） |

## 模型 XML 纪律（三条血泪根修——新模型必读）

1. **hinge `range` 单位 = 度**（MuJoCo XML 约定，编译转弧度）——直接写弧度数值
   会把关节锁死在百分之一弧度（±0.6 → ±0.0105 rad）。包 profile 用弧度，逐项换算。
2. **自碰撞位掩码**：机器人 geom `contype=1 conaffinity=0`（机器人↔机器人永不相碰）；
   地面/桌/物体 `contype=1 conaffinity=1`（与机器人相碰）。否则链式相邻体的
   质量段互相穿插 → 数十个接触点啃掉伺服扭矩。
3. **每个带关节的 body 必须有质量**（几何或 inertial）——零质量关节体编译报错。
