# 灵智系统 RL 训练平台技术架构（通用型——多厂商本体接入）

> 平台定位：把"训练产出"变成"过验收门的本体包权重"——训练环境按引擎契约构建，产物 ONNX
> 逐位对齐观测/动作规格，经仿真平台验收门全绿后交付本体包。
>
> 落地位置：平台 = `rl-training/`（仓库根，与 `simulation/`、`body-sdk/` 同级）——本文件位于平台目录内。
>
> **训练产物用途 = 灵智系统的四族策略与快脑**——装入本体包，由灵智认知层按 词+参数 派发执行。
>
> 契约唯一权威 = 灵智系统引擎契约（随合作交付，不在本仓库）；
> 本体包契约样例 = unitree-g1（G1 29 DOF，随对接包提供）；
> 验收门 = `simulation/suites/`（goal_sweep / longevity / faults / sim2sim / baselines / tri 在环）。
>
> 通用性声明：平台机制与"四族"无关——四族只是预置实例。任何 obs→action 闭环策略
> （任意本体、任意行数、循环网络、自定义任务）在同一平台按同一流程训练与验收。

---

## 1. 策略形态论证——为什么具身智能必须是这种策略

本平台训练的不是"控制器"，也不是"轨迹"，是**认知系统可调用的技能词**：goal 条件化 + 契约绑定 +
绝对关节目标。这个形态不是偏好，是灵智系统分工（泛化=认知层组合，丝滑=训练权重，连贯=分层纪律）
推导出的唯一解——本节给立论，其后各层（L0~L6）都是这个立论的工程兑现。

### 1.1 具身智能对策略的三条本质要求（为什么必须是这样）

| 要求 | 含义 | 对策略形态的强制 |
|---|---|---|
| **可组合** | 任务无限由认知组合承接——云端认知层按 词+参数 无限派发，不靠预编排 | 策略必须暴露稳定函数签名（obs 布局/动作语义/goal 参数）——黑盒/引擎锁死策略进不了词表；cmd_vel 类遥控接口使认知泛化与动作执行断裂 |
| **状态连续** | 动作衔接不回中间姿势——动量/姿态被下一个动作直接继承，过渡是动力学涌现 | 策略必须从任意当前状态接管：观测含 q/dq，训练覆盖随机化初始状态；不需要插值混合器 |
| **物理像人** | 人类动作是同组力学成本下的近优解（最小 jerk/能耗/姿态经济） | 奖励编码人类同款成本（jerk 惩罚/能耗/姿态先验必选）——动作风格被物理逼出而非贴皮模仿；风格先验从人类示范克隆（模仿预训练） |

三条合起来排除的正是行业惯例三件套（单轨迹表演 / 遥控终端 / 特权观测）——README《厂商对接指南》
"三个差异"表是它的操作口径，此处是立论出处。

### 1.2 泛化从哪来——参数空间，不是轨迹数量

| 机制 | 泛化单位 | 结果 |
|---|---|---|
| goal 条件化 + 全域课程采样 | 目标空间一次学完 | 换杯子位置 ≠ 新任务——认知层改 goal 参数即可，策略平滑跟踪任意目标点（tri_02 解析预测判据） |
| 四族原语 × 词表组合 | 组合空间 | 单技能有限、组合无限：任务无限由认知组合承接，新原语仅在运动模式级增量（技能库收敛口径） |
| 风格/强度标量条件化 | 连续自由度 | 同一语义词在不同标量下幅度/节奏连续变化——不是档位枚举（族4 goal 第 4 位） |

划界（诚实）：泛化的上限 = 词表覆盖度——词表外的新运动模式需要训练；"想"的泛化在认知层、
不在策略——策略只负责"怎么做"的力学，两层各守各的时钟（认知低频派发，策略高频闭环，互不越界）。

### 1.3 为什么表演/遥控式策略到不了具身智能（反面论证）

| 惯例形态 | 结构性缺陷 |
|---|---|
| 单轨迹表演策略 | 只会复现训练轨迹——无 goal 空间，环境变一点就废；演示视频是它的全部能力 |
| 遥控终端接口（cmd_vel 类） | 策略是被遥控的执行器，认知层无法声明式调用——上层智能与下层执行接口断裂 |
| 特权观测训练 | 仿真真值/上帝视角进观测，真机无此输入——权重上机即瞎（Sim2Real 第一杀手） |
| 自定指标自验 | 无公开判据可复现——表演文化给"看起来对"，具身交付要机械断言 |

结论：**具身智能的策略 = 被认知系统调用的契约化技能词**。L0 契约层逐条锚定引擎、L3 导出
三重校验、L5 验收机械断言、Sim2Sim 对拍——全部是这句话的工程兑现。

## 2. 分层架构总图

```
L0 契约层   contracts/——obs 布局/动作语义/ONNX IO/快脑槽（引用引擎文档，只引用不复制）
    ↓ 契约驱动
L1 环境层   envs/——Isaac Lab GPU 并行环境（G1 = 29 DOF，与本体包 profile 同源）
    ↓
L2 训练层   tasks/ + algos/——四族任务定义 × PPO/SAC/循环网络/模仿预训练
    ↓ checkpoint
L3 导出层   scripts/export_onnx.py——归一化烘焙 + 布局校验 + 软限位余量检查
    ↓ *.onnx
L4 打包层   scripts/package_build.py——policies/ + fastbrains/ + [policies.rows] 行声明 + 签名
    ↓ body-package
L5 验收层   scripts/verify_in_sim.py——仿真平台验收门（同一扇门，机械断言）
    ↓ 全绿
L6 部署层   实机标定回填 → 真机（安全链在环 + 保守增益）→ 实机验收
```

每层产物可独立替换：换本体 = 换 envs；换任务 = 换 tasks；换算法 = 换 algos——验收门不变。

## 3. 文件夹结构（权威）

平台落地后与本目录其他 app 同级：

```
<仓库根>/
├── simulation/                                ← 仿真验证平台（验证面）
├── body-sdk/                                  ← 本体包对接套件（交付面：规范/签名工具/校验器/样例清单）
├── rl-training/                               ← RL 训练平台（本方案，本文件所在）
└── apps/                                      ← 灵智系统内部（云端引擎/端侧引擎/本体包/伴生服务——不开源面）
```

`rl-training/` 内部结构（与实现逐文件对齐）：

```
rl-training/
├── README.md                                  ← 入口：快速开始 / 厂商对接指南 / 后端口径 / 诚实边界
├── 灵智系统RL训练平台技术架构.md               ← 本文件（平台总架构方案）
├── contracts/                                 ← L0 契约（与引擎文档同源，只引用不复制）
│   ├── obs_layout.md                          ← 观测布局逐维契约：[q(J)|dq(J)|tau(J)|imu(10)|contact(6k)|goal(4)]
│   │                                            宽度合法集 = {b·J + 10 + 6k}(+4 goal)——G1 J=29, b=3: 97/101/109
│   ├── action_contract.md                     ← 动作语义：J 维绝对关节目标(rad)，软限位钳制(margin 0.05)
│   ├── onnx_io.md                             ← ONNX 规格：单输入[1,obs_dim]/首输出[1,J]/opset 17/禁特权信息
│   └── fastbrain_contract.md                  ← 快脑槽：输入=obs(97)+当前goal(4)，输出=goal(4)，goal_stream 交叉校验
├── rltrain/                                   ← 核心库（契约驱动的参考实现）
│   ├── contract.py                            ← 宽度合法集/软限位/中立位校验（引擎装载校验镜像）
│   ├── g1_cfg.py                              ← G1 配置：解析 unitree-g1 包 profile（单一来源）+ 自由底座变体生成
│   ├── obs_assembler.py                       ← 观测装配器（白名单制——特权信息进不了观测）
│   ├── tasks.py                               ← 任务 yaml 装载/校验/行声明生成
│   ├── env_mj.py                              ← MuJoCo 向量环境（自由底座 + PD 伺服 + 四族/抓取奖励/终止）
│   ├── mjcf_gen.py                            ← 参数化本体生成器（自由底座/质量标定/灵巧手/物体/按压面）
│   ├── isaaclab_env.py                        ← Isaac Lab GPU 适配器（契约不变；首跑按版本校准）
│   ├── sac.py                                 ← SAC（力控类备选——与 PPO 同导出路径）
│   ├── urdf_gen.py                            ← URDF 生成器（Sim2Sim 策略级资产，从编译模型导出）
│   ├── demos.py                               ← 演示数据录制/加载（il_pretrain 数据面）
│   └── ppo.py                                 ← PPO(torch) + 观测归一化（导出时烘焙进计算图）
├── envs/                                      ← L1 训练环境（多厂商本体接入点）
│   ├── g1/                                    ← 宇树 G1（29 DOF：腿6×2+腰3+臂7×2；手包 45）——首个完整接入实例
│   │   ├── g1_cfg.py                          ← 入口（实现唯一来源 = rltrain/g1_cfg.py，防双份漂移）
│   │   ├── body_spec.yaml                     ← 本体规格（几何/质量/五指灵巧手/物体[样例]——宇树规格回填）
│   │   ├── scene.yaml                         ← 场景：平地/±3cm 不平地面/扰动源（100N 推力）
│   │   ├── randomize.yaml                     ← 引用（不复制）simulation/suites/sweeps/domain_randomization.yaml
│   │   └── assets/                            ← 生成产物（g1_train_free.xml / gen/*.xml / g1_pb.urdf）
│   ├── agibot/README.md                       ← 智元机器人——接入点（规格[待回填]）
│   ├── tiangong/README.md                     ← 天工机器人（北京人形机器人创新中心）——接入点
│   ├── galbot/README.md                       ← 银河通用（Galbot）——接入点
│   ├── engineai/README.md                     ← 众擎机器人（EngineAI）——接入点
│   ├── booster/README.md                      ← 加速进化（Booster Robotics）——接入点
│   ├── xsquare/README.md                      ← 自变量机器人（X Square）——接入点
│   ├── ubtech/README.md                       ← 优必选（UBTech，Walker S）——接入点
│   ├── fourier/README.md                      ← 傅利叶智能（Fourier，GR 系列）——接入点
│   ├── kepler/README.md                       ← 开普勒（Kepler，先行者 K2）——接入点
│   ├── robotera/README.md                     ← 星动纪元（Robot Era，STAR 系列）——接入点
│   ├── songyan/README.md                      ← 松延动力（Songyan Dynamics，小诺系列）——接入点
│   ├── tesla/README.md                        ← 特斯拉（Tesla，Optimus）——接入点
│   ├── figure/README.md                       ← Figure AI（美国，Figure 02/03）——接入点
│   ├── generic/README.md                      ← 通用本体模板（三步接入，流水线/验收门不变）
│   └── sensors/README.md                      ← 装配器说明（实现唯一来源 = rltrain/obs_assembler.py）
├── tasks/                                     ← L2 训练任务定义
│   ├── family_balance.yaml                    ← 族1 动态平衡（obs 97，纯本体感觉，多实例=步态风格）
│   ├── family_reach.yaml                      ← 族2 操作·目标条件化（obs 101，goal 课程采样）
│   ├── family_reach_contact.yaml              ← 族2b 操作·力觉（obs 109，contact 噪声随机化必选）
│   ├── family_deform.yaml                     ← 族3 形变体（obs 97，形态库：蹲/跪/弯/仰/转/趴）
│   ├── family_expression.yaml                 ← 族4 表达（obs 97；可选 goal 标量=风格强度）
│   ├── family_grasp.yaml                      ← 族2c 抓取（手包 J=41，obs 149——hands+objects+goal_contact）
│   ├── family_athletic.yaml                   ← 高动态全身（翻跟头/跳跃——airborne 语义）
│   ├── family_dodge.yaml                      ← 反应闪避（威胁向量经 goal 块——契约演化提案）
│   ├── family_recover.yaml                    ← 摔倒恢复（倒地初始位 self-right）
│   ├── family_soft.yaml                       ← 柔性接触（flex 布+恒力擦拭）
│   ├── family_pour.yaml                       ← 倒水（杯焊手+粒子水，洒出计数）
│   ├── family_needle.yaml                     ← 超高精度（穿针 3mm 缝+插入深度）
│   ├── family_bimanual.yaml                   ← 双手协同（双臂力耦合——goal 条件化课程，无限双手场景）
│   ├── family_inhand.yaml                     ← 手内精细（手包 J=45，指尖级转角课程）
│   ├── family_intercept.yaml                  ← 拦截同步（下落球时间同步到达——接物/拳击躲避扩展）
│   ├── family_strike.yaml                     ← 冲击击打（立柱球冲量——击打力学原语）
│   ├── fastbrain_tactician.yaml               ← 快脑（输入 101 → 输出 goal 4，rate 50Hz）
│   └── custom/template.yaml                   ← 第三方任意策略任务（通用性扩展点：reward/终止/课程模板）
├── algos/                                     ← L2 算法配置
│   ├── ppo.yaml                               ← PPO（参考后端已实现；4096 env = isaaclab 规模）
│   ├── sac.yaml                               ← SAC（力控类备选——已实现，train.py --algo sac）
│   ├── recurrent.yaml                         ← 循环网络（研究级——引擎单输入契约，导出仅放行 MLP）
│   └── il_pretrain.yaml                       ← 模仿学习预训练（Diffusion Policy/ACT——适配点）
├── scripts/
│   ├── train.py                               ← 入口：train.py --task family_reach（mujoco 后端默认；isaaclab 诚实报错）
│   ├── eval_gate.py                           ← 验收门（训练侧）：ONNX 确定性回放 + 任务判据（阈值 = 任务 yaml gate 段）
│   ├── export_onnx.py                         ← 导出：归一化烘焙 + 单输入单输出校验 + 软限位钳制头（循环网络拒收）
│   ├── sim2sim_check.py                       ← Isaac↔MuJoCo 对拍（<5°；pybullet 缺 = 诚实报错 exit 2）
│   ├── package_build.py                       ← 产出本体包：权重落位 + 行声明 + 自检 + 签名（--run 可重复=整包交付）
│   ├── record_demo.py                         ← 演示数据录制 CLI（il_pretrain 数据面）
│   └── verify_in_sim.py                       ← 装载门（V1~V5 本地门 / --docker 集成门=仿真平台全套件）
├── tests/                                     ← 平台自验证套件（一条命令，机械断言，零 GPU）
│   ├── run_all.py                             ← 总入口（--fast 快速档）
│   └── t_*.py                                 ← 契约/装配器/生成器/六环境/学习冒烟/导出打包往返/负例注入
├── runs/                                      ← 训练产出（gitignore，不入库）
│   └── <run_id>/                              ← ckpt.pt / ckpt_best.pt / manifest.json / metrics.jsonl / exports/ / package/ / tb/
└── docs/training_log_template.md              ← 训练日志纪律（未过验收门的权重不得进交付段）
```

## 4. L0 契约层——训练的"宪法"（逐条锚定引擎）

| 契约 | 内容 | 对训练的约束 |
|---|---|---|
| 观测布局 | `[q(29)|dq(29)|tau(29)|imu(10)|contact(6k)|goal(4)]`，块序拼接无间隔 | 环境观测装配器**逐位**产出；G1 宽度：97（基础）/101（+goal）/109（+contact k=2） |
| 引擎不归一化 | 单位 rad / rad·s⁻¹ / N·m / SI | 归一化**必须烘焙进 ONNX 计算图** |
| 动作语义 | J=29 维**绝对**关节目标(rad)，每拍整向量下发 | 训练动作空间一致（或导出头内转换）；速度型行为由策略自身积分实现 |
| 软限位钳制 | 出线钳入 `[min+0.05, max−0.05]` | 限位外动作无训练意义；标定中立位须在软限位内（装载校验） |
| ONNX IO | 恰 1 输入 `[1,obs_dim]` f32；首输出 `[1,J]` f32；禁止多输入/动态形状 | 导出校验项 |
| **禁特权信息** | 观测只允许现实世界同款传感器来源（编码器/IMU/力矩/腕部六维） | 物体真值位姿/上帝视角状态**不得**进观测——装配器按布局白名单实现 |
| 平滑性 | 出线逐拍直驱电机，无滤波 | 加速度/jerk 惩罚**必选**（tri_03：20ms 采样 max_step<0.05rad） |
| 时序 | tick=1/frequency_hz；回合帽 max_duration_ms（缺省 30000） | 推理延迟须留余量（引擎侧无超时） |

## 5. 四族训练任务规格（G1 预置实例）

### 4.1 族1 动态平衡（family=balance，obs_dim=97，goal_conditioned=false）

| 项 | 规格 |
|---|---|
| 观测 | 纯本体感觉（q/dq/tau/imu），无 goal 块 |
| 接口形态 | **行走速度/步态风格 = 多实例行**（每风格一行：自然/快走/慢踱 ≥3 档）——认知按行名派发；cmd_vel 类接口已废除，训练按实例拆分 |
| 目标能力 | 行走 0~1.5m/s；抗扰 100N·0.5s 不摔；±3cm 不平地面不摔；双臂 2kg 负载挥动配平 |
| reward | 姿态保持（躯干姿态/重心高度）+ 存活 + 速度域覆盖 + 能耗 + jerk 惩罚 |
| 随机化重点 | 推力扰动（幅值/方向/时机）、地面不平、质量/质心 ±20% |
| 步数参考 | 1~5B 步；验收 = tri_01（扰动注入回归目标位形） |

### 4.2 族2 操作·目标条件化（family=manipulation，obs_dim=101，goal_conditioned=true）

| 项 | 规格 |
|---|---|
| 观测 | 97 + goal(4)——`[goal_x, goal_y, goal_z, goal_f]`（基座系 m / 归一 0~1） |
| **goal 条件化训练** | 课程采样：xyz ∈ 作业域包络内均匀撒点，f ∈ [0,1] 全域覆盖——策略学会跟随**任意**目标（认知层派发参数直接注入） |
| 目标能力 | 重复定位 ≤5mm；任务空间精度由像素伺服承载（IBVS·分析式，与标定漂移解耦） |
| reward | 末端到位误差 + 力约束（goal_f 语义）+ jerk 惩罚 |
| 步数参考 | 1~10B 步；可选模仿学习预训练（Diffusion Policy/ACT）；验收 = tri_02/06 |

### 4.3 族2b 操作·力觉（family=manipulation，obs_dim=109，obs_contact=true）

| 项 | 规格 |
|---|---|
| 观测 | 97 + contact(12, k=2：0=左腕 1=右腕，各六维力/矩)——路序包文档化 |
| 目标能力 | 插入/按压类以力特征为到位判据；接触响应 ≤50ms；双手力耦合任务左右腕力同观测 |
| reward | 力轨迹跟踪 + 位姿到位 + 滑移惩罚 |
| **随机化必选** | 接触力噪声/偏置随机化（obs_contact 行硬约束）；正常作业力 << `safety.contact_force_limit_n`（防误触发联锁） |
| 硬件前提 | 腕部六维力传感必须——无传感器 = 该行不可声明（诚实缺帧语义） |
| 验收 | tri_06 接触（25N 注入 → 输出偏移可解析预测）+ tri_07（200N → 立即收场 + estop） |

### 4.4 族3 形变体（family=deform，obs_dim=97）

| 项 | 规格 |
|---|---|
| 形态库 | 蹲(0~0.4m)/跪/弯腰(0~90°)/后仰(0~30°)/转身(±180°)/趴——按实例行拆分（deform_ramp 等） |
| 输出形态 | 参考实现 `act = q + Δ`（权重在 q 块取单位阵+偏置 Δ，经反馈逐拍积分）——速度型行为的契约内做法 |
| 目标能力 | 转移时间物理合理（蹲 0.8~1.5s/转身 1~2s），全程平衡，保持 30s 无漂移，任意形态→站立 1~2s |
| reward | 形态到达 + 平衡 + 重心轨迹平滑 + jerk（tri_03 断言） |
| 步数参考 | 0.5~3B 步 |

### 4.5 族4 表达（family=expression，obs_dim=97；可选 goal 标量条件化）

| 项 | 规格 |
|---|---|
| 词汇 | 开放——本体包按声明格式自定义表达原语名（greet_wave 等），引擎不设词表 |
| 风格条件化 | 可选 goal_conditioned 行：goal 第 4 位 = 风格/强度标量（0~1），同一语义词不同标量下幅度/节奏不同——训练分布覆盖 |
| reward | 自然度（平滑/节律）+ 软限位安全；情感状态等高层信号经认知层编码进标量 |
| 验收 | 平滑断言（tri_03 同判据）+ 软限位钳制 |

### 4.6 快脑（fastbrains：tactician，输入 101 → 输出 goal(4)，rate 50Hz）

| 项 | 规格 |
|---|---|
| 运行形态 | 以任务节拍消费肌肉同源观测（97，与肌肉逐位同源含 goal 块）+ 当前 goal(4)，产出 goal 更新写入流式信箱；肌肉（reach_goal）逐拍执行——肌肉与快脑并驱 |
| 训练方式 | ① 监督（专家轨迹的 goal 决策蒸馏）② RL（外环：最终任务奖励回传快脑）③ 规则混合 |
| 输出约束 | goal 全有限（派发校验：任一非有限 = 派发 Err）+ 域内（作业包络）+ 变化率限制（防抖） |
| 失效安全 | estop/联锁凌驾快脑；回合预算硬帽 budget_ms（缺省 60000）超时诚实收场 |
| 交叉校验 | `outputs.goal_stream = "reach_goal"` 须对应存在且 goal_conditioned 的策略行（包装载期校验） |

## 5b. 扩展族谱注册表——运动模式级增量实例（2026-11-16 补齐）

> 族谱完整性原则：**家族枚举冻结（balance/manipulation/deform/expression 四族——引擎零改动），
> 新力学维度 = 既有家族下的实例行**（每实例一个 yaml——与本表全部条目同构）。
> 下列 10 个实例已过契约门（t_contract），任务定义即可训练状态（权重=产线排期产物）。

### 档2 新力学维度（5 实例——manipulation 家族下的新力学结构）

| 实例 | 力学结构 | 判据 |
|---|---|---|
| family_tool（tool_use） | **工具使用三级力链**——手-工具-物体：抓握稳定+力传递（轻触/常规/冲击由 goal_f 连续参数化） | 工具作用效果+不掉工具 |
| family_heavy（heavy_haul） | **全身力量协同**——腿蹬地→躯干→臂持物联合发力链（负荷 3~15kg 随机） | 负荷位移+姿态稳定+不倾覆 |
| family_fabric（fabric_fold） | **织物精细**——高自由度可变形体抓取点序列（粒子-质簧代理 64 粒） | 形态匹配率≥0.85+不撕裂 |
| family_phys_interact（phys_assist） | **人机物理交互**——变阻抗跟随（人主导度连续）+人侧力上限（联锁柔顺版） | 过力事件=0+交互完成 |
| family_tactile（tactile_blind） | **触觉盲操作**——触觉流搜索-辨识-操作闭环（⚠️依赖触觉阵列数据面——未装不可交付） | 辨识正确率≥0.8+完成 |

### 档1 步态/平衡实例扩展（5 实例——balance 家族课程级）

| 实例 | 覆盖 | 判据 |
|---|---|---|
| family_gait_side（gait_side） | 侧移 0.40 m/s | 速度跟踪+不摔 |
| family_gait_back（gait_back） | 后退 0.30 m/s | 同上 |
| family_gait_run（gait_run） | 奔跑 1.8 m/s（腾空相） | 速度跟踪+不摔 |
| family_gait_carry（gait_carry） | 持物行走 1~3kg（priority=carry） | 速度+不摔+全程持握 |
| family_gait_terrain（gait_terrain） | 不平地形（坡/碎石/台阶课程——mjcf_gen 生成） | 速度+不摔 |

### 族谱全景（登记口径）

四家族枚举 × 实例行全集 = **28 个任务定义**（原 18 + 本表 10）。
再新增力学维度 = 再加实例行（family 枚举不动）——**族谱按需生长、引擎永不变**。
尚属未来扩展（本表外）：多机协同（系统层）、飞行/骑行（新实体形态+对应家族）。

## 6. L3 导出层（export_onnx.py 检查单）

| 检查 | 判据 |
|---|---|
| 归一化烘焙 | 图内完成（引擎不归一化） |
| IO 形状 | 恰 1 输入 `[1,obs_dim]`；首输出 `[1,J]`；opset 固定 |
| 特权信息审计 | 训练观测装配器逐位对照契约白名单——多余模态 = 导出拒绝 |
| 软限位一致性 | 输出经 `[min+margin, max−margin]` 裁剪头（或训练已保证域内） |
| 循环网络状态 | LSTM/GRU hidden state 显式进出图（单步图），或文档化"每回合重置"语义 |
| 对拍 | Sim2Sim：Isaac↔MuJoCo 关节轨迹偏差 <5°、稳定域匹配 |

## 7. L4 打包层（package_build.py）

1. 权重落位 `policies/*.onnx`、`fastbrains/tactician.onnx`
2. 生成 `[[policies.rows]]` 行：family（枚举 balance/manipulation/deform/expression）/ name / weights / frequency_hz / priority / obs_dim（∈ 合法宽度集）/ goal_conditioned / obs_contact
3. 生成 `[[fastbrains.rows]]`：name / kind=onnx / weights / outputs.goal_stream / rate_hz / budget_ms
4. 装载前自检：obs_dim ∈ {b·J+10+6k}(+4)；操作词绑定实例存在；标定中立位 ∈ 软限位（G1 样例：elbow −0.75 = 限位 [−1.5,0] 软域中点）；
   **快脑交付 = 肌肉 + 快脑两条**（`--muscle-run`，goal_stream 交叉校验强制肌肉行在场）
5. 签名（sign_package 三步：占位→签→回填）

## 8. L5 验收层——与仿真平台同一扇门

```bash
python scripts/verify_in_sim.py --package unitree-g1
# 内部顺序调用 simulation/suites/：仿真台冒烟 → goal_sweep → longevity → faults → sim2sim → baselines → tri 在环
```

| 门 | 判据（引用引擎验收协议） |
|---|---|
| 装载 | load_unverified + 派发装载通过 |
| tri_01 平衡 | 扰动注入后回归目标位形 |
| tri_02 操作 | goal 变更 → 稳态偏移符合解析预测 |
| tri_03 形变 | 20ms 采样 max_step<0.05rad |
| tri_05 本体接地 | 观测 = 快照逐通道 |
| tri_06 移动目标/接触 | 跟踪不脱靶；25N 注入可解析预测 |
| tri_07 力联锁 | 200N 注入 → 立即收场 + estop 锁存 |
| Sim2Sim | MuJoCo 对拍 <5° |
| 长跑/故障/金轨迹 | L1~L5 / F1~F4 / hash 逐位 |

## 9. L6 部署层（真机阶梯）

1. 实机标定回填：中立位/限位/sign → 本体包（装载校验：中立位 ∈ 软限位）
2. G1 对接前置：unitree_sdk2 `rt/lowcmd` 为 IDL 结构 + CRC 校验——载荷编码按 LowCmd IDL 实现后方可驱动真机（包文件头边界 1）
3. 真机试跑：安全链在环 + 保守增益 → 按实测微调
4. 实机验收：行走 10m+8 字 / 蹲转蹲 / 抓放——不摔不偏离

## 10. 通用性：其他策略怎么接（任意本体 / 任意任务 / 循环网络）

| 接入什么 | 要做什么 | 不变的是什么 |
|---|---|---|
| 其他本体（非 G1） | `envs/generic/` 填 J/关节序/限位/KP·KD；契约宽度按 {b·J+10+6k}(+4) 重算 | 契约层公式、导出检查单、验收门 |
| 其他任务（四族之外） | `tasks/custom/*.yaml`（reward/终止/课程模板）；族标签取最贴近制式，实例名开放 | 训练→导出→打包→验收流水线 |
| 循环网络策略 | `algos/recurrent.yaml`（LSTM/GRU + 状态显式导出） | 同上 |
| 任意需要训练的策略 | 平台四件解耦件（环境/任务/算法/导出验收）——替换 envs+contracts 即适配新域；验收门按域裁剪 | 分层流程本身 |

引擎侧开放度边界（诚实声明）：family 为引擎枚举制式（四族），开放的是**实例名/行数/词/权重**——"任意策略" = 四族制式内任意实例 + 平台流程本身域无关。

技能库收敛口径（README「技能库收敛口径」节为唯一权威）：新任务优先认知组合（零训练），
新原语仅在运动模式级增量——四族各 5~15 原语即达运动完备集，任务无限由组合承接。

## 11. 训练基础设施与时长参考

| 项 | 规格 |
|---|---|
| GPU | **单张 RTX 4090（24GB）即可跑**——4096 并行环境 × obs 97~109 维 float32 显存占用很小，瓶颈在接触物理仿真吞吐与 RL 更新 |
| 仿真器 | Isaac Lab（GPU 并行）；Sim2Sim 交叉验证 = MuJoCo |
| 算法 | PPO/SAC + 域随机化 + 课程学习；操作族可加模仿预训练 |
| 随机化 | 质量/质心 ±20%，摩擦/阻尼 ±30%，电机延迟 0~20ms，IMU/编码器高斯噪声，obs_contact 行加力觉噪声 |

**单张 4090 训练时长参考**（吞吐假设：行走类 3~5 万步/秒，接触类 1.5~3 万步/秒——29 DOF 人形接触仿真是吞吐瓶颈）：

| 训练任务 | 步数预算 | 墙钟（单卡连续跑） |
|---|---|---|
| 族1 balance | 1~5B | 约 6 小时 ~ 2 天 |
| 族3 deform | 0.5~3B | 约 3 小时 ~ 1 天 |
| 族2 manipulation（reach，101） | 1~10B | 约 1 ~ 4 天 |
| 族2b manipulation（contact，109） | 1~10B | 约 1 ~ 6 天 |
| 快脑 tactician（小网络蒸馏/外环 RL） | — | 数小时 ~ 1 天 |
| 导出 + Sim2Sim 对拍 + 验收门 | — | 数小时 |

合计：**全部四族 + 快脑顺序跑约 1~2 周**；多卡可按族并行（每族一张卡）压缩到数天。
诚实说明：上表是**收敛跑本身的墙钟**；首次从零调通的日历时间由奖励工程/课程迭代主导（以周~月计），收敛判据 = 奖励平台 + 验收门通过，不以固定步数/时长为准。

## 12. 诚实边界

| 边界 | 说明 |
|---|---|
| Sim2Real 残余差距 | 不可归零（全行业如此）——平台把它从"未知风险"压缩为"阶梯上真机的小量微调"，安全链把最坏情况封顶 |
| 训练成本 | GPU 时长见 §11；平台定义契约与流水线，不保证收敛时长 |
| 族枚举固定 | 引擎 family 枚举 = 四族制式；开放度在实例/行/词层（§10） |
| G1 真机前置 | lowcmd IDL 载荷编码未完成（包文件头边界 1）——完成前本平台验证到"包装载 + 仿真验收全绿"为止 |

## 13. 一条命令的完整流水线

```bash
# ① 训练（任一族/快脑/自定义任务）
python scripts/train.py --task family_reach --algo ppo
# ② 训练中评估过验收子集 → 收敛（奖励平台 + eval_gate 全绿）
# ③ 导出 ONNX（布局/特权/限位三重校验）
python scripts/export_onnx.py --run runs/<run_id>/best
# ④ Sim2Sim 对拍
python scripts/sim2sim_check.py --run runs/<run_id>/best
# ⑤ 打本体包（行声明 + 签名）
python scripts/package_build.py --out <交付目录>   # 交付面工具 = body-sdk/(规范/签名/校验)
# ⑥ 仿真平台验收门（同一扇门，机械断言）
python scripts/verify_in_sim.py --package unitree-g1
# ⑦ 实机标定回填 → 真机阶梯（§9）
```
