# 灵智系统仿真验证平台（simulation/）

> 定位（双职能）：**交付判定**（确定性 / 契约 / 安全链回归）+ **能力判定**（任务成功三信号）。
> 被验系统 = 灵智系统——本平台基于灵智系统创建并完成测试验证（保密声明见 RELEASE.md）。
> 唯一自建件 = 桥接器（bridge/）；仿真器（MuJoCo）、验收套件、任务集 = 本目录内数据与配置。
> 实测：`runner.run_all` **8/8 套件全绿（wall 18.7s，EXIT=0）**；云↔端金任务 e2e **GOLDEN_E2E_PASS**。

---

## 落地机制

**桥接器伪装本体端点**：

```
[本地 DDS 域（--domain 参数隔离）]
   ↓
[桥接器 = 虚拟机器人]  订阅指令 topic → PD 伺服 + 重力/科氏前馈 → mj_step(N 子步) → 发布传感 topic
```

伺服说明：`tau = kp·(tgt−q) − kd·dq + qfrc_bias`——qfrc_bias 为 MuJoCo 每步算好的
重力+科氏项（真机低层控制器的标准配置），使仿真低层保真度与真机一致。

## 与现有具身智能仿真平台对比

| 类型 | 代表 | 验证对象 | 与本平台的差异 |
|---|---|---|---|
| 物理引擎 | MuJoCo / PyBullet / Isaac Sim | 物理仿真本身 | 本平台以其为底座，验证其上的系统全链 |
| 能力 Benchmark | RoboDojo / ManiSkill / RLBench / Habitat | agent 在任务上的成功率与排名 | 本平台任务成功判据层与其方法论同构（三信号），另含 benchmark 不覆盖的系统交付判定（确定性回归/故障注入/安全链/契约门） |
| 数字 Agent 框架 | LLM→tool 循环框架 | 云端工具调用 | 无物理身体——不含 DDS/策略/安全链/极速闭环 |
| 机器人中间件 | ROS 2 / Gazebo | 消息传递 + 传感器仿真 | 不含云端认知引擎（目标分解/生命周期） |

本平台验证范围 = 云端认知 + 端侧执行 + 感知闭环 + 安全链 + 生命周期全链：

- **机械断言**：数值阈值 / hash 逐位比对 / 任务成功三信号，无"看起来对"判断
- **开放接口**：线协议（CDR-LE 428B/664B）、工具 schema、本体包规范全部文档化
- **引擎专有**：引擎为灵智系统专有组件，不随本目录开源——验证经其公开接口进行
- **第三方复现**：`docker compose` 一条命令跑全套件，无需凭据

---

## 套件一览（`runner.run_all` 8 套件，实测 8/8）

| # | 套件 | 判定 | 断言要点 |
|---|---|---|---|
| 1 | smoke_local | 桥接器本地冒烟 | 4 项（装载/步进/伺服/传感） |
| 2 | smoke_dds | DDS 线契约 | D1~D4（CDR-LE 编码/频率/隔离域/回环） |
| 3 | smoke_biped | 双足平衡回归 | B1~B4（平衡回归/可解析预测闭环实证） |
| 4 | goal_sweep_arm | 动作泛化扫掠（臂） | 10/10 点：到位精度/限位尊重/力矩有界 |
| 5 | goal_sweep_biped | 动作泛化扫掠（双足） | 10/10 点同判据 |
| 6 | longevity_biped | 长时运行 | L1~L5：轨迹 hash 逐位确定性 |
| 7 | fault_injection_arm | 故障注入 | F1~F4：断流/冻结帧/急停/力超限机械实证 |
| 8 | **task_success** | **任务成功判据（能力水平语义层判定）** | 10 任务 / 6 维度 / 双本体：到达·稳定·安全三信号 + 部分成功评分 + **系统得分 98.0/100** |

## 任务成功判据层（能力水平语义判定）

判定命题：goal_conditioned 策略回合 = 一次"任务"——单 episode 内按序到达声明的
路标位形（连续执行），**三信号机械判定**：

- **S1 到达**：路标窗口稳态位形与**解析预测**的残差 ≤ tol
  （预测 = 策略 ONNX 权重自描述导出的线性稳态解 `q* = (I−W_q)⁻¹(W_gx·gx + b)`
  ——判据随策略权重自动重算，非硬编码增益）
- **S2 稳定**：窗口收尾 max|Δq| ≤ settle_tol（到位且保持，非路过）
- **S3 安全**：限位违规 = 0 且力矩有界（违规 = 该任务计 0）

**任务集（10 任务 / 6 维度 / 双本体）**：

| 维度 | 任务 | 判定 |
|---|---|---|
| precision | reach_half / reach_far | 稳态 vs 解析预测残差 |
| horizon | long_horizon_3wp / 5wp（多路标按序） | 逐路标三信号 |
| repeat | return_repeat（A→B→A） | 重复一致性（实测差 = 0.000） |
| timed | timed_reach（限时 60 拍） | 限时内收敛 |
| disturbance | disturb_recover / hold_under_disturbance | 带载收敛 / 持扰保持（持续外扰注入） |
| cross_body | biped_reach / biped_long_horizon | 同判据第二本体复跑 |

**系统得分** = 100 × Σ任务分 / 任务数（**实测 98.0/100**——唯一扣分 = 带载任务的
合同内垂坠响应，物理如实）；安全违规 = 得分记 0。每次回归自动重算——策略升级后
得分变化即**语义级回归信号**（实现级 hash 回归的互补维度）。

---

## 目录结构（实际落地）

```
simulation/
├── README.md / RELEASE.md            ← 本文件 + 保密/边界声明
├── bridge/                           ← 桥接器（唯一自建件）——虚拟机器人 DDS 端点
│   ├── server.py                     ← 主循环：收目标流 → mj_step → 传感回流
│   ├── mj_model.py                   ← MuJoCo 装载/步进/传感 + PD 伺服（含重力/科氏前馈）
│   ├── smoke_local.py / smoke_dds.py / smoke_biped.py   ← 三段冒烟套件
│   ├── faults.py / time_control.py / wire_types.py / joint_sim_shim.py / camera_stream.py
├── packages/                         ← 仿真本体包（与 apps/body-packages 契约同构）
│   ├── sim-arm-6dof/                 ← 6DOF 臂：reach/greet_wave 策略 + tactician 快脑
│   └── sim-biped-29dof/              ← 29DOF 双足：balance/deform/reach/greet_wave
├── assets/
│   ├── models/                       ← arm_6dof / biped_29dof / sim2sim_pendulum
│   ├── scenes/                       ← tabletop_workspace / biped_ground / kitchen / factory / traffic
│   └── objects/                      ← 物体库（规划位）
├── suites/                           ← 8 验收套件（机械断言）
│   ├── sweeps/goal_sweep.py + domain_randomization.yaml
│   ├── longevity/run_soak.py
│   ├── fault_injection/run_faults.py
│   ├── sim2sim/compare.py + replay_pb.py
│   ├── baselines/capture.py          ← 金轨迹捕获
│   └── task_success/run_tasks.py     ← 任务成功判据（本 README「任务成功判据层」节）
├── runner/run_all.py                 ← 总编排器：8 套件一键 + 聚合报告（reports/）
├── baselines/trajectories/           ← 金轨迹（arm_6dof.json）
├── tasks/                            ← 任务 schema + 任务集（core_10/factory/kitchen/traffic）
├── tools/gen_sim_policies.py         ← 仿真策略生成器（行为可解析预测的线性策略）
├── training/                         ← Isaac Lab 训练环境配置（规划位）
├── demo/                             ← 网页演示（index.html + serve.py）
├── docker/                           ← compose.yaml / compose.fullstack.yaml + 金任务场景脚本 a1~a9
├── net/                              ← DDS 域配置位（当前经 --domain 参数隔离）
└── reports/                          ← 运行产出（gitignore——每次验证生成）
```

---

## 边界规则（硬约束）

| 规则 | 内容 |
|---|---|
| 随机化单一权威 | 域随机化表 = suites/sweeps/domain_randomization.yaml，training/ 引用不复制 |
| 报告不入库 | reports/ 全部 gitignore；baselines/trajectories/ 金轨迹才入库 |

---

## 快速启动（实测命令）

```bash
# 全套验收（8 套件一键，~18s；任一 FAIL = 退出码 1）
cd simulation && python -m runner.run_all

# 任务成功判据套件（可 --tasks 子串过滤）
cd simulation && python -m suites.task_success.run_tasks

# 云↔端金任务 e2e（真 Rust device-agent + joint_sim 反馈源；仓库根执行）
bash e2e/golden/run.sh

# 第三方独立复现（无凭据）
docker compose -f simulation/docker/compose.yaml up
```

诚实边界（真机阶梯 T1-T4 / 真训权重 / 触觉力控执行面 / 多视角融合）见 RELEASE.md 与
仓库根 `测试报告.md` §诚实边界——逐项归属，不在本文件展开。
