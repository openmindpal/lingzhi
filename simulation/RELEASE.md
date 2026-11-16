# 灵智系统仿真验证平台——发布说明（simulation/）

本仿真平台基于灵智系统创建并完成测试验证。全部验收套件以灵智系统为被验对象：

- 端侧引擎经其公开 DDS 接口接入仿真桥接器（虚拟机器人端点）
- 云端认知引擎经公开 REST/SSE 接口接入断言脚本（金任务场景 A1~A9）
- 感知链经工具 schema（open_vocab / device.perception.latest / macro_latest）接入任务流

验证矩阵（见下）记录验收判据与验证结果；引擎实现不在记录范围。
平台结构与套件判据详见 [README.md](./README.md)（双职能定位 / 8 套件 / 任务成功判据层）。

## 保密声明

灵智系统引擎为灵智系统专有组件，不在本平台开源范围：

| 组件 | 分发形态 |
|---|---|
| 端侧引擎（Rust） | release 二进制接入验证，源码不分发 |
| 云端认知引擎 | 不随本目录分发，验证仅经其公开接口 |

本平台开源件 = 下列八目录（桥接器 + 验收套件 + 任务集 + 本体包 + 资产 + 金标准 + 演示 + Docker 验证包）。

## 开源件清单（八目录）

| # | 件 | 位置 | 复现入口 |
|---|---|---|---|
| 1 | 桥接器（唯一自建件，13 文件） | simulation/bridge/ | `python -m bridge.server --model … --package …` |
| 2 | 验收套件（8 套件） | simulation/suites/ | `python -m runner.run_all` |
| 3 | 任务集（22 任务） | simulation/tasks/（schema + core_10 + factory + kitchen + traffic） | 第三方加 JSON 即接入 |
| 4 | 仿真本体包（2 包） | simulation/packages/（sim-arm-6dof / sim-biped-29dof） | 装载门 + 套件引用 |
| 5 | 仿真资产 | simulation/assets/（models×4 / scenes×5 / objects 规划位） | 套件/任务集引用 |
| 6 | 金标准 | simulation/baselines/trajectories/（固定种子金轨迹） | `python -m suites.baselines.capture --verify` |
| 7 | 演示画面 | simulation/demo/（仪表盘） | `python -m demo.serve --port 8089` |
| 8 | Docker 验证包（15 文件） | simulation/docker/（Dockerfile + compose×2 + 金任务场景脚本 a1~a9 + run-fullstack + entry-joint-sim + std_probe） | 见复现步骤 |

## 接口契约

| 接口 | 线格式 | 方向 |
|---|---|---|
| 关节指令（JointTargets） | CDR-LE 428B | 引擎 → 桥接器（虚拟本体） |
| 关节状态（JointStates，含 IMU/FT 块） | CDR-LE 664B | 桥接器 → 引擎（传感回流） |
| 认知派发 / 回执 | REST + SSE / JSON | 云端 ↔ 引擎（金任务断言面） |

## 复现步骤（第三方验证者，全部无凭据）

```bash
# ① 仿真台全绿（8 套件，Linux 容器内）
docker build -f simulation/docker/Dockerfile -t mindpal-sim-verifier .
docker compose -f simulation/docker/compose.yaml run --rm verifier

# ② DDS 线契约（consumer 探针 ↔ bridge）
docker compose -f simulation/docker/compose.yaml --profile live up -d bridge
docker compose -f simulation/docker/compose.yaml run --rm consumer

# ③ tri 引擎在环（真 DDS + 真 ORT + MuJoCo resident，8/8）
docker compose -f simulation/docker/compose.yaml --profile engine run --rm engineloop

# ④ 云↔端全栈金任务（A1~A9 场景 + 流程 C/C2 具身闭环；GOLDEN_E2E_PASS 裁决）
#    可选：DEEPSEEK_API_KEY=sk-… 环境变量注入真模型路由（down -v 清卷后自动恢复绑定）
bash simulation/docker/run-fullstack.sh

# ⑤ Sim2Sim 对拍（100 控制拍 |Δq| < 5°）
python -m suites.sim2sim.compare

# ⑥ 金轨迹回归锚（固定种子 hash 逐位）
python -m suites.baselines.capture --verify
```

## 已实证验证矩阵

| # | 套件 | 判据 | 结果 |
|---|---|---|---|
| 1 | 仿真台冒烟（S1~S4 / D1~D4 / B1~B4） | 模型/包一致 + PD 闭环 + 线打包 428B/664B | ✓ Win+Linux 双实证 |
| 2 | goal_sweep 扫掠（双包） | 20 点 × 到位精度/限位/力矩 | ✓ 20/20 |
| 3 | 长跑 | L1 确定性 hash / L2 RSS / L3 限位 / L4 力矩 / L5 加速 | ✓ |
| 4 | 故障注入 | F1 断流 / F2 冻结帧 / F3 estop / F4 力超限 | ✓ |
| 5 | 线契约 | C 包装库 ↔ Python 桥全块读回 | ✓ PROBE_OK |
| 6 | 生命周期 | A3 挂起→续跑→checkpoint 注入→迭代硬帽 | ✓ |
| 7 | 厨房/工厂/交通编排 | A4/A5/A6 goal 参数化×3 + DB 核验 | ✓ 9/9 |
| 8 | Sim2Sim | 100 控制拍 max\|Δq\| < 5° | ✓ j1 0.55° / j2 3.50° |
| 9 | 金轨迹回归 | 固定种子 SHA-256 hash 逐位 | ✓ |
| 10 | **任务成功判据（双本体）** | 10 任务 × 三信号（到达/稳定/安全）+ 部分成功评分 + 系统得分 | ✓ **10/10，98.0/100** |
| 11 | **云↔端金任务 e2e** | A 认知循环 / B 工具链 / C 派发接缝 / C2 具身运动闭环 | ✓ **GOLDEN_E2E_PASS** |

## 任务集

| 任务集 | 文件 | 任务数 | 场景 | 感知依赖 |
|---|---|---|---|---|
| core_10 | tasks/sets/core_10.json | 10 | tabletop | 无（goal 参数化） |
| kitchen | tasks/sets/kitchen.json | 4 | kitchen.xml | reach_goal |
| factory | tasks/sets/factory.json | 4 | factory.xml | reach_goal |
| traffic | tasks/sets/traffic.json | 4 | traffic.xml | reach_goal |

任务 schema：`tasks/schema.json`（JSON Schema 2020-12）。第三方扩展 = 加 JSON 文件，零代码。

## 感知链接口

| 工具 | 功能 | 输入 | 输出 |
|---|---|---|---|
| device.perception.latest | 读取主相机 OV 检出 | （空） | count + detections[label/bearing/dist] + detector_available |
| device.perception.macro_latest | 读取宏相机 OV 检出 | （空） | count + detections[label/camera/pixel/ee_dir] |

## 诚实边界（当前未覆盖，逐项归属）

- **Sim2Sim 视窗**：100 控制拍（0.2s）——相漂移随视窗 Lyapunov 放大
- **真机阶梯**（T1 标定回填 / T2 首指令 / T3 保守增益在环 / T4 实机验收）——仿真全绿 ≠ 真机绿
- **真训权重**：各本体策略/快脑 ONNX 由训练管线产出（推理/装载/派发链路已由夹具权重在环验证）
- **仿真本体能力**：无夹爪（抓取 = reach_goal 位姿到达）；夹爪开合 = 本体能力扩展项
- **触觉力控闭环 / 多视角融合**：后续能力（obs_contact 观测机制与安全联锁已就位）
