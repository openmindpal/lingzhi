# rl-training——灵智系统 RL 训练平台(参考实现)

> 被验系统 = 灵智系统。平台定位:把"训练产出"变成"过验收门的本体包权重"。
> **训练产物用途 = 灵智系统的四族策略与快脑**——装入本体包,由灵智认知层按 词+参数 派发执行。
> 总架构: `灵智系统RL训练平台技术架构.md`(本目录内,权威规划)。
> 契约唯一权威 = 灵智系统引擎契约(随合作交付,不在本仓库)。

## 快速开始(单机参考流水线,mujoco 后端,CPU 可跑)

```bash
cd rl-training
# ① 训练(任一族/快脑/自定义)
python scripts/train.py --task family_reach --envs 16 --iters 50 --steps 256
# ② 导出 ONNX(布局/特权/限位三重校验,归一化烘焙)
python scripts/export_onnx.py --run runs/<run_id>
# ③ 验收门(训练侧,机械断言)
python scripts/eval_gate.py --run runs/<run_id>
# ④ 打本体包(继承 unitree-g1 profile + 行声明 + 签名)
python scripts/package_build.py --run runs/<run_id> --sign-key <pem>
# ⑤ 包装载门
python scripts/verify_in_sim.py --package runs/<run_id>/package
# ⑥ Sim2Sim(工具链基准 / 策略在环)
python scripts/sim2sim_check.py --mode smoke
# ⑦ 集成门(仿真平台全套件,需 docker 镜像)
python scripts/verify_in_sim.py --docker
# ⑧ 平台自检(一条命令,零 GPU——契约/装配/生成器/环境/学习/导出打包/负例)
python tests/run_all.py            # 全套(约 2~4 分钟)
python tests/run_all.py --fast     # 快速(约 30 秒)
# ⑨ 整包交付(多 run 合成一个本体包:四族全部行+快脑)
python scripts/package_build.py --run runs/<balance> --run runs/<reach> \
       --run runs/<deform> --run runs/<fb> --sign-key <pem>
```

## 目录

| 目录 | 内容 |
|---|---|
| contracts/ | 契约镜像(观测布局/动作语义/ONNX IO/快脑槽)——唯一权威 = 灵智系统引擎契约(随合作交付) |
| rltrain/ | 核心库:contract / g1_cfg(与 unitree-g1 包逐项同源)/ obs_assembler(白名单装配)/ tasks / env_mj(自由底座+PD+电机延迟/课程)/ **rewards(后端无关奖励/终止共享层——MuJoCo 与 Isaac 同函数同权重)** / mjcf_gen(参数化本体:手/物体/墙/地形)/ isaaclab_env(GPU 后端,direct 风格)/ sac / urdf_gen / demos(演示数据面)/ ppo |
| envs/ | g1(body_spec[样例]+scene/randomize 引用制+资产)/ generic(其他本体接入点)/ sensors |
| tasks/ | 17 运动模式任务(稳定/步态/位置/力觉/形态/节律/抓取/腾空/闪避/恢复/柔性/倒水/穿针/双手/手内/**拦截/击打**)+ 快脑 + custom 模板 |
| algos/ | ppo / sac / recurrent(研究级,导出门仅 MLP)/ il_pretrain(demos 数据面已立) |
| scripts/ | train → export_onnx → eval_gate → package_build(单/整包合并) → verify_in_sim → sim2sim_check → record_demo |
| **body-sdk(姊妹套件)** | **交付面**:`body-sdk/`(仓库根,与 simulation/ 同级)——本体包规范/签名工具/独立校验器/样例清单(本平台打包/签名自动调用) |
| tests/ | **平台自验证套件**:run_all.py 一条命令,机械断言,零 GPU |
| runs/ | 训练产出(gitignore:ckpt/ckpt_best/manifest/metrics/exports/package/tb) |
| docs/ | 训练日志纪律 |

## 厂商对接指南(运动能力供货口径)

> 自检声明:本节每条均可溯源至已公开材料(contracts/、灵智系统引擎契约(随合作交付)、本 README),
> 零新增技术信息;引擎内部实现/源码路径/内部参数不在此节范围。

| 问 | 答 |
|---|---|
| 你有什么 | 训练团队 + GPU(4090 级即可起步)+ 本体资产(MJCF/URDF,关节名/序/限位与本体包同源) |
| 你交付什么 | 契约对齐的 ONNX 策略行(obs 逐位 = contracts/obs_layout、动作 = 绝对关节目标、软限位内)+ 签名本体包 |
| 流程五步 | 本体建模(按包 profile)→ 域随机化训练(本平台流水线)→ Sim2Sim(<5°)→ 实机标定回填 → 交付 |
| 验收谁判 | tri 在环套件 + 仿真平台门——机械判据公开可复现,同门质检(判据表见灵智系统验收协议,随合作交付) |
| 上线后谁调用 | 灵智认知层按 词 + 参数 派发(词 = 你声明的策略行名;参数 = goal/速度/风格等契约参数) |

与既有训练流程的三个差异(对照厂商惯例):

| 厂商惯例 | 本平台口径 | 原因 |
|---|---|---|
| 自有接口(cmd_vel 速度指令等) | 已废除——行走速度/步态风格 = 多实例行,按行名派发 | 策略是被外部系统调用的技能词,不是遥控终端 |
| 观测可含仿真特权信息 | 白名单制(现实传感器同款来源) | 真机无此输入 = 权重上机即瞎(Sim2Real 第一杀手) |
| 自定性能指标自验 | 同一套机械判据门(tri_01~07 + 仿真套件) | 公开判据、可复现、同门质检 |

> 为什么必须是这种策略(三条本质要求 / 泛化来源 / 惯例形态的结构性缺陷)——立论见
> 《灵智系统RL训练平台技术架构》§1(策略形态论证)。

## 契约要点(训练侧硬约束)

- 观测 = 白名单装配(`[q(29)|dq|tau|imu(10)|contact(6k)?|goal(4)?]`),G1 宽度 97/101/109;
  **禁特权信息**(装配器拒收契约外键);引擎不归一化 → 导出烘焙。
- 动作 = J=29 **绝对关节目标**(neutral + tanh(a)·scale,软限位钳制头)。
- ONNX = 单输入 `[1,obs_dim]` / 首输出 `[1,J]` / opset 17;循环网络导出**当前拒收**
  (引擎 IO 单输入——见 algos/recurrent.yaml 诚实边界)。
- family = 四族引擎枚举;开放度 = 实例名/行/词。

## 后端

| 后端 | 状态 | 用途 |
|---|---|---|
| mujoco(默认) | ✓ 单机即可跑(CPU/单卡) | 流水线参考实现、小规模训练、冒烟 |
| isaaclab | ✓ 后端已重写(rltrain/isaaclab_env.py,direct 风格:共享奖励层/真实 contact 传感器/契约 imu(10)/电机延迟环形缓冲/推力课程/行级 auto-reset);**GPU 首跑按文件头 C1~C8 校准点核对 API**;v1 机制覆盖 = 无外物全集(balance/gait/reach/reach_contact/deform/expression/recover/athletic),外物/柔性/粒子任务构造期诚实拒收(USD 导出适配点) | GPU 大规模并行(4096 env,RTX 4090 单卡) |

## 欠账关闭状态(原 §10 清单)

| 项 | 状态 |
|---|---|
| isaaclab GPU 适配器 | ✅ 已重写(direct 风格 + rewards 共享层 + 批式向量化;动作=set_joint_position_target) / ⏳ GPU 首跑校准(文件头 C1~C8:资产/场景/传感器/物理批写 API 逐项核对)+ mass/friction 抖动接钩(C5) |
| isaaclab 外物机制 | ⏳ mjcf_gen 资产的 USD 导出适配点——外物/柔性/粒子任务(dodge/intercept/strike/pour/needle/bimanual/inhand/grasp/wipe)GPU 路径诚实拒收至该项落地 |
| 参数化本体生成器 | ✅ rltrain/mjcf_gen.py——自由底座/总质量标定/**灵巧手(Dex3-1 样例)**/被抓物体+桌面/按压面;几何[样例]→宇树规格回填 |
| 灵巧手包全链 | ✅ J=41(grasp_can): 训练→导出[1,149]→[1,41]→打包→装载门 PASS |
| SAC 力控备选 | ✅ rltrain/sac.py + train.py --algo sac(导出同路径) |
| Sim2Sim URDF 资产 | ✅ rltrain/urdf_gen.py 从编译模型自动生成 g1_pb.urdf(sim2sim --mode policy 自动补资产) |
| 自验证测试套件 | ✅ tests/run_all.py 一条命令(契约/装配/生成器/16环境/GPU契约[共享奖励/终止层+isaaclab诚实门]/学习冒烟/导出打包往返/负例注入) |
| 整包交付(多策略合并) | ✅ package_build --run 可重复——四族全部行+快脑合成一个本体包(j 一致性/行名唯一/goal_stream 交叉校验) |
| mujoco 随机化补齐 | ✅ 电机延迟(0~20ms 环形缓冲)/不平地形(±3cm 台块)/扰动课程(set_curriculum) |
| best-ckpt+训练中评估+tb | ✅ eval_quick 每 N 迭代→ckpt_best.pt+manifest.best;tensorboard 可选依赖 |
| 演示数据面(il_pretrain) | ✅ rltrain/demos.py + scripts/record_demo.py(格式/加载器/CLI 已立) |
| pybullet(sim2sim 运行时) | 依赖项:`pip install pybullet`;未安装 = sim2sim 诚实报错(不假装对拍) |
| 宇树官方资产回填 | ⏳ body_spec.yaml 的[样例]几何/质量 → 官方规格书(生成器不变) |
| lowcmd IDL 载荷编码 | ⏳ 端侧对接项(非平台)——完成前验证止步"包装载+仿真验收全绿" |
| il_pretrain 适配点 | ✅ BC 预训练实装(--pretrain-demos:演示 obs/act → 策略热启动 + RL 微调;契约不变)/Diffusion·ACT 仍为扩展点 |
| recurrent 导出门 | 契约约束(引擎单输入)——保持拒收,非欠账 |

GPU 时长参考(架构文档 §10):四族+快脑收敛跑合计约 1~2 周(单卡顺序)/ 数天(按族并行);
收敛判据 = 奖励平台 + 验收门通过,不以固定步数为准。

## 族×实例总表(15 任务的组织结构——族清晰,实例务实)

**族 = 引擎制式标签(4 个,固定);实例 = 族内原语(按力学结构划分,参数化泛化)——"乱"的感觉来自只看实例数;看族层:每族涵盖清晰。**

| 制式族 | 实例(row) | 力学结构 | 归并空间(远期) |
|---|---|---|---|
| **balance 动态平衡** | balance_hold | 站立/抗扰(不倒) | — |
| balance | athletic_air | 腾空/跳跃(airborne 语义) | 远期:全身动力学统一课程(研究方向) |
| balance | dodge_react | 感知反应位移 | 远期:同上(扰动通道不同) |
| balance | self_right | 摔倒恢复(倒地翻转回位) | 远期:同上(多接触滚动) |
| balance | (行走/步态实例) | 行走/步态变体 | ⏳待定义(多实例行) |
| **manipulation 操作** | reach_goal | 位置到达(自由空间) | — |
| manipulation | reach_contact | 力觉到位 | 远期:wipe 并入(力控到达+跟踪=移动 goal 变体) |
| manipulation | wipe_const | 恒力贴面+轨迹 | 远期:同上 |
| manipulation | needle_insert | 精细对准插入 | 远期:reach 精档课程候选 |
| manipulation | pour_hold | 持液搬运(液体动力学) | — |
| manipulation | bimanual_coord | 双臂力耦合 | — |
| manipulation | grasp_lift(手包) | 手指包络 | — |
| manipulation | inhand_spin(手包) | 手内旋转 | — |
| **deform 形变体** | deform_ramp | 形态转移(蹲) | 跪/弯/转身 = 同族实例行 |
| **expression 表达** | greet_wave | 节律表达 | 风格标量参数化扩展 |

**为什么实例分开训而不是一个大策略**:①奖励/初始分布/接触 schedule 不同——合并会互相干扰 ②单原语收敛快、验收清晰、失败可归因 ③词表按行名派发——组织机制天然容纳多实例。远期归并 = 研究方向(全身统一课程),当前分开 = 务实正确。

## 技能库收敛口径(原语增长边界)

原则:**增长的是任务,收敛的是原语**——新任务优先走认知组合(零训练),新原语仅在"运动模式级"增量。

| 新需求类型 | 训练? | 承载路径 |
|---|---|---|
| 新物体 | ❌ | goal 实体引用 + 感知参数(YOLOX/GDINO 检出) |
| 新序列/新流程 | ❌ | 认知层分解(原语序列组合) |
| 新参数域(更远/更重) | ❌ | 课程采样扩域(goal_box/f_range) |
| **新运动模式**(力学结构不同) | ✅ | 新原语进 tasks/custom → 同一流水线训练 |

边际递减:任务覆盖 ≈ 原语数 N × 参数组合 × 序列组合(认知编排)——四族各 5~15 原语,
总量几十个即达**运动完备集**(终态);此后新增原语仅补力学结构差异,边际趋零。

攻坚清单(运动模式级新原语,有限项、清单制):

| 模式 | 任务定义 | 训练 |
|---|---|---|
| 高动态全身(翻跟头/跳跃) | ✅ family_athletic(airborne 语义) | ⏳ GPU |
| 反应闪避(拳击躲避) | ✅ family_dodge(威胁向量经 goal 块——契约演化提案,真机前与引擎 §2.2 对齐) | ⏳ GPU |
| 摔倒恢复(self-right) | ✅ family_recover(倒地初始位) | ⏳ GPU |
| 柔性接触(擦拭/布) | ✅ family_soft(flex 布+恒力贴面) | ⏳ GPU |
| 流体动态(倒水不洒) | ✅ family_pour(杯焊手+粒子水,洒出计数) | ⏳ GPU |
| 超高精度稳态(穿针) | ✅ family_needle(3mm 缝+插入深度) | ⏳ GPU |
| **双手协同**(双臂力耦合) | ✅ family_bimanual(goal 条件化课程——无限双手场景)+ 云端原子 robot.bimanual.coordinate | ⏳ GPU |
| **手内精细**(指尖级) | ✅ family_inhand(手包 J=45,转角课程——无限手内场景) | ⏳ GPU |
| **拦截同步**(接物/时间同步到达) | ✅ family_intercept(下落球+时间误差判定,契约演化提案: goal 块剩余时间维) | ⏳ GPU |
| **冲击击打**(挥摆冲量) | ✅ family_strike(立柱球受击,冲量代理=球速) | ⏳ GPU |

落地规则:新任务接入先判**原语复用**——复用路径 = 认知分解 + 参数(tasks/custom 不动);
确认运动模式缺失才进 tasks/ 训练,并在 docs/ 训练日志登记"缺失原语"条目。

## 诚实边界

- 训练变体 = 冒烟拓扑 + body_spec 叠加(参数化生成器)——几何/质量为[样例],
  宇树规格书回填后重生成;完整本体重精化 = 官方资产替换点(拓扑接口不变)。
- reach_contact 生产训练需按压面(spec 已含,press_wall 开关)+ 真机腕部六维力传感。
- **灵巧手不在 29 DOF 内**(五指外接设备)——手指级抓握走 family_grasp(assets.hands=true,
  J 29→29+2×N,spec.fingers 段链驱动;[样例] 拇指2+食指2+中指2+无名指1+小指1=8/手 → J=45,
  契约公式自动适配);已训 29 关节族策略不因手配置作废(手 = 叠加外设)。
- Sim2Sim policy 模式 URDF 自动生成;pybullet 缺失 = 诚实报错(不假装对拍)。
- G1 真机前置:unitree_sdk2 lowcmd IDL 载荷编码(包文件头边界 1)未完成前,
  平台验证到"包装载 + 仿真验收全绿"为止。
