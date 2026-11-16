# envs/——多厂商本体接入点

| 目录 | 厂商/本体 | 状态 |
|---|---|---|
| g1/ | 宇树 G1(29 DOF;五指手包 45) | ✅ 完整接入实例(body_spec/body_cfg/资产/任务全通) |
| agibot/ | 智元机器人(Agibot) | 适配点——规格[待回填] |
| tiangong/ | 天工机器人(北京人形机器人创新中心) | 适配点 |
| galbot/ | 银河通用(Galbot) | 适配点 |
| engineai/ | 众擎机器人(EngineAI) | 适配点 |
| booster/ | 加速进化(Booster Robotics) | 适配点 |
| xsquare/ | 自变量机器人(X Square) | 适配点 |
| ubtech/ | 优必选(UBTech,Walker S) | 适配点 |
| fourier/ | 傅利叶智能(Fourier,GR 系列) | 适配点 |
| kepler/ | 开普勒(Kepler,先行者 K2) | 适配点 |
| robotera/ | 星动纪元(Robot Era,STAR 系列) | 适配点 |
| songyan/ | 松延动力(Songyan Dynamics,小诺系列) | 适配点 |
| tesla/ | 特斯拉(Tesla,Optimus) | 适配点 |
| figure/ | Figure AI(美国,Figure 02/03) | 适配点 |
| generic/ | 通用本体模板 | 任何其他本体的接入入口 |

接入三步(全部同构):①本体配置(profile:J/关节序/限位/KP·KD/中立位)
②模型资产(MJCF/URDF,与 profile 同源)③任务定义(tasks/custom 模板)。
平台契约与验收门不变;契约宽度公式 {b·J+10+6k}(+4) 随 J 自动适配。
