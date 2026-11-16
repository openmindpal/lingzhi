# envs/g1/assets/——训练资产

| 文件 | 来源 | 说明 |
|---|---|---|
| g1_train_free.xml | 自动生成 | 训练变体 = 冒烟 MJCF(biped_29dof.xml)+ pelvis freejoint;由 `rltrain.g1_cfg.build_free_base_mjcf()` 生成(删本文件即重生成) |
| g1_pb.urdf | 待生成 | PyBullet 重放资产(Sim2Sim policy 模式需要)——MJCF→URDF 转换:①MuJoCo 导出(mujoco.MjModel → mjcf2urdf 工具链)②或按包 profile 手工构造(关节名/序/限位同源);缺失 = sim2sim_check --mode policy 诚实报错,不跳过 |
