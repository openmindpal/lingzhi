# envs/sensors/——观测装配器

实现唯一来源 = `rltrain/obs_assembler.py`(避免双份实现漂移)。

职责:
- 按契约布局逐位拼装 `[q|dq|tau|imu|contact?|goal?]`(块序无间隔);
- **白名单制**:只接受物理传感来源键(q/dq/tau/imu/contact/goal)——特权信息
  (物体真值位姿/上帝视角状态)进观测 = 拒绝(Sim2Real 第一防线);
- 宽度自检(≠ 契约宽度即抛)+ 非有限值自检(传感故障诚实暴露,不补零冒充)。

契约: contracts/obs_layout.md(唯一权威 = 灵智系统引擎契约 §2,随合作交付)。
