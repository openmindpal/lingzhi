# examples/unitree-g1——样例本体包(清单样例)

> 本目录 = **清单样例**(body-package.toml:段结构/策略行/快脑行/标定/签名口径的完整示范)。
> **权重(policies/*.onnx)不随样例分发**——权重 = 各厂商经 rl-training 平台训练产出的交付物。
> 因此对本目录直接跑 `tools/validate.py` 会如实报"权重缺失"——这是机械校验的正确行为。
>
> **传输微码待办**:本样例 `[protocol]` 已声明宇树 `rt/lowcmd`/`rt/lowstate` 制式(声明段 ✓);
> 低层载荷编码微码(microcode/*.so——LowCmd IDL 打包 + CRC)为待办项(样例 toml 头部
> "已知边界"即此标注)。规格见 `spec/本体包规范.md` §0.5——引擎零改动,每厂商一次性。
>
> **传感声明注记**:joint_state(rt/lowstate,含 q/dq/tau+IMU)与相机已声明;
> **腕部六维力×2 的接触力数据面未声明**(reach_contact 行的待办——选配硬件到位后按
> spec §2.5 声明,无传感 = 该行不可交付);音频同样属选配声明。
>
> 完整可验证包的产出路径:rl-training 训练 → export_onnx → package_build(自动继承本样例
> 的 profile/标定为基座,重生成交付段)→ 产出含权重的完整包 → validate 全绿 → 交付。
