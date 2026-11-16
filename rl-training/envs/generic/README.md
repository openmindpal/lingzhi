# envs/generic/——其他本体接入点(通用性)

任意非 G1 本体接入 = 三步,流水线/验收门不变:

1. **本体配置**:填 profile(J/关节序/限位/KP·KD/中立位)——来源 = 该本体的 body-package.toml
   (契约宽度按公式重算:`{b·J+10+6k}(+4 goal)`);
2. **MJCF/URDF**:提供与包 profile 同源的模型资产(关节名/序/限位一致——环境构造期逐名校验);
3. **任务定义**:复制 `tasks/custom/template.yaml`,改 family(四族取最贴近制式)/reward/gate。

引擎侧开放度边界(诚实声明):family = 引擎枚举制式(四族);开放的是实例名/行/词/权重。
契约/导出/打包/验收四层零改动。
