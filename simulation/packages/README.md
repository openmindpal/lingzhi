
# 灵智系统仿真本体包（packages/）

> 被验系统 = 灵智系统（引擎为灵智系统专有组件，不在本平台开源范围——保密声明见 RELEASE.md）。

> 与 `apps/body-packages/` **契约同构**（同一 schema、同一校验器、同一装载门），
> （`simulation/net/cyclonedds.localhost.xml`），由桥接器扮演端点。

## 规则

| 规则 | 内容 |
|---|---|
| 契约 | 字段级规范随合作交付（引擎侧填写规范，与本目录包体逐字段同规） |
| 权重 | policies/、fastbrains/ 内 ONNX = 测试用权重（生成器产出或训练产出），测试策略/快脑全部在此 |

## 包清单

| 包 | 用途 |
|---|---|
| sim-arm-6dof/ | 冒烟最小包——桥接器自测、套件冒烟、obs 宽度集最小算例 |
| sim-biped-29dof/ | 双足对照包——G1 仿真实例（对照 fixtures/tri-family），tri 全套+扫掠+长跑 |
