


# 灵智系统仿真任务集（tasks/）

> 被验系统 = 灵智系统——任务经公开接口派发至灵智系统执行并回收判据。

## schema.json

任务定义 schema——字段含：goal（自然语言+结构化目标）/ 判据（successCriteria
（物体/位置/干扰）。schema 定型前以本 README 占位，首任务集落地时冻结。

## 第三方扩展（验证权开放）

1. 按 schema 加 `sets/<你的任务集>.json`——不改套件代码
2. `python -m runner.run_all --taskset sets/<你的>.json`
3. 报告含逐任务通过率——独立得出结论（§5.2 复现流程第 4 步）



任务集如实报告失败，不粉饰。
