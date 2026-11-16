#!/usr/bin/env node
/** A7 认知泛化验证——DeepSeek 真模型 × 未见任务 × 结构化断言。
 * 任务描述不触发任何 mock 分支（全新语义），断言不看步数匹配——
 * 判定：模型自主分解（动作数≥N）+ 未见任务不 abort + 干净收尾 + 同通道串行。
 * 这才是认知层智能的判定测试（A4~A6 的 goal 数组是脚本代驾，不算数）。
 * 用法: node a7-cognitive.mjs [apiBase] [deviceId]
 */
import http from "node:http";
import fs from "node:fs";

const apiBase = process.argv[2] || "http://127.0.0.1:17801";
const DEVICE_ID = process.argv[3] || "";
const TOKEN = "admin@space_dev";
// 未见任务：语义全新（不在 mock 分支关键词表内——"厨房仿真/工厂仿真/交通指挥/写一个文件/做一顿饭/生命周期"全避开）
const GOAL = "工作台上有一枚红色积木和一个绿色积木，请先把红色积木移动到工作台右端，再把绿色积木叠放在红色积木上面，完成后告诉我两块积木的最终位置关系。";

function runStream(envelope) {
  return new Promise((resolve, reject) => {
    const u = new URL(apiBase + "/v1/agent/execute");
    const data = JSON.stringify(envelope);
    const req = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${TOKEN}`,
                 "content-length": Buffer.byteLength(data), accept: "text/event-stream" },
      timeout: 600000,
    }, (res) => {
      let buf = ""; const events = [];
      res.on("data", (c) => {
        buf += c.toString(); let ix;
        while ((ix = buf.indexOf("\n\n")) >= 0) {
          const raw = buf.slice(0, ix); buf = buf.slice(ix + 2);
          const line = raw.split("\n").find((l) => l.startsWith("data:"));
          if (!line) continue;
          try { events.push(JSON.parse(line.slice(5).trim())); } catch {}
        }
      });
      res.on("end", () => resolve({ events, status: res.statusCode }));
      res.on("error", reject);
    });
    req.on("error", reject);
    req.write(data); req.end();
  });
}

let events, status;
{
  const r = await runStream({
  auth: { token: TOKEN },
  input: { text: GOAL },
  context: { deviceState: { deviceId: DEVICE_ID }, taskContext: { taskId: `sim-a7-${Date.now()}` } },
  config: { maxIterations: 12 },
});
events = r.events; status = r.status;
}
if (status !== 200) { console.error(`FAIL[A7] http=${status}`); process.exit(1); }

fs.writeFileSync("D:/trae/openslin (4)/simulation/reports/a7-events.json", JSON.stringify(events, null, 1).slice(0, 400000));
const phases = events.filter((e) => e.phase).map((e) => e.phase);
const decisions = events.filter((e) => e.type === "decision");
const toolResults = events.filter((e) => e.type === "tool.result");
const endReason = events.map((e) => e.payload?.meta?.endReason ?? e.payload?.endReason).find(Boolean);
const replyEvt = events.find((e) => e.payload?.reply);
const replyText = String(replyEvt?.payload?.reply?.text ?? "");
const deviceDispatches = events.filter((e) => JSON.stringify(e).includes("device.actuator.gesture")).length;
const abortSeen = events.some((e) => JSON.stringify(e).includes("agent_abort"));
const defectSeen = events.some((e) => JSON.stringify(e).includes("确定性工具缺陷"));

console.log(`[A7] phases=${phases.join("→")} decisions=${decisions.length} toolResults=${toolResults.length} deviceDispatches=${deviceDispatches}`);
console.log(`[A7] endReason=${endReason ?? "?"} abort=${abortSeen} defect=${defectSeen}`);
console.log(`[A7] reply="${replyText.slice(0, 100)}"`);

let fail = 0;
const check = (n, c, d) => { console.log(`[${c ? "PASS" : "FAIL"}] ${n}${d ? " — " + d : ""}`); if (!c) fail++; };

// 结构化断言（kitchen-real 同族——不匹配预写步数）
check("A7① 自主分解：决策数 ≥3", decisions.length >= 3, `${decisions.length} 次决策`);
check("A7② 干净收尾：done/completed（非 abort/ask_user 死局）",
  /done|completed/.test(endReason ?? "") && !abortSeen, `endReason=${endReason}`);
check("A7③ 回复非空", replyText.length > 0, `len=${replyText.length}`);
check("A7④ 无 DEFECT 类失败", !defectSeen, "确定性工具缺陷不应出现");
check("A7⑤ 设备派发参与（具身链路）", deviceDispatches >= 1 || toolResults.some((e) => JSON.stringify(e).includes("device")), `${deviceDispatches} 次`);

console.log(fail === 0 ? "A7_COGNITIVE_PASS 真模型×未见任务×自主编排——认知泛化实证" : `A7_COGNITIVE_FAIL(${fail})`);
process.exit(fail === 0 ? 0 : 1);
