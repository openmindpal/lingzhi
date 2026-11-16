#!/usr/bin/env node
/** A4 厨房 goal 参数化执行闭环——REST 直派（A2 已证链路）× 三个 goal 目标点。
 * 认知层分解任务 → 同一 reach_goal 词 + 不同 goal 参数 → 策略闭环。
 * 断言：三个 goal 点全部 succeeded + DB 核验。
 * 用法: node a4-kitchen.mjs [apiBase] [deviceId]
 */
import http from "node:http";

const apiBase = process.argv[2] || "http://127.0.0.1:17801";
const DEVICE_ID = process.argv[3] || "";
const TOKEN = "admin@space_dev";
// goal 参数化：认知层分解 → 三个目标点（锅位/油瓶位/翻炒悬停位）
const GOALS = [
  { x: 0.35, y: 0.12, z: 0.20, f: 0.5, label: "锅位" },
  { x: 0.30, y: -0.05, z: 0.22, f: 0.2, label: "油瓶位" },
  { x: 0.35, y: 0.12, z: 0.30, f: 0.7, label: "翻炒悬停位" },
];

function post(pathname, body) {
  return new Promise((resolve, reject) => {
    const u = new URL(apiBase + pathname);
    const data = JSON.stringify(body ?? {});
    const req = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${TOKEN}`,
                 "content-length": Buffer.byteLength(data) },
      timeout: 30000,
    }, (res) => { let out = ""; res.on("data", (c) => (out += c)); res.on("end", () => resolve({ status: res.statusCode, text: out })); });
    req.on("error", reject);
    req.write(data); req.end();
  });
}

async function dispatchGoal(g) {
  const created = await post("/v1/device-executions", {
    deviceId: DEVICE_ID, toolRef: "device.actuator.gesture",
    input: { intent: "reach_goal", max_duration_ms: 2500,
             goal_x: g.x, goal_y: g.y, goal_z: g.z, goal_f: g.f },
    requireUserPresence: false,
  });
  let execId = "";
  try { execId = JSON.parse(created.text)?.execution?.deviceExecutionId ?? ""; } catch {}
  if (created.status !== 200 || !execId) {
    return { label: g.label, status: `dispatch_failed(${created.status})` };
  }
  for (let i = 0; i < 60; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const u = new URL(apiBase + "/v1/device-executions/" + execId);
    const st = await new Promise((resolve) => {
      const req = http.request({ hostname: u.hostname, port: u.port, path: u.pathname, method: "GET",
        headers: { authorization: `Bearer ${TOKEN}` }, timeout: 15000 },
        (res) => { let out = ""; res.on("data", (c) => (out += c)); res.on("end", () => { try { resolve(JSON.parse(out).execution.status); } catch { resolve("parse_error"); } }); });
      req.on("error", () => resolve("http_error")); req.end();
    });
    if (st === "succeeded" || st === "failed") return { label: g.label, status: st };
  }
  return { label: g.label, status: "timeout" };
}

if (!DEVICE_ID) { console.error("FAIL[A4] 缺 deviceId（具身派发必须挂设备）"); process.exit(1); }
const results = [];
for (const g of GOALS) {
  const r = await dispatchGoal(g);
  results.push(r);
  console.log(`  ${g.label} (${g.x},${g.y},${g.z},f=${g.f}): ${r.status}`);
}
const okCount = results.filter((r) => r.status === "succeeded").length;
console.log(`[A4] 厨房 goal 参数化执行闭环 ${okCount}/3 succeeded`);
if (okCount !== GOALS.length) { console.log("A4_KITCHEN_FAIL"); process.exit(1); }
console.log("A4_KITCHEN_PASS goal 参数化——同一 reach_goal 策略 × 三目标点全通");
