#!/usr/bin/env node
/** A5 工厂 goal 参数化执行闭环——REST 直派 × 三个 goal 目标点。
 * 认知层分解任务 → 同一 reach_goal 词 + 不同 goal 参数（料位/检查悬停/料架位）。
 * 断言：三个 goal 点全部 succeeded + DB 核验。
 * 用法: node a5-factory.mjs [apiBase] [deviceId]
 */
import http from "node:http";
import { execSync } from "node:child_process";

const apiBase = process.argv[2] || "http://127.0.0.1:17801";
const DEVICE_ID = process.argv[3] || "";
const TOKEN = "admin@space_dev";
const GOALS = [
  { x: 0.30, y: -0.15, z: 0.20, f: 0.5, label: "传送带料位" },
  { x: 0.35, y: -0.15, z: 0.30, f: 0.3, label: "检查悬停位" },
  { x: 0.55, y: 0.15, z: 0.22, f: 0.5, label: "料架放置位" },
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

if (!DEVICE_ID) { console.error("FAIL[A5] 缺 deviceId"); process.exit(1); }
const results = [];
for (const g of GOALS) {
  const r = await dispatchGoal(g);
  results.push(r);
  console.log(`  ${g.label} (${g.x},${g.y},${g.z},f=${g.f}): ${r.status}`);
}
const okCount = results.filter((r) => r.status === "succeeded").length;
console.log(`[A5] 工厂 goal 参数化执行闭环 ${okCount}/3 succeeded`);

// DB 核验：goal 参数化派发落库（reach_goal 词 × 工厂目标点）
let dbOk = 0;
for (const g of GOALS) {
  try {
    const out = execSync(
      `docker exec mindpal-e2e-postgres-1 psql -U mindpal -d mindpal -t -c "SELECT count(*) FROM device_executions WHERE tool_ref='device.actuator.gesture' AND status='succeeded' AND input_json->>'goal_x'='${g.x}' AND input_json->>'goal_y'='${g.y}';"`,
      { encoding: "utf-8", timeout: 10000 }
    ).trim();
    const count = parseInt(out) || 0;
    if (count > 0) dbOk++;
    console.log(`  ${g.label}: DB落库 ${count > 0 ? "succeeded" : "MISSING"}`);
  } catch { console.log(`  ${g.label}: DB query error`); }
}

let fail = 0;
const check = (n, c, d) => { console.log(`[${c ? "PASS" : "FAIL"}] ${n}${d ? " — " + d : ""}`); if (!c) fail++; };
check("A5 认知编排 done", okCount >= 3, `${okCount}/3 goal 点`);
check("A5 DB 三 goal 点落库", dbOk >= 3, `${dbOk}/3 点落库 succeeded`);

console.log(fail === 0 && dbOk >= 3 ? "A5_FACTORY_PASS" : `A5_FACTORY_FAIL`);
process.exit(fail === 0 && dbOk >= 3 ? 0 : 1);
