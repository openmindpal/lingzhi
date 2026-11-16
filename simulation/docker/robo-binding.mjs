#!/usr/bin/env node
/**
 * robo-binding.mjs——感知绑定在环矩阵（RoboDojo 型任务集·几何绑定维度）。
 *
 * 被测链 = 真渲染深度（MuJoCo 桥 http_depth 供数）→ 引擎抓取规划（去投影点云
 * → 接触模型候选 → 评分排序）→ 位姿目标。判定 = 机械判据（非"看起来对"）：
 *   HIT     最佳抓取的全部接触点落在目标盒 AABB+10mm 带内（基座系，GT 由桥真值路由给出）
 *   SELECT  clutter 集：最佳抓取质心距目标盒 < 距干扰盒（择优不抓错对象）
 * 扰动矩阵（Robo/Robo+/Dojo 三级泛化同构）：
 *   L1 基线 → L2 位置(6)/尺寸(4)/干扰(1) → L3 组合(3)——15 回合。
 * 产出：simulation/reports/robo_binding_<ts>.json（逐回合 + 分维度成功率）；
 * 任一回合一未得裁决（派发链断/回执缺格）= 退出码 1（裁决完整性硬门）。
 * 用法: node simulation/docker/robo-binding.mjs [apiBase] [deviceId]
 */
import http from "node:http";
import fs from "node:fs";

const API = process.argv[2] || "http://127.0.0.1:17801";
const RESIDENT = { host: "127.0.0.1", port: 8299 };
const DEVICE_ID = process.argv[3] || "";
const TOKEN = "admin@space_dev";
const W = 320, H = 240, FX = 400, FY = 400, CX = 160, CY = 120;
const AABB_MARGIN_M = 0.01;   // 接触点容差带（单位测试同源 + 量化余量）
const ROI_MARGIN_PX = 40;     // GT 投影 bbox 外扩（检测框语义：顶面+迎面双侧）

function req(method, hostname, port, path, body) {
  return new Promise((resolve, reject) => {
    const data = body === undefined ? null : JSON.stringify(body);
    const headers = { "content-type": "application/json", authorization: `Bearer ${TOKEN}` };
    if (data) headers["content-length"] = Buffer.byteLength(data);
    const r = http.request({ hostname, port, path, method, headers, timeout: 30000 }, (res) => {
      let out = "";
      res.on("data", (c) => (out += c));
      res.on("end", () => resolve({ status: res.statusCode, text: out }));
    });
    r.on("error", reject);
    r.on("timeout", () => r.destroy(new Error("timeout")));
    if (data) r.write(data);
    r.end();
  });
}

async function setVariant(ep) {
  const r = await req("POST", RESIDENT.host, RESIDENT.port, "/variant", {
    dx: ep.dx ?? 0, dy: ep.dy ?? 0, half: ep.half ?? 0.05, clutter: !!ep.clutter,
  });
  if (r.status !== 200) throw new Error(`variant 失败 ${r.status} ${r.text.slice(0, 120)}`);
  return JSON.parse(r.text);
}

async function getGT() {
  const r = await req("GET", RESIDENT.host, RESIDENT.port, "/gt.json");
  if (r.status !== 200) throw new Error(`gt 失败 ${r.status}`);
  return JSON.parse(r.text);
}

async function dispatchPlan(roi) {
  const created = await req("POST", new URL(API).hostname, new URL(API).port,
    "/v1/device-executions", {
      deviceId: DEVICE_ID, toolRef: "device.grasp.plan",
      input: { roi }, requireUserPresence: false,
    });
  let body = null;
  try { body = JSON.parse(created.text); } catch {}
  const execId = body?.execution?.deviceExecutionId;
  if (created.status !== 200 || !execId) throw new Error(`派发失败 ${created.status} ${created.text.slice(0, 160)}`);
  for (let i = 0; i < 60; i++) {
    await new Promise((r2) => setTimeout(r2, 1000));
    const rr = await req("GET", new URL(API).hostname, new URL(API).port, `/v1/device-executions/${execId}`);
    let b2 = null;
    try { b2 = JSON.parse(rr.text); } catch {}
    const st = b2?.execution?.status;
    if (st === "succeeded" || st === "failed") return b2.execution;
  }
  throw new Error("60s 未到终态");
}

function projectROI(center, half) {
  // 目标盒 8 角投影 → bbox 外扩（针孔 y 向下，与引擎 deproject 同源）
  let u0 = W, v0 = H, u1 = 0, v1 = 0, anyFront = false;
  for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
    const p = [center[0] + sx * half, center[1] + sy * half, center[2] + sz * half];
    if (p[2] <= 0.03) continue;
    anyFront = true;
    const u = CX + (FX * p[0]) / p[2], v = CY + (FY * p[1]) / p[2];
    u0 = Math.min(u0, u); u1 = Math.max(u1, u);
    v0 = Math.min(v0, v); v1 = Math.max(v1, v);
  }
  if (!anyFront) throw new Error("GT 盒整体在相机背后——场景/声明失配");
  u0 = Math.max(0, Math.floor(u0 - ROI_MARGIN_PX));
  v0 = Math.max(0, Math.floor(v0 - ROI_MARGIN_PX));
  u1 = Math.min(W - 1, Math.ceil(u1 + ROI_MARGIN_PX));
  v1 = Math.min(H - 1, Math.ceil(v1 + ROI_MARGIN_PX));
  return [u0, v0, u1, v1];
}

function centroid(pts) {
  const c = [0, 0, 0];
  for (const p of pts) for (let i = 0; i < 3; i++) c[i] += p[i];
  return c.map((v) => v / pts.length);
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function judge(exec, gt, ep) {
  const digest = exec.outputDigest ?? {};
  const grasps = digest.grasps ?? [];
  if (exec.status !== "succeeded" || !Array.isArray(grasps) || grasps.length === 0) {
    return { verdict: "MISS", reason: exec.status === "succeeded" ? "zero_grasps" : `plan_${exec.status}`,
             digest_hint: JSON.stringify(digest).slice(0, 200) };
  }
  const best = grasps[0];
  const cs = best?.closing?.MultiFinger?.contacts ?? best?.closing?.Jaw?.contacts;
  if (!Array.isArray(cs) || cs.length < 3) {
    return { verdict: "MISS", reason: "malformed_contacts", digest_hint: JSON.stringify(best).slice(0, 200) };
  }
  // 帧契约：引擎输出 = 声明 mount 系（本验证声明 mount=世界位姿 → 输出=世界系）
  // ——GT 判据同帧取 pos_world（pos_engine_base 保留作帧自检，不参与判定）
  const center = gt.target.pos_world;
  const half = gt.target.half_m;
  const cen = centroid(cs);
  const inBand = cs.every((p) => p.every((v, i) => Math.abs(v - center[i]) <= half + AABB_MARGIN_M));
  const hitTarget = inBand;
  const nearest = ep.clutter
    ? dist(cen, center) < dist(cen, gt.distractor.pos_world)
    : true;
  if (hitTarget && nearest) {
    return { verdict: "HIT", centroid: cen, contacts: cs.length,
             d_target: +dist(cen, center).toFixed(4) };
  }
  return { verdict: "MISS", centroid: cen, contacts: cs.length,
           reason: inBand ? "picked_distractor" : "off_object",
           d_target: +dist(cen, center).toFixed(4),
           d_distractor: ep.clutter ? +dist(cen, gt.distractor.pos_world).toFixed(4) : undefined };
}

const EPISODES = [
  { dim: "L1_robo", name: "baseline" },
  { dim: "L2_position", name: "dx-0.10", dx: -0.10 },
  { dim: "L2_position", name: "dx-0.05", dx: -0.05 },
  { dim: "L2_position", name: "dx+0.05", dx: 0.05 },
  { dim: "L2_position", name: "dx+0.10", dx: 0.10 },
  { dim: "L2_position", name: "dy-0.08", dy: -0.08 },
  { dim: "L2_position", name: "dy+0.08", dy: 0.08 },
  { dim: "L2_size", name: "half0.03", half: 0.03 },
  { dim: "L2_size", name: "half0.04", half: 0.04 },
  { dim: "L2_size", name: "half0.06", half: 0.06 },
  { dim: "L2_size", name: "half0.075", half: 0.075 },
  { dim: "L2_clutter", name: "clutter_on", clutter: true },
  { dim: "L3_dojo", name: "pos+size", dx: 0.10, half: 0.06 },
  { dim: "L3_dojo", name: "pos2+size2", dx: -0.08, dy: 0.08, half: 0.04 },
  { dim: "L3_dojo", name: "pos+size+clutter", dx: 0.05, dy: -0.08, half: 0.06, clutter: true },
];

let harnessOk = true;
const results = [];
for (const ep of EPISODES) {
  let row = { dim: ep.dim, name: ep.name };
  try {
    await setVariant(ep);
    await new Promise((r) => setTimeout(r, 1200));   // 渲染线程刷新深度缓存
    const gt = await getGT();
    row.gt = gt.target;
    const roi = projectROI(gt.target.pos_engine_base, gt.target.half_m);
    row.roi = roi;
    const exec = await dispatchPlan(roi);
    const j = judge(exec, gt, ep);
    row = { ...row, ...j };
  } catch (e) {
    row = { ...row, verdict: "ERROR", reason: String(e.message ?? e).slice(0, 200) };
    harnessOk = false;
  }
  results.push(row);
  console.log(`[${row.verdict}] ${ep.dim}/${ep.name}` +
    (row.reason ? ` (${row.reason})` : "") +
    (row.centroid ? ` 质心=[${row.centroid.map((v) => v.toFixed(3)).join(",")}] d=${row.d_target}` : ""));
}

// ── 分维度成功率 + 总率 ──
const byDim = {};
for (const r of results) {
  (byDim[r.dim] ??= { n: 0, hit: 0 });
  byDim[r.dim].n++;
  if (r.verdict === "HIT") byDim[r.dim].hit++;
}
const hitN = results.filter((r) => r.verdict === "HIT").length;
console.log("=== 分维度成功率 ===");
for (const [dim, s] of Object.entries(byDim)) {
  console.log(`${dim}: ${s.hit}/${s.n} = ${((s.hit / s.n) * 100).toFixed(0)}%`);
}
console.log(`TOTAL: ${hitN}/${results.length} = ${((hitN / results.length) * 100).toFixed(0)}%`);

const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const report = { suite: "robo_binding", episodes: results, byDim, hitN, total: results.length };
fs.writeFileSync(`simulation/reports/robo_binding_${stamp}.json`, JSON.stringify(report, null, 2));

if (!harnessOk) {
  console.error("ROBO_BINDING_HARNESS_FAIL——存在未得裁决回合（派发链/回执缺陷）");
  process.exit(1);
}
console.log("ROBO_BINDING_HARNESS_PASS——15 回合全部裁决（成功率见上，实测值如实入册）");
