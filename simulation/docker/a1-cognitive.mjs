#!/usr/bin/env node
/** A1 认知循环在环——SSE 全链（循环四相完整 + 回复非空）。
 * 用法: node a1-cognitive.mjs [apiBase]  （复用 e2e golden 流程A 的契约形状）
 */
import http from "node:http";

const apiBase = process.argv[2] || "http://127.0.0.1:17801";
const TOKEN = "admin@space_dev";

function runStream(envelope) {
  return new Promise((resolve, reject) => {
    const u = new URL(apiBase + "/v1/agent/execute");
    const data = JSON.stringify(envelope);
    const req = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${TOKEN}`,
                 "content-length": Buffer.byteLength(data), accept: "text/event-stream" },
      timeout: 180000,
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

const { events, status } = await runStream({
  auth: { token: TOKEN },
  input: { text: "什么是具身智能" },
  context: { deviceState: {}, taskContext: { taskId: `sim-fs-a1-${Date.now()}` } },
  config: {},
});
if (status !== 200) { console.error(`FAIL[A1] http=${status}`); process.exit(1); }
const phases = events.filter((e) => e.phase).map((e) => e.phase);
const replyEvt = events.find((e) => e.payload?.reply || e.payload?.requestId);
const replyText = replyEvt?.payload?.reply?.text ?? "";
const endReason = events.map((e) => e.payload?.endReason).find(Boolean);
console.log(`[A1] 循环四相: ${phases.join(" → ")} endReason=${endReason ?? "(流内未含)"}`);
const ok = ["observe", "think", "decide", "act"].every((p) => phases.includes(p))
  && replyText.length > 0;
if (!ok) { console.error("FAIL[A1] 循环四相不完整或回复为空"); process.exit(1); }
console.log(`A1 PASS 回复="${replyText.slice(0, 40)}…"`);
