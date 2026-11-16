#!/usr/bin/env node
/** [诊断] A4 认知模式——LLM 环内派发 device 词 */
import http from "node:http";
import fs from "node:fs";

const apiBase = process.argv[2] || "http://127.0.0.1:17801";
const DEVICE_ID = process.argv[3] || "";
const TOKEN = "admin@space_dev";

function runStream(envelope) {
  return new Promise((resolve, reject) => {
    const u = new URL(apiBase + "/v1/agent/execute");
    const data = JSON.stringify(envelope);
    const req = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${TOKEN}`,
                 "content-length": Buffer.byteLength(data), accept: "text/event-stream" },
      timeout: 300000,
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

const { events } = await runStream({
  auth: { token: TOKEN },
  input: { text: "厨房仿真：拿锅、倒油、翻炒，完成后汇报" },
  context: { deviceState: { deviceId: DEVICE_ID }, taskContext: { taskId: `sim-kc-diag-${Date.now()}` } },
  config: {},
});
fs.writeFileSync("simulation/reports/a4-events.json", JSON.stringify(events, null, 1).slice(0, 300000));
const endReason = events.map((e) => e.payload?.meta?.endReason ?? e.payload?.endReason).find(Boolean);
const replyEvt = events.find((e) => e.payload?.reply);
console.log("DIAG endReason=" + (endReason ?? "?") + " reply=" + String(replyEvt?.payload?.reply?.text ?? "").slice(0, 60));
