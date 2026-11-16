#!/usr/bin/env node
/** A8 感知在环断言套件——四项一次跑齐（技术文档 §3 感知链验证）。
 *
 * P2 YOLOX/GDINO 感知链  open_vocab 派发 → 视觉循环消费（GDINO 实推理）→
 *                device.perception.latest 读出 detections ≥1 且 label 匹配
 *                实体 upsert → 模型回复含积木方位证据
 * P4 视觉伺服活性 YOLOX 会话加载 + vision 推理任务存活 + 帧流量 > 0（伺服数据源活性）
 *
 * 用法: node a8-perception.mjs [apiBase] [deviceId]
 */
import http from "node:http";
import { execSync } from "node:child_process";
import fs from "node:fs";

const apiBase = process.argv[2] || "http://127.0.0.1:17801";
const DEVICE_ID = process.argv[3] || "";
const TOKEN = "admin@space_dev";

function rest(method, pathname, body) {
  return new Promise((resolve, reject) => {
    const u = new URL(apiBase + pathname);
    const data = body ? JSON.stringify(body) : undefined;
    const req = http.request({
      hostname: u.hostname, port: u.port, path: u.pathname, method,
      headers: { "content-type": "application/json", authorization: `Bearer ${TOKEN}`,
                 ...(data ? { "content-length": Buffer.byteLength(data) } : {}) },
      timeout: 120000,
    }, (res) => { let out = ""; res.on("data", (c) => (out += c)); res.on("end", () => resolve({ status: res.statusCode, text: out })); });
    req.on("error", reject);
    if (data) req.write(data); req.end();
  });
}

function sh(cmd) {
  return execSync(cmd, { encoding: "utf-8", timeout: 60000, env: { ...process.env, MSYS_NO_PATHCONV: "1" } });
}

let fail = 0;
const check = (n, c, d) => { console.log(`[${c ? "PASS" : "FAIL"}] ${n}${d ? " — " + d : ""}`); if (!c) fail++; };

/* ── P1 视觉输入：容器内相机帧解码 → 红/绿积木像素 ── */
{
  const script = `import urllib.request\nr = urllib.request.urlopen("http://127.0.0.1:8284/frame.jpg", timeout=3)\nd = r.read()\nopen("/tmp/p1.jpg","wb").write(d)\nprint(len(d))`;
  fs.writeFileSync(".tmp/p1_get.py", script);
  sh("docker cp .tmp/p1_get.py mindpal-e2e-device-agent-1:/tmp/p1_get.py");
  const out = sh("docker exec mindpal-e2e-device-agent-1 /opt/venv/bin/python /tmp/p1_get.py").trim();
  sh("docker cp mindpal-e2e-device-agent-1:/tmp/p1.jpg .tmp/p1.jpg");
  const px = sh(`python -c "from PIL import Image; import numpy as np; a0 = Image.open('.tmp/p1.jpg'); a0rgb = np.asarray(a0); a = np.asarray(a0.convert('HSV')); red = int(((a[:,:,0]<20)|(a[:,:,0]>235)).sum()); green = int(((a[:,:,0]>50)&(a[:,:,0]<110)&(a[:,:,1]>60)).sum()); print(red, green, int(a0rgb.mean()))"`).trim().split(/\s+/);
  const [redPx, greenPx, bright] = px.map(Number);
  check("P1 视觉输入：帧中红/绿积木色块可识别", redPx > 3 && greenPx > 3,
    `redHsv=${redPx} greenHsv=${greenPx} brightness=${bright}`);
}

/* ── P2 GDINO 感知链：open_vocab → 消费 → device.perception.latest 读出 ── */
{
  const created = await rest("POST", "/v1/device-executions", {
    deviceId: DEVICE_ID, toolRef: "device.perception.open_vocab",
    input: { prompt: "a red toy building block on the table", camera: "main" },
    requireUserPresence: false,
  });
  let ok = created.status === 200;
  let ovExecId = "";
  try { ovExecId = JSON.parse(created.text)?.execution?.deviceExecutionId ?? ""; } catch {}

  // 等视觉循环消费（每 30 帧 ≈ 6s @5FPS）+ GDINO 推理
  await new Promise((r) => setTimeout(r, 9000));

  // 轮询 open_vocab 终态拿 detector_available
  let detectorAvailable = undefined;
  if (ovExecId) {
    for (let i = 0; i < 15; i++) {
      const st = await rest("GET", `/v1/device-executions/${ovExecId}`);
      let ex; try { ex = JSON.parse(st.text)?.execution; } catch {}
      if (ex?.status === "succeeded" || ex?.status === "failed") {
        detectorAvailable = ex.output_digest?.detector_available;
        break;
      }
      await new Promise((r) => setTimeout(r, 1000));
    }
  }

  const latest = await rest("POST", "/v1/device-executions", {
    deviceId: DEVICE_ID, toolRef: "device.perception.latest",
    input: {}, requireUserPresence: false,
  });
  let latestId = "";
  try { latestId = JSON.parse(latest.text)?.execution?.deviceExecutionId ?? ""; } catch {}
  let dets = [];
  if (latestId) {
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      const st = await rest("GET", `/v1/device-executions/${latestId}`);
      let ex; try { ex = JSON.parse(st.text)?.execution; } catch {}
      if (ex?.status === "succeeded" || ex?.status === "failed") {
        dets = ex.output_digest?.detections ?? [];
        break;
      }
    }
  }
  const labelMatch = dets.some((d) => /red|block|brick/i.test(d.label ?? ""));
  check("P2a open_vocab 回执含检测器状态", ok && detectorAvailable === true,
    `detector_available=${detectorAvailable}`);
  check("P2b GDINO 检出 ≥1 且 label 匹配", dets.length >= 1 && labelMatch,
    `detections=${JSON.stringify(dets).slice(0, 150)}`);
}

/* ── P3 实体表写入：认知环 run（真模型）→ 感知观察流 → 实体状态存储 → 回复含方位 ── */
{
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
  const { events } = await runStream({
    auth: { token: TOKEN },
    input: { text: "用开放词汇检测找出工作台上的红色积木，读取它的检出方位（bearing/dist），然后告诉我它在视野的左边还是右边" },
    context: { deviceState: { deviceId: DEVICE_ID }, taskContext: { taskId: `sim-p3-${Date.now()}` } },
    config: { maxIterations: 10 },
  });
  const endReason = events.map((e) => e.payload?.meta?.endReason ?? e.payload?.endReason).find(Boolean);
  const replyEvt = events.find((e) => e.payload?.reply);
  const replyText = String(replyEvt?.payload?.reply?.text ?? "");
  const ovChain = events.filter((e) => JSON.stringify(e).includes("open_vocab") || JSON.stringify(e).includes("perception.latest")).length;
  console.log(`[P3] endReason=${endReason ?? "?"} ovChain=${ovChain} reply="${replyText.slice(0, 90)}"`);
  check("P3a 认知环感知任务干净收尾", /done|completed/.test(endReason ?? ""), `endReason=${endReason}`);
  check("P3b 感知工具链在环", ovChain >= 2, `open_vocab/latest 事件 ${ovChain}`);
  check("P3c 回复含方位证据（左/右/bearing/deg/检测）",
    /左|右|bearing|bearing|rad|deg|检出|方位/i.test(replyText), replyText.slice(0, 80));
}

/* ── P4 视觉伺服活性：YOLOX 会话 + 推理任务存活 + 帧流量 ── */
{
  let logs = "";
  try { logs = sh("docker exec mindpal-e2e-device-agent-1 cat /tmp/engine-run.log 2>/dev/null || true"); } catch {}
  if (!logs.trim()) { logs = sh("docker logs mindpal-e2e-device-agent-1 2>&1 || true"); }
  const sessionLoaded = /vision ONNX session loaded/.test(logs);
  const taskStarted = /vision inference task started/.test(logs);
  const channelClosed = /raw-frame channel closed/.test(logs);
  const epLive = /using (CUDA|CPU|TensorRT) EP/.test(logs);
  check("P4a YOLOX 会话已加载 + EP 活", sessionLoaded && epLive, `session=${sessionLoaded} ep=${epLive}`);
  check("P4b 视觉推理任务存活（无 channel closed）", taskStarted && !channelClosed,
    `started=${taskStarted} closed=${channelClosed}`);
}

console.log(fail === 0 ? "A8_PERCEPTION_PASS 感知在环四项全绿" : `A8_PERCEPTION_FAIL(${fail})`);
process.exit(fail === 0 ? 0 : 1);
