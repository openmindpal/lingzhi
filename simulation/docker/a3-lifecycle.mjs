#!/usr/bin/env node
/** A3 生命周期断言——ask_user 挂起→同会话续跑→checkpoint 注入→done；迭代硬帽 completed。
 * 契约锚点：
 *   - ask_user 终态持久化 resume:session:{taskId}（24h，一次性消费）
 *   - 同 sessionId 下一次 execute 自动装载 checkpointedTask → think 注入【会话未竟任务】
 *   - envelope.config.maxIterations 迭代硬顶 → endReason=completed（"达到迭代上限"）
 * 用法: node a3-lifecycle.mjs [apiBase]
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

function loopEnd(events) {
  const e = events.filter((e) => e.type === "loop.end" || e.payload?.meta?.endReason || e.payload?.endReason).pop();
  const p = e?.payload ?? {};
  return { endReason: p.meta?.endReason ?? p.endReason, reply: p.reply?.text ?? p.reply ?? "", raw: p };
}

let fail = 0;
const check = (name, cond, detail) => {
  console.log(`[${cond ? "PASS" : "FAIL"}] ${name}${detail ? " — " + detail : ""}`);
  if (!cond) fail++;
};

/* ── L1a ask_user 挂起 ── */
{
  const { events } = await runStream({
    auth: { token: TOKEN },
    input: { text: "生命周期问询：帮我决定该选什么颜色" },
    context: { deviceState: {}, taskContext: { taskId: "sim-lc-1" } },
    config: {},
  });
  const r = loopEnd(events);
  check("L1a ask_user 挂起", r.endReason === "ask_user",
    `endReason=${r.endReason} reply=${String(r.reply).slice(0, 50)}`);
}

/* ── L1b 同会话续跑（checkpoint 装载 + 答复→done）── */
{
  const { events } = await runStream({
    auth: { token: TOKEN },
    input: { text: "用户答复：选蓝色" },
    context: { deviceState: {}, taskContext: { taskId: "sim-lc-1" } },
    config: {},
  });
  const r = loopEnd(events);
  const reply = String(r.reply);
  check("L1b 续跑 done", r.endReason === "done", `endReason=${r.endReason}`);
  check("L1b checkpoint 未竟任务注入", reply.includes("checkpoint=true"),
    `reply=${reply.slice(0, 60)}`);
}

/* ── L2 迭代硬帽（maxIterations=1 + 永不 done 分支 → completed）── */
{
  const { events } = await runStream({
    auth: { token: TOKEN },
    input: { text: "生命周期预算：持续盘点工作区文件" },
    context: { deviceState: {}, taskContext: { taskId: "sim-lc-2" } },
    config: { maxIterations: 1 },
  });
  const r = loopEnd(events);
  check("L2 迭代硬帽 completed", r.endReason === "completed",
    `endReason=${r.endReason} msg=${String(r.raw.message ?? "").slice(0, 40)}`);
}

console.log(fail === 0 ? "LIFECYCLE_PASS" : `LIFECYCLE_FAIL(${fail})`);
process.exit(fail === 0 ? 0 : 1);
