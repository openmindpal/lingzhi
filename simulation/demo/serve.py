"""演示画面服务——验证平台仪表盘的静态 + API 后端（零依赖，stdlib）。

路由：
  GET /                     → demo/index.html（仪表盘）
  GET /api/summary          → 最新报告汇总（run_all/goal_sweep/longevity/fault_injection 各取最新）
  GET /api/tasks            → 任务集（core_10）
  GET /cam                  → 桥接器相机帧代理（http://127.0.0.1:8284/frame.jpg——桥在即 200）
  GET /<path>               → simulation/ 根静态文件

用法: python -m demo.serve [--port 8089]   （工作目录 = simulation/）
"""
from __future__ import annotations

import argparse
import json
import tomllib
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
CAM_URL = "http://127.0.0.1:8284/frame.jpg"


def latest(prefix: str) -> dict | None:
    """reports/ 下 prefix 最新一份报告（文件名含时间戳，取字典序最大）。"""
    files = sorted(REPORTS.glob(f"{prefix}_*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:
        return None


def summary() -> dict:
    sweeps: dict[str, dict] = {}
    for f in sorted(REPORTS.glob("goal_sweep_*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            sweeps[r.get("package", "?")] = r          # 每包取最新（后写覆盖）
        except Exception:
            pass
    return {
        "run_all": latest("run_all"),
        "goal_sweeps": sweeps,
        "longevity": latest("longevity"),
        "fault_injection": latest("fault_injection"),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/demo", "/demo/"):
            body = (ROOT / "demo" / "index.html").read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
        elif path == "/api/summary":
            self._send(200, json.dumps(summary(), ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")
        elif path == "/api/tasks":
            self._send(200, (ROOT / "tasks" / "sets" / "core_10.json").read_bytes(),
                       "application/json; charset=utf-8")
        elif path == "/cam":
            try:
                with urllib.request.urlopen(CAM_URL, timeout=3) as r:
                    self._send(200, r.read(), "image/jpeg")
            except Exception as e:
                self._send(503, str(e).encode("utf-8"), "text/plain; charset=utf-8")
        elif path == "/healthz":
            self._send(200, b"ok", "text/plain")
        else:
            f = (ROOT / path.lstrip("/")).resolve()
            if not str(f).startswith(str(ROOT)) or not f.is_file():
                self._send(404, b"not found", "text/plain")
                return
            ctype = "application/json" if f.suffix == ".json" else (
                "text/yaml" if f.suffix == ".yaml" else "text/plain; charset=utf-8")
            self._send(200, f.read_bytes(), ctype)

    def log_message(self, *a):  # 静默
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8089)
    args = ap.parse_args()
    print(f"[demo] http://127.0.0.1:{args.port}/  （root={ROOT}）")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
