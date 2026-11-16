"""仿真相机 HTTP 通路——runtime http_image/http_depth 后端的供帧端（零runtime改动相机路线）。

runtime acquisition 声明 kind="http_image" → GET device URL → JPEG 提取（SOI/EOI 扫描）
→ ImageFrame；kind="http_depth" → GET device URL → 原始 Z16（LE u16 毫米，行主序）。
本模块 = 两个 GET 的服务端。

线程纪律（GL 上下文析构竞争根修）：渲染器在**专用渲染线程**内创建/使用/关闭——
生命周期全程单线程独占；HTTP 线程只回缓存的最新帧。渲染缺席（无 GL/无 PIL/
无相机）= 503 honest缺帧（runtime侧感知事实 TTL 语义接管）。

深度契约（与引擎 http_depth 后端逐位对齐）：
  /depth.raw = width×height×2 字节 LE u16，毫米 **z-depth**（针孔平面距离），
  0 = 无回波背景。MuJoCo 深度渲染 = 沿视线距离（ray distance）——逐像素按
  z = d / sqrt(1+((u-cx)/fx)² + ((v-cy)/fy)²) 换算为 z-depth（内参 = 包声明，
  与引擎 deproject_points 的针孔模型同源）。
"""
from __future__ import annotations

import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import mujoco
import numpy as np

# 无回波判定阈（米）——超过 = 背景无几何，按引擎契约记 0（去投影跳过）
NO_RETURN_M = 6.0


class FrameServer:
    """渲染线程 @5Hz 产帧 → /frame.jpg 回最新 JPEG；depth=True 加供 /depth.raw Z16。"""

    def __init__(self, model, cam_name: str, width: int, height: int,
                 host: str = "127.0.0.1", port: int = 8284, fps: float = 5.0,
                 depth: bool = False, intr: dict | None = None):
        self.model = model
        self.cam_name = cam_name
        self.size = (width, height)
        self.interval = 1.0 / max(fps, 0.1)
        self.depth = depth
        self.intr = intr or {}
        self._latest: bytes | None = None
        self._latest_depth: bytes | None = None
        self._stop = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/depth.raw":
                    if not outer.depth or outer._latest_depth is None:
                        self.send_response(outer.depth and 503 or 404)
                        self.send_header("Content-Type", "text/plain")
                        self.end_headers()
                        self.wfile.write(b"depth renderer unavailable")
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(outer._latest_depth)))
                    self.end_headers()
                    self.wfile.write(outer._latest_depth)
                    return
                if outer._latest is None:
                    self.send_response(503)
                    self.send_header("Content-Type", "text/plain")
                    self.end_headers()
                    self.wfile.write(b"renderer unavailable (headless/GL/PIL)")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(outer._latest)))
                self.end_headers()
                self.wfile.write(outer._latest)

            def log_message(self, *a):  # 静默访问日志
                pass

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.http_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)

    def _render_depth(self, renderer) -> bytes:
        """MuJoCo ray-distance → z-depth(毫米 LE u16)，0 = 无回波。单线程内调用。"""
        w, h = self.size
        fx = float(self.intr.get("fx", 400.0))
        fy = float(self.intr.get("fy", 400.0))
        cx = float(self.intr.get("cx", w / 2.0))
        cy = float(self.intr.get("cy", h / 2.0))
        renderer.enable_depth_rendering()
        try:
            d = renderer.render()  # float64 (h, w) = 沿视线距离（米）
        finally:
            renderer.disable_depth_rendering()
        u = (np.arange(w, dtype=np.float64) - cx) / fx
        v = (np.arange(h, dtype=np.float64) - cy) / fy
        norm = np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)
        z = d / norm                                   # z-depth（米）
        mm = np.clip(z * 1000.0, 0, 65535).astype("<u2")
        mm[d >= NO_RETURN_M] = 0                       # 无回波 → 0（引擎契约）
        return mm.tobytes()

    def _render_loop(self) -> None:
        """独占 GL 的渲染循环：创建→产帧→关闭全在本线程（跨线程 = exit 段错误根因）。"""
        from PIL import Image
        cam_id = mujoco.mj_name2id(self.model.model, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        if cam_id < 0:
            return
        renderer = mujoco.Renderer(self.model.model, height=self.size[1], width=self.size[0])
        try:
            while not self._stop.is_set():
                if self.depth:
                    self._latest_depth = self._render_depth(renderer)
                renderer.update_scene(self.model.data, camera=cam_id)
                rgb = renderer.render()
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG")
                self._latest = buf.getvalue()
                self._stop.wait(self.interval)
        finally:
            renderer.close()   # 与创建同线程——GL 上下文纪律

    def start(self) -> None:
        self.http_thread.start()
        self.render_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.render_thread.join(timeout=2.0)
        self.httpd.shutdown()
