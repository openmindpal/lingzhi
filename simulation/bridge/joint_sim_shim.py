#!/usr/bin/env python3
"""joint_sim 兼容 shim——MuJoCo 物理背板 + resident 常驻模式（runtime tri-family 测试直接可用）。

CLI 契约 = packages/ddswrapper/test/joint_sim.c 逐参兼容：
  joint_sim <domain> <cmd_topic> <state_topic> [hz]
      [--stop-after ms] [--freeze-after ms dur]
      [--perturb-after ms slot rad] [--ft-after ms mag]

双模式：
  resident（容器内默认）：
    stub 被测试 spawn 时把 argv POST 给常驻身体（127.0.0.1:8299/spawn）后即退出——
    测试 kill(stub) 不影响身体流（C 版 joint_sim 毫秒级启动的隐含假设：
    杀旧→起新间隙 < watchdog timeout 200ms；python 冷启动 1.5s 必然断流——
    resident 消解该间隙：流跨回合连续，spawn 仅重置回合旗标）。
    启动：python -m bridge.joint_sim_shim --resident
  standalone（调试）：无 resident 在场时回退为本进程闭环（冷启动语义如实保留）。

语义对照（C 版 → 本 shim）：目标跟随 = MuJoCo mj_step + 包 profile PD；
perturb = qpos 瞬时踢；freeze = 停发；ft-after = ft[0][2]=ft[1][2]=mag（双腕路序）；
n=len(mapping)、n_ft=2（fixture 双腕声明）、imu = MuJoCo 真传感。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import tomllib
import urllib.request

from cyclonedds.core import Qos, Policy
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic

from .mj_model import MjModel, substep_ratio
from .server import BEST_EFFORT, SIM_HZ
from .wire_types import JointStates, JointTargets

PKG_DEFAULT = "/build/apps/device-agent-rs/fixtures/tri-family/body-package.toml"
MODEL_DEFAULT = "/build/simulation/assets/models/biped_29dof.xml"
FT_PATHS = 2          # 双腕路数（fixture reach_contact = 97+6·2）
RESIDENT_URL = "http://127.0.0.1:8299/spawn"


def parse_args(argv: list[str]) -> dict:
    """joint_sim CLI → 配置 dict（stub 与 standalone 共用）。"""
    cfg = {"domain": int(argv[1]), "cmd": argv[2], "state": argv[3], "hz": 50,
           "stop_after_ms": -1, "freeze_at_ms": -1, "freeze_dur_ms": 0,
           "perturb_at_ms": -1, "perturb_slot": -1, "perturb_mag": 0.0,
           "ft_at_ms": -1, "ft_mag": 0.0}
    hz_raw = argv[4] if len(argv) > 4 else None
    i = 5 if (hz_raw and not hz_raw.startswith("--")) else 4
    if hz_raw and not hz_raw.startswith("--"):
        cfg["hz"] = max(1, int(hz_raw))
    while i < len(argv):
        a = argv[i]
        if a == "--stop-after":
            cfg["stop_after_ms"] = int(argv[i + 1]); i += 2
        elif a == "--freeze-after":
            cfg["freeze_at_ms"] = int(argv[i + 1]); cfg["freeze_dur_ms"] = int(argv[i + 2]); i += 3
        elif a == "--perturb-after":
            cfg["perturb_at_ms"] = int(argv[i + 1]); cfg["perturb_slot"] = int(argv[i + 2])
            cfg["perturb_mag"] = float(argv[i + 3]); i += 4
        elif a == "--ft-after":
            cfg["ft_at_ms"] = int(argv[i + 1]); cfg["ft_mag"] = float(argv[i + 2]); i += 3
        else:
            i += 1
    return cfg


class ResidentBody:
    """常驻身体：流跨回合连续；/spawn 仅重置回合旗标与 DDS 端点。"""

    def __init__(self):
        self.lock = threading.Lock()
        self.cfg: dict | None = None
        self.t0 = time.monotonic()
        self.perturbed = False
        self.tick = 0
        self.model: MjModel | None = None
        self.channels = None
        self.targets: list[float] = []
        self.reader = None
        self.writer = None
        self.active_topics = None
        self.participants: dict[int, DomainParticipant] = {}
        self.depth_mount_xyz = [0.35, 0.0, 1.2]
        self.depth_mount_rpy = [0.0, 55.0, 0.0]

    # ── 感知绑定验证面：场景变体与真值路由（resident HTTP 暴露——judge 用）──
    SCENE_TABLE_TOP_Z = 0.75
    SCENE_TARGET_BASE = [0.665, 0.0]

    def set_variant(self, p: dict) -> dict:
        """绑定验证变体：目标盒位姿/尺寸 + 干扰盒在/离场（静态体——渲染即真值）。"""
        import mujoco
        with self.lock:
            if self.model is None:
                raise RuntimeError("model 未就绪（spawn 未发生）")
            half = float(p.get("half", 0.05))
            dx = float(p.get("dx", 0.0))
            dy = float(p.get("dy", 0.0))
            m = self.model
            tb = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
            tg = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
            db = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_BODY, "distractor_box")
            dg = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_GEOM, "distractor_geom")
            m.model.body_pos[tb] = [self.SCENE_TARGET_BASE[0] + dx, self.SCENE_TARGET_BASE[1] + dy,
                                    self.SCENE_TABLE_TOP_Z + half]
            m.model.geom_size[tg] = [half, half, half]
            if p.get("clutter"):
                m.model.body_pos[db] = [self.SCENE_TARGET_BASE[0] + dx + 0.16,
                                        self.SCENE_TARGET_BASE[1] + dy + 0.12,
                                        self.SCENE_TABLE_TOP_Z + 0.05]
            else:
                m.model.body_pos[db] = [3.0, 3.0, 0.3]   # 移出视场角
            mujoco.mj_forward(m.model, m.data)
            return {"target": list(m.model.body_pos[tb]), "half": list(m.model.geom_size[tg]),
                    "clutter": bool(p.get("clutter"))}

    def get_gt(self) -> dict:
        """真值读数：世界系位姿/尺寸 + 引擎基座系换算（权威 = engine_frames）。"""
        import mujoco
        from .engine_frames import world_to_engine_base
        with self.lock:
            if self.model is None:
                raise RuntimeError("model 未就绪（spawn 未发生）")
            m = self.model
            out: dict = {}
            tb = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
            tg = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
            db = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_BODY, "distractor_box")
            pos_w = [float(v) for v in m.model.body_pos[tb]]
            half = float(m.model.geom_size[tg][0])
            out["target"] = {"pos_world": pos_w, "half_m": half,
                             "pos_engine_base": [float(v) for v in world_to_engine_base(
                                 pos_w, self.depth_mount_xyz, self.depth_mount_rpy)]}
            pos_d = [float(v) for v in m.model.body_pos[db]]
            out["distractor"] = {"pos_world": pos_d,
                                 "pos_engine_base": [float(v) for v in world_to_engine_base(
                                     pos_d, self.depth_mount_xyz, self.depth_mount_rpy)]}
            return out

    def spawn(self, argv: list[str]) -> None:
        cfg = parse_args(argv)
        print(f"joint_sim(resident): SPAWN argv={argv} -> ft_at={cfg['ft_at_ms']} mag={cfg['ft_mag']} stop={cfg['stop_after_ms']}", file=sys.stderr, flush=True)
        pkg_path = os.environ.get("MINDPAL_SIM_PKG", PKG_DEFAULT)
        model_path = os.environ.get("MINDPAL_SIM_MODEL", MODEL_DEFAULT)
        with self.lock:
            self.cfg = cfg
            self.t0 = time.monotonic()
            self.perturbed = False
            self.tick = 0
            if self.model is None:
                ccfg = tomllib.loads(open(pkg_path, encoding="utf-8").read())
                self.mapping = ccfg["protocol"]["joint_mapping"]["ordered"]
                self.kp = float(ccfg["profile"]["dynamics"]["kp"])
                self.kd = float(ccfg["profile"]["dynamics"]["kd"])
                self.model = MjModel(model_path)
                self.channels = self.model.channels(self.mapping)
                self.targets = [0.0] * len(self.mapping)
                self.model.reset()
            self._ensure_dds(cfg["domain"], cfg["cmd"], cfg["state"], cfg["hz"])
            print(f"joint_sim(resident): domain={cfg['domain']} cmd={cfg['cmd']} "
                  f"state={cfg['state']} hz={cfg['hz']}", file=sys.stderr, flush=True)

    def _ensure_dds(self, domain: int, cmd: str, state: str, hz: int) -> None:
        key = (domain, cmd, state)
        if self.active_topics == key:
            return
        dp = self.participants.get(domain)
        if dp is None:
            dp = DomainParticipant(domain)
            self.participants[domain] = dp
        self.reader = DataReader(dp, Topic(dp, cmd, JointTargets), qos=BEST_EFFORT)
        self.writer = DataWriter(dp, Topic(dp, state, JointStates), qos=BEST_EFFORT)
        self.active_topics = key
        self.n_sub = substep_ratio(SIM_HZ, float(hz), self.model.timestep)
        self.period = 1.0 / hz

    def run_forever(self) -> None:
        import mujoco
        while True:
            time.sleep(0.05)                     # 等首次 spawn
            while True:
                with self.lock:
                    cfg = self.cfg
                    if cfg is None:
                        break
                    el = (time.monotonic() - self.t0) * 1000.0
                    frozen = cfg["freeze_at_ms"] >= 0 and \
                        cfg["freeze_at_ms"] <= el < cfg["freeze_at_ms"] + cfg["freeze_dur_ms"]
                    if cfg["stop_after_ms"] >= 0 and el >= cfg["stop_after_ms"]:
                        self.cfg = None          # 停发（standalone 语义 = 退出进程）
                        break
                    if cfg["perturb_at_ms"] >= 0 and el >= cfg["perturb_at_ms"] and not self.perturbed:
                        self.perturbed = True
                        if 0 <= cfg["perturb_slot"] < len(self.mapping):
                            qa, _ = self.channels[cfg["perturb_slot"]]
                            self.model.data.qpos[qa] += cfg["perturb_mag"]
                            mujoco.mj_forward(self.model.model, self.model.data)
                            print(f"joint_sim: PERTURB slot={cfg['perturb_slot']} "
                                  f"mag={cfg['perturb_mag']:.3f}", file=sys.stderr, flush=True)
                    for s in self.reader.take():
                        n = min(int(s.n), len(self.mapping))
                        self.targets = [float(v) for v in s.q[:n]]
                    self.model.step(self.targets, self.channels, self.kp, self.kd, self.n_sub)
                    snap = self.model.read_state(self.channels)
                    ft = [[0.0] * 6 for _ in range(8)]   # 线类型容量恒 8 路
                    if cfg["ft_at_ms"] >= 0 and el >= cfg["ft_at_ms"]:
                        if self.tick % 50 == 0:
                            print(f"joint_sim(resident): FT 注入中 mag={cfg['ft_mag']} el={el:.0f}ms", file=sys.stderr, flush=True)
                        ft[0][2] = cfg["ft_mag"]
                        ft[1][2] = cfg["ft_mag"]
                    if not frozen:
                        self.writer.write(JointStates(
                            n=len(self.mapping), q=snap["q"], dq=snap["dq"], tau=snap["tau"],
                            imu=snap["imu"], n_ft=FT_PATHS, ft=ft))
                    self.tick += 1
                nxt = self.t0 + (self.tick * self.period)
                remain = nxt - time.monotonic()
                if remain > 0:
                    time.sleep(remain)


def resident_main() -> int:
    """常驻模式：HTTP /spawn 接收 stub 转发的 argv。--auto-spawn = 启动即按
    包 DDS 配置自spawn（runtimedispatch gate要求反馈流先于首派发在场）。"""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    body = ResidentBody()
    threading.Thread(target=body.run_forever, daemon=True).start()
    if "--auto-spawn" in sys.argv:
        pkg_path = os.environ.get("MINDPAL_SIM_PKG", PKG_DEFAULT)
        ccfg = tomllib.loads(open(pkg_path, encoding="utf-8").read())
        dds = ccfg["protocol"]["dds"]
        argv_spawn = [sys.argv[0],
                      str(dds["domain_id"]),
                      next(iter(dds["actuator_topics"].values()))["topic_name"],
                      next(iter(dds["sensor_topics"].values()))["topic_name"],
                      str(ccfg["protocol"]["frequency_hz"])]
        body.spawn(argv_spawn)
        # 感知在环的数据源：模型调 camera.capture 前先有帧可取
        from .camera_stream import FrameServer
        # 供数相机 = 声明里第一台 HTTP 供数面（http_image/http_depth——主相机可能是 v4l2）
        cam = next((c for c in ccfg.get("acquisition", {}).get("cameras") or []
                    if c.get("kind") in ("http_image", "http_depth")), {}) or {}
        if cam.get("kind") == "http_image":
            intr = cam.get("intrinsics", {})
            fs = FrameServer(body.model, "wrist_cam",
                             int(intr.get("cx", 320) * 2), int(intr.get("cy", 240) * 2),
                             port=8284)
            fs.start()
            print("joint_sim(resident): camera FrameServer on :8284/frame.jpg",
                  file=sys.stderr, flush=True)
        if cam.get("kind") == "http_depth":
            intr = cam.get("intrinsics", {})
            body.depth_mount_xyz = [float(v) for v in cam.get("mount_xyz_m") or body.depth_mount_xyz]
            body.depth_mount_rpy = [float(v) for v in cam.get("mount_rpy_deg") or body.depth_mount_rpy]
            fs = FrameServer(body.model, "depth_cam",
                             int(intr.get("cx", 160) * 2), int(intr.get("cy", 120) * 2),
                             port=8284, depth=True, intr=intr)
            fs.start()
            print("joint_sim(resident): depth FrameServer on :8284/depth.raw "
                  f"mount={body.depth_mount_xyz} rpy={body.depth_mount_rpy}",
                  file=sys.stderr, flush=True)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            if self.path == "/variant":
                try:
                    out = json.dumps(body.set_variant(json.loads(raw))).encode()
                    self.send_response(200)
                except Exception as e:
                    out = json.dumps({"error": str(e)}).encode()
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            argv = json.loads(raw)["argv"]
            body.spawn(argv)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def do_GET(self):
            if self.path == "/gt.json":
                try:
                    out = json.dumps(body.get_gt()).encode()
                    self.send_response(200)
                except Exception as e:
                    out = json.dumps({"error": str(e)}).encode()
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, *a):
            pass

    # 0.0.0.0 = 端口旁路面（compose 发布 8299 → 宿主 judge 路由可达；
    # docker-proxy 转发到容器 eth0——绑 127.0.0.1 时旁路连接被挂断，实测）
    print("joint_sim(resident): listening 0.0.0.0:8299/spawn", file=sys.stderr, flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8299), Handler).serve_forever()
    return 0


def standalone_main(argv: list[str]) -> int:
    """无 resident 回退：本进程闭环（冷启动语义如实保留）。"""
    cfg = parse_args(argv)
    pkg_path = os.environ.get("MINDPAL_SIM_PKG", PKG_DEFAULT)
    model_path = os.environ.get("MINDPAL_SIM_MODEL", MODEL_DEFAULT)
    ccfg = tomllib.loads(open(pkg_path, encoding="utf-8").read())
    mapping = ccfg["protocol"]["joint_mapping"]["ordered"]
    kp = float(ccfg["profile"]["dynamics"]["kp"])
    kd = float(ccfg["profile"]["dynamics"]["kd"])
    m = MjModel(model_path)
    channels = m.channels(mapping)
    m.reset()
    n_sub = substep_ratio(SIM_HZ, float(cfg["hz"]), m.timestep)
    dp = DomainParticipant(cfg["domain"])
    reader = DataReader(dp, Topic(dp, cfg["cmd"], JointTargets), qos=BEST_EFFORT)
    writer = DataWriter(dp, Topic(dp, cfg["state"], JointStates), qos=BEST_EFFORT)
    print(f"joint_sim(mujoco shim): domain={cfg['domain']} cmd={cfg['cmd']} "
          f"state={cfg['state']} hz={cfg['hz']} slots={len(mapping)}", file=sys.stderr)
    import mujoco
    t0 = time.monotonic()
    targets = [0.0] * len(mapping)
    perturbed = False
    tick = 0
    period = 1.0 / cfg["hz"]
    while True:
        el = (time.monotonic() - t0) * 1000.0
        if cfg["stop_after_ms"] >= 0 and el >= cfg["stop_after_ms"]:
            print(f"joint_sim: FAULT stop-after {cfg['stop_after_ms']}ms 触发——"
                  f"进程退出（反馈断流）\nelapsed={el:.0f}ms", file=sys.stderr)
            return 3
        frozen = cfg["freeze_at_ms"] >= 0 and \
            cfg["freeze_at_ms"] <= el < cfg["freeze_at_ms"] + cfg["freeze_dur_ms"]
        if cfg["perturb_at_ms"] >= 0 and el >= cfg["perturb_at_ms"] and not perturbed:
            perturbed = True
            if 0 <= cfg["perturb_slot"] < len(mapping):
                qa, _ = channels[cfg["perturb_slot"]]
                m.data.qpos[qa] += cfg["perturb_mag"]
                mujoco.mj_forward(m.model, m.data)
                print(f"joint_sim: PERTURB slot={cfg['perturb_slot']} mag={cfg['perturb_mag']:.3f}",
                      file=sys.stderr)
        targets = [0.0] * len(mapping)       # 无有效指令 = 保持零目标（不崩溃）
        dbg = os.environ.get("MINDPAL_SIM_DEBUG")
        for s in reader.take():
            if not hasattr(s, "n"):
                continue                     # InvalidSample（DDS 无效样本——诚实跳过）
            n = min(int(s.n), len(mapping))
            targets = [float(v) for v in s.q[:n]]
            if dbg and tick % 50 == 0:
                cur = m.read_state(channels)["q"][:5]
                print(f"joint_sim(debug) tick={tick} 收到q[:5]={[round(v, 3) for v in targets[:5]]} "
                      f"实际q[:5]={[round(v, 3) for v in cur]}",
                      file=sys.stderr)
        m.step(targets, channels, kp, kd, n_sub)
        snap = m.read_state(channels)
        ft = [[0.0] * 6 for _ in range(8)]
        if cfg["ft_at_ms"] >= 0 and el >= cfg["ft_at_ms"]:
            ft[0][2] = cfg["ft_mag"]
            ft[1][2] = cfg["ft_mag"]
        if not frozen:
            writer.write(JointStates(n=len(mapping), q=snap["q"], dq=snap["dq"],
                                     tau=snap["tau"], imu=snap["imu"],
                                     n_ft=FT_PATHS, ft=ft))
        if frozen and tick % cfg["hz"] == 0:
            print("joint_sim: FAULT frozen（冻结帧中——连接在、数据死）", file=sys.stderr)
        tick += 1
        nxt = t0 + (tick * period)
        remain = nxt - time.monotonic()
        if remain > 0:
            time.sleep(remain)


def main(argv: list[str]) -> int:
    if "--resident" in argv:
        return resident_main()
    # stub 模式：resident 在场 → 转发即退（测试 kill 无害）；缺席 → standalone
    try:
        req = urllib.request.Request(
            RESIDENT_URL, data=json.dumps({"argv": argv}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=2).read()
        print("joint_sim(stub): forwarded to resident", file=sys.stderr)
        return 0
    except Exception:
        return standalone_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
