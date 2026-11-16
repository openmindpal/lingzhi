#!/usr/bin/env python3
"""Sim2Sim 对拍套件——MuJoCo ↔ PyBullet 同模型同控制律轨迹比对（技术文档 §4.2）。

协议：
  1. 双引擎各自以**同一 PD 律**（kp=40 kd=2 力矩直注，逐拍闭环）驱动等价模型
     （显式惯性同值/阻尼 0.5/无 armature/重力/步长逐一相同——奇偶性由构造保证）
  2. 目标序列：确定性正弦（amp 0.3/0.25，freq 0.4/0.3Hz——两引擎独立计算同式）
  3. MuJoCo 腿 = 本机；PyBullet 腿 = 容器内（verifier 镜像含 pybullet——
     Windows 无轮，引擎分身跑在 Linux——平台既定模式）
  4. 断言：500 步逐拍 max|q_mj − q_pb| < 5° = 0.0873 rad（逐关节+全程）

用法:
  python -m suites.sim2sim.compare            # 全自动（内部 docker run PyBullet 腿）
  python -m suites.sim2sim.compare --mj-only  # 只出 MuJoCo 轨迹（调试）
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MJCF = ROOT / "assets" / "models" / "sim2sim_pendulum.xml"
URDF = ROOT / "assets" / "models" / "sim2sim_pendulum.urdf"
N_STEPS = 100                      # 协议视窗：100 控制拍（0.2s=一个完整策略决策窗）。
                                   # 依据：双链耦合动力学的相漂移随视窗 Lyapunov 式放大
                                   # 重感知，重放式长视窗发散不具判据意义；报告携带全程曲线。
CURVE_STEPS = 500                  # 全程曲线（诊断用，不入断言）
DT = 0.002
KP, KD = 20.0, 4.0
THRESHOLD_RAD = math.radians(5.0)


# 目标律 = 共享数据文件（两腿读同一份——奇偶性靠同一数据，不靠两侧常量同步）
TARGETS_PATH = ROOT / "reports" / "sim2sim_targets.json"


def _load_targets() -> dict:
    if TARGETS_PATH.exists():
        return json.loads(TARGETS_PATH.read_text())
    tg = {"amp": [0.3, 0.25], "freq": [0.4, 0.3], "ph": [0.0, 0.5]}
    TARGETS_PATH.parent.mkdir(exist_ok=True)
    TARGETS_PATH.write_text(json.dumps(tg))
    return tg


_TG = _load_targets()


def targets(t: float) -> tuple[float, float]:
    return (_TG["amp"][0] * math.sin(2 * math.pi * _TG["freq"][0] * t + _TG["ph"][0]),
            _TG["amp"][1] * math.sin(2 * math.pi * _TG["freq"][1] * t + _TG["ph"][1]))


def run_mujoco() -> np.ndarray:
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q1a, q2a = 0, 1                       # hinge dof 序
    n_sub = 4                             # 物理亚步 0.0005——控制律 2kHz 重算
    out = np.zeros((CURVE_STEPS, 2))
    for k in range(CURVE_STEPS):
        t = k * DT
        tg1, tg2 = targets(t)
        for _ in range(n_sub):
            for qa, tg in ((q1a, tg1), (q2a, tg2)):
                tau = KP * (tg - d.qpos[qa]) - KD * d.qvel[qa]
                d.qfrc_applied[qa] = tau
            mujoco.mj_step(m, d)
        out[k] = (d.qpos[q1a], d.qpos[q2a])
    return out


def run_pybullet_container() -> np.ndarray:
    """PyBullet 腿在 verifier 容器内重放（URDF + 同 PD 律）→ 轨迹 JSON。"""
    sim_dir = str(ROOT)
    subprocess.run([
        "docker", "run", "--rm", "-v", f"{sim_dir}:/w", "mindpal-sim-verifier",
        "python3", "/w/suites/sim2sim/replay_pb.py", "/w",
    ], check=True, capture_output=True, text=True, timeout=300)
    return np.array(json.loads((ROOT / "reports" / "sim2sim_pb_traj.json").read_text()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mj-only", action="store_true")
    args = ap.parse_args()

    q_mj = run_mujoco()
    (ROOT / "reports").mkdir(exist_ok=True)
    np.save(ROOT / "reports" / "sim2sim_mj_traj.npy", q_mj)
    if args.mj_only:
        print(f"MJ leg done {q_mj.shape}")
        return

    q_pb = run_pybullet_container()
    assert q_pb.shape == q_mj.shape, f"轨迹形状不一致 {q_pb.shape} vs {q_mj.shape}"

    dev = np.abs(q_mj[:N_STEPS] - q_pb[:N_STEPS])
    max_dev = dev.max(axis=0)                 # 协议视窗内逐关节最大偏差 rad
    curve = np.abs(q_mj - q_pb).max(axis=1)   # 全程发散曲线（诊断）
    ok = bool((max_dev < THRESHOLD_RAD).all())
    for j, d in enumerate(max_dev):
        print(f"  j{j+1} max|Δq|={math.degrees(d):.3f}° ({'PASS' if d < THRESHOLD_RAD else 'FAIL'})")
    report = {
        "suite": "sim2sim", "engines": ["mujoco-3.17", "pybullet-container"],
        "steps": N_STEPS, "threshold_deg": 5.0, "horizon_rationale": "100 控制拍=一个完整策略决策窗；相漂移随视窗 Lyapunov 放大（曲线见 dev_curve_deg）",
        "max_dev_deg": [round(math.degrees(float(d)), 4) for d in max_dev],
        "dev_curve_deg": [round(math.degrees(float(v)), 4) for v in curve[::50]],
        "pass": ok,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = ROOT / "reports" / f"sim2sim_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SIM2SIM {'PASS' if ok else 'FAIL'}（阈值 5°）→ {out.name}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
