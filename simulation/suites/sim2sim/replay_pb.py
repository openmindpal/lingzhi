#!/usr/bin/env python3
"""PyBullet 腿——容器内重放（同 PD 律同目标律，输出轨迹 JSON 供宿主比对）。
用法（容器内）: python3 /w/suites/sim2sim/replay_pb.py /w
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pybullet as p

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/w")
URDF = ROOT / "assets" / "models" / "sim2sim_pendulum.urdf"
N_STEPS = 500
DT = 0.002
KP, KD = 20.0, 4.0


_tg = json.loads((ROOT / "reports" / "sim2sim_targets.json").read_text())     if (ROOT / "reports" / "sim2sim_targets.json").exists() else     {"amp": [0.3, 0.25], "freq": [0.4, 0.3], "ph": [0.0, 0.5]}


def targets(t: float) -> tuple[float, float]:
    return (_tg["amp"][0] * math.sin(2 * math.pi * _tg["freq"][0] * t + _tg["ph"][0]),
            _tg["amp"][1] * math.sin(2 * math.pi * _tg["freq"][1] * t + _tg["ph"][1]))


cid = p.connect(p.DIRECT)
p.setGravity(0, 0, -9.81, physicsClientId=cid)
p.setTimeStep(DT, physicsClientId=cid)
obj = p.loadURDF(str(URDF), useFixedBase=True,
                 flags=p.URDF_USE_INERTIA_FROM_FILE, physicsClientId=cid)  # 显式惯性——默认按几何重算（实测响应差 200×）
n_j = p.getNumJoints(obj, physicsClientId=cid)
jmap = {}
for i in range(n_j):
    info = p.getJointInfo(obj, i, physicsClientId=cid)
    jmap[info[1].decode()] = i
p.changeDynamics(obj, jmap["j1"], linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
p.changeDynamics(obj, jmap["j2"], linearDamping=0.0, angularDamping=0.0, physicsClientId=cid)
for name in ("j1", "j2"):
    p.setJointMotorControl2(obj, jmap[name], p.VELOCITY_CONTROL,
                            force=0, physicsClientId=cid)   # 关默认电机——纯力矩域

p.setTimeStep(0.0005, physicsClientId=cid)   # 物理亚步——与 MuJoCo 同 2kHz 控制律

out = np.zeros((N_STEPS, 2))
for k in range(N_STEPS):
    t = k * DT
    tg1, tg2 = targets(t)
    for _ in range(4):
        states = p.getJointState(obj, jmap["j1"], physicsClientId=cid)[0:2] + \
            p.getJointState(obj, jmap["j2"], physicsClientId=cid)[0:2]
        (q1, dq1, q2, dq2) = states
        tau1 = KP * (tg1 - q1) - KD * dq1
        tau2 = KP * (tg2 - q2) - KD * dq2
        # 关节空间力矩直注（TORQUE_CONTROL）——applyExternalTorque 的 LINK_FRAME
        # 自由力矩在铰链链上语义不等于关节扭矩（实测 2 拍翻转发散）
        p.setJointMotorControl2(obj, jmap["j1"], p.TORQUE_CONTROL, force=tau1, physicsClientId=cid)
        p.setJointMotorControl2(obj, jmap["j2"], p.TORQUE_CONTROL, force=tau2, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
    out[k] = (p.getJointState(obj, jmap["j1"], physicsClientId=cid)[0],
              p.getJointState(obj, jmap["j2"], physicsClientId=cid)[0])

(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / "sim2sim_pb_traj.json").write_text(json.dumps(out.tolist()))
print(f"PB leg done {out.shape} q_end={out[-1].round(4).tolist()}")
