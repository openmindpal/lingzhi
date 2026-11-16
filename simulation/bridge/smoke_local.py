#!/usr/bin/env python3
"""桥接器本地冒烟（零 DDS）——模型装载/步进/传感快照/线打包。

判据（全部机械断言）：
  S1 模型关节 = 包 joint_mapping/state_order 同序（桥接器启动门）
  S2 正弦目标驱动 → 关节位形移动（PD 闭环物理事实）
  S3 传感快照：n=6、槽位填充、IMU 四元数归一（现实同款传感量）
  S4 线打包：JointTargets 体 424B / JointStates CDR 664B（wrapper 对拍依据）
用法: python -m bridge.smoke_local
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .mj_model import MjModel, substep_ratio
from .wire_types import (JOINTSTATES_BODY_BYTES, MP_JOINT_SLOTS,
                         JOINTTARGETS_BODY_BYTES, pack_jointtargets_body)

PKG = Path(__file__).resolve().parents[1] / "packages" / "sim-arm-6dof" / "body-package.toml"
MODEL = Path(__file__).resolve().parents[1] / "assets" / "models" / "arm_6dof.xml"
SIM_HZ = 500.0


def main() -> None:
    import tomllib
    cfg = tomllib.loads(PKG.read_text(encoding="utf-8"))
    mapping = cfg["protocol"]["joint_mapping"]["ordered"]
    state_order = cfg["profile"]["state_order"]
    kp = float(cfg["profile"]["dynamics"]["kp"])
    kd = float(cfg["profile"]["dynamics"]["kd"])

    # S1 模型/包一致性（桥接器启动门同款）
    m = MjModel(str(MODEL))
    assert state_order == m.joint_names[:len(state_order)], f"S1 FAIL {m.joint_names}"
    channels = m.channels(mapping)
    print(f"S1 PASS 模型关节=包 state_order 同序: {m.joint_names[:6]}")

    m.reset()
    n_sub = substep_ratio(SIM_HZ, 50.0, m.timestep)
    q0 = [m.data.qpos[qa] for qa, _ in channels]
    for t in range(100):                       # 100 拍 @50Hz = 2s
        targets = [0.3 * math.sin(0.05 * t) for _ in range(6)]
        m.step(targets, channels, kp, kd, n_sub)
    q1 = [m.data.qpos[qa] for qa, _ in channels]
    moved = max(abs(a - b) for a, b in zip(q0, q1))
    assert moved > 0.05, f"S2 FAIL 关节未动 max|Δq|={moved}"
    print(f"S2 PASS PD 闭环位形移动 max|Δq|={moved:.3f} rad")

    # S3 传感快照契约
    snap = m.read_state(channels)
    assert snap["n"] == 6 and len(snap["q"]) == MP_JOINT_SLOTS
    assert snap["q"][6:] == [0.0] * (MP_JOINT_SLOTS - 6), "槽外必须填零"
    w = sum(v * v for v in snap["imu"][:4]) ** 0.5
    assert abs(w - 1.0) < 1e-3, f"IMU 四元数未归一 |q|={w}"
    assert snap["n_ft"] == 0 and all(all(v == 0 for v in r) for r in snap["ft"]), \
        "无力觉模型必须诚实 n_ft=0"
    print(f"S3 PASS 快照 n=6 槽位对齐 IMU|q|={w:.4f} n_ft=0(诚实缺省)")

    body = pack_jointtargets_body(6, [0.1] * 6, 40.0, 2.0)
    assert len(body) == JOINTTARGETS_BODY_BYTES == 424, f"S4 FAIL JointTargets 体 {len(body)}"
    import struct
    n = struct.unpack("<I", body[:4])[0]
    q0v = struct.unpack("<f", body[4:8])[0]
    kp_v = struct.unpack("<f", body[144:148])[0]
    assert (n, abs(q0v - 0.1) < 1e-6, abs(kp_v - 40.0) < 1e-6) == (6, True, True)
    from .wire_types import JointStates
    s = JointStates(n=6, q=[0.1] * 6 + [0.0] * 29, dq=[0.0] * 35, tau=[0.0] * 35,
                    imu=[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_ft=0,
                    ft=[[0.0] * 6 for _ in range(8)])
    raw = s.serialize(use_version_2=False)     # XCDR1（CDR-LE——wrapper 同线格式）
    assert len(raw) == JOINTSTATES_BODY_BYTES + 4 == 664, f"S4 FAIL JointStates {len(raw)}≠664"
    print(f"S4 PASS JointTargets 体 424B / JointStates CDR {len(raw)}B（wrapper 对拍依据）")

    print("SMOKE PASS 全绿（bridge 本地冒烟 4/4）")


if __name__ == "__main__":
    main()
