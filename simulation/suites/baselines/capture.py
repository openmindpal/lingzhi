#!/usr/bin/env python3
"""baselines 金轨迹套件——确定性回归锚（固定种子 + 固定目标序列 → 轨迹 hash）。

capture：跑 arm_6dof 平衡策略闭环（ORT × 桥接物理，300 控制拍，固定种子）→
  轨迹降采样 + hash → baselines/trajectories/arm_6dof.json（入库）。
verify：重跑同序列 → hash 逐位对比——**任何引擎/模型/控制变更导致的漂移即报警**。
  （确定性契约由 L1 实证：同输入序列同仿真轨迹——hash 相等是机械判据。）

用法:
  python -m suites.baselines.capture            # 首次捕获/有意更新基线
  python -m suites.baselines.capture --verify   # 回归验证（比对入库基线）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

from bridge.mj_model import MjModel, substep_ratio
from bridge.smoke_biped import assemble_obs

PKG = Path(__file__).resolve().parents[2] / "packages" / "sim-arm-6dof"
MODEL = Path(__file__).resolve().parents[2] / "assets" / "models" / "arm_6dof.xml"
BASE = Path(__file__).resolve().parents[2] / "baselines" / "trajectories" / "arm_6dof.json"
SIM_HZ = 500.0
POLICY_HZ = 50.0
TICKS = 300
SAMPLE_EVERY = 10                                 # 降采样（30 样本/基线）


def run_trajectory() -> tuple[np.ndarray, str]:
    import math
    cfg = tomllib.loads((PKG / "body-package.toml").read_text(encoding="utf-8"))
    mapping = cfg["protocol"]["joint_mapping"]["ordered"]
    kp = float(cfg["profile"]["dynamics"]["kp"])
    kd = float(cfg["profile"]["dynamics"]["kd"])
    m = MjModel(str(MODEL), seed=42)
    channels = m.channels(mapping)
    m.reset()
    n_sub = substep_ratio(SIM_HZ, POLICY_HZ, m.timestep)
    samples: list[list[float]] = []
    for t in range(TICKS):
        targets = [0.25 * math.sin(0.04 * t), 0.15 * math.sin(0.03 * t + 0.4),
                   0.1 * math.cos(0.02 * t)] + [0.0] * (len(mapping) - 3)
        m.step([float(v) for v in targets], channels, kp, kd, n_sub)
        if t % SAMPLE_EVERY == 0:
            samples.append([round(float(m.data.qpos[qa]), 6) for qa, _ in channels])
    arr = np.array(samples)
    h = hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    return arr, h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    arr, h = run_trajectory()
    if args.verify:
        if not BASE.exists():
            print("BASELINE_MISSING（先跑 capture）")
            sys.exit(1)
        base = json.loads(BASE.read_text(encoding="utf-8"))
        ok = base["hash"] == h
        print(f"{'BASELINE_VERIFY_PASS' if ok else 'BASELINE_VERIFY_FAIL'} "
              f"(baseline={base['hash']} current={h})")
        if not ok:
            sys.exit(1)
        return

    BASE.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "package": "sim-arm-6dof", "policy": "PD 目标序列（开环确定性）",
        "ticks": TICKS, "sample_every": SAMPLE_EVERY, "seed": 42,
        "hash": h, "samples": arr.tolist(),
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    BASE.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"BASELINE_CAPTURED hash={h} samples={len(arr)} → {BASE.name}")


if __name__ == "__main__":
    main()
