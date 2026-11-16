#!/usr/bin/env python3
"""goal_sweep——动作泛化参数扫掠套件（技术文档 §3.2，引擎缺席版：策略 ORT × 桥接物理）。

扫掠：goal_conditioned 操作策略 × goal_x 网格 × goal_f 两档——逐点闭环运行，
机械断言（每点全过才算过）：
  A1 到位精度   max|q_steady − act_target| < bound（伺服权威——重力垂坠有界）
  A2 限位尊重   稳态 q ∈ [min+margin, max−margin] 逐通道（violation=0）
  A3 力矩有界   max|tau| < bound（正常作业力 << 联锁值纪律）

报告：simulation/reports/goal_sweep_<pkg>_<ts>.json（逐点行+汇总通过率）。
断言一律机械数值阈值——无"看起来对"通道。
用法: python -m suites.sweeps.goal_sweep --package packages/sim-arm-6dof --model assets/models/arm_6dof.xml
"""
from __future__ import annotations

import argparse
import json
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort

from bridge.mj_model import MjModel, substep_ratio
from bridge.smoke_biped import assemble_obs

DEFAULTS = {"track_bound": 0.25, "tau_bound": 80.0, "ticks": 80, "points": 5}


def soft_limits(cfg: dict) -> tuple[dict[str, tuple[float, float]], float]:
    margin = float(cfg.get("safety", {}).get("soft_limit_margin_rad", 0.05))
    lims = {name: (float(j["limits"]["min"]), float(j["limits"]["max"]))
            for name, j in cfg["profile"]["joints"].items()}
    return lims, margin


def run_point(m: MjModel, sess: ort.InferenceSession, channels, kp, kd, n_sub,
              goal: list[float], base_dim: int, ticks: int) -> dict:
    m.reset()
    act = np.zeros(len(channels), dtype=np.float32)
    tau_max = 0.0
    for _ in range(ticks):
        obs = assemble_obs(m, channels, goal=goal)
        act = sess.run(None, {"obs": obs[None, :]})[0][0]
        m.step([float(v) for v in act], channels, kp, kd, n_sub)
        tau = np.abs(np.array([m.data.qfrc_applied[da] for _, da in channels]))
        tau_max = max(tau_max, float(tau.max()))
    q = np.array([m.data.qpos[qa] for qa, _ in channels], dtype=np.float32)
    return {"act": act, "q": q, "track_err": float(np.max(np.abs(q - act))),
            "tau_max": tau_max}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--points", type=int, default=DEFAULTS["points"])
    ap.add_argument("--policy", default=None, help="goal_conditioned 行名（缺省取第一行）")
    args = ap.parse_args()

    pkg = Path(args.package)
    cfg = tomllib.loads((pkg / "body-package.toml").read_text(encoding="utf-8"))
    rows = [r for r in cfg["policies"]["rows"] if r.get("goal_conditioned")]
    if not rows:
        raise SystemExit("包内无 goal_conditioned 行——goal_sweep 无执行对象")
    row = next((r for r in rows if args.policy and r["name"] == args.policy), rows[0])
    pol_path = pkg / row["weights"]
    mapping = cfg["protocol"]["joint_mapping"]["ordered"]
    kp = float(cfg["profile"]["dynamics"]["kp"])
    kd = float(cfg["profile"]["dynamics"]["kd"])
    lims, margin = soft_limits(cfg)
    base_dim = 3 * len(mapping) + 10

    m = MjModel(args.model)
    channels = m.channels(mapping)
    n_sub = substep_ratio(500.0, float(row["frequency_hz"]), m.timestep)
    sess = ort.InferenceSession(str(pol_path), providers=["CPUExecutionProvider"])

    points_in: list[dict] = []
    for gx in np.linspace(0.1, 0.9, args.points):
        for gf in (0.2, 0.8):
            points_in.append({"gx": float(gx), "gf": gf})

    results = []
    fails = 0
    for p in points_in:
        goal = [p["gx"], 0.0, 0.5, p["gf"]]
        r = run_point(m, sess, channels, kp, kd, n_sub, goal, base_dim, DEFAULTS["ticks"])
        violations = sum(
            1 for i, name in enumerate(mapping)
            if not (lims[name][0] + margin - 1e-6 <= r["q"][i] <= lims[name][1] - margin + 1e-6)
        )
        ok = r["track_err"] < DEFAULTS["track_bound"] and violations == 0 \
            and r["tau_max"] < DEFAULTS["tau_bound"]
        fails += 0 if ok else 1
        results.append({
            "goal": goal,
            "track_err": round(r["track_err"], 4),
            "tau_max": round(r["tau_max"], 1),
            "limit_violations": violations,
            "pass": ok,
        })
        print(f"  gx={p['gx']:.2f} gf={p['gf']:.1f}  track={r['track_err']:.3f} "
              f"tau={r['tau_max']:.0f}N·m viol={violations}  {'PASS' if ok else 'FAIL'}")

    total = len(results)
    passed = total - fails
    report = {
        "suite": "goal_sweep",
        "mode": "bridge_policy_loop",
        "package": cfg["core"]["package_name"],
        "policy": row["name"],
        "points": results,
        "summary": {
            "total": total, "passed": passed, "failed": fails,
            "success_rate": round(passed / total, 3),
            "max_track_err": max(r["track_err"] for r in results),
            "max_tau": max(r["tau_max"] for r in results),
        },
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_dir = Path(__file__).resolve().parents[2] / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"goal_sweep_{cfg['core']['package_name']}_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    rate = report["summary"]["success_rate"]
    print(f"GOAL_SWEEP {cfg['core']['package_name']}/{row['name']}: "
          f"{passed}/{total} 通过率 {rate:.0%} → {out.name}")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
