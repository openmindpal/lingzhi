#!/usr/bin/env python3
"""task_success——任务成功判据套件（具身智能能力水平语义层判定 + 系统得分）。

判定命题：goal_conditioned 策略回合 = 一次"任务"——单 episode 内按序到达声明的
路标位形（连续执行，不逐路标重置——长时程语义），三信号机械判定（缺一即该路标不计）：
  S1 到达    该路标窗口稳态 q 与解析预测的残差 ≤ tol
             （预测 = 策略 ONNX 权重自描述导出的线性稳态解——q* = act*（伺服收敛）
               ⇒ q* = (I−W_q)⁻¹(W_gx·gx + b)；非硬编码增益）
  S2 稳定    窗口收尾 max|Δq| ≤ settle_tol（到位且保持，非路过）
  S3 安全    限位违规 = 0 且 max|tau| < tau_bound（违规/超限 = 该任务计 0）
部分成功评分 = Σ 路标分 / 路标数（每路标 = clamp(1 − 残差/tol, 0, 1)）。
带载任务（load_windows）：窗口内外扰力矩持续注入——带载残差与恢复残差分别入册。
维度子分（precision/horizon/repeat/timed/disturbance/cross_body）→ 系统得分 0~100。

报告：simulation/reports/task_success_<ts>.json（逐路标行 + 任务汇总 + 维度分解 +
系统得分 + 轨迹 hash 确定性指纹）；任一任务非 SUCCESS = 退出码 1。
用法:
  python -m suites.task_success.run_tasks                # 全任务集（双本体）
  python -m suites.task_success.run_tasks --tasks biped  # 名含子串过滤
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from bridge.mj_model import MjModel, substep_ratio
from bridge.smoke_biped import assemble_obs

DEFAULTS = {"tol": 0.25, "settle_tol": 0.005, "ticks_per_wp": 80, "tau_bound": 80.0,
            "settle_window": 8,
            # tol = 系统定位精度契约（与 goal_sweep A1 track_bound 同源——伺服权威）
            }
ARM = {"package": "packages/sim-arm-6dof", "model": "assets/models/arm_6dof.xml"}
BIPED = {"package": "packages/sim-biped-29dof", "model": "assets/models/biped_29dof.xml"}


def default_tasks() -> list[dict]:
    """任务集：6 判定维度 × 10 任务（双本体）。物体操作类（推杯/抓杯）待真训策略。"""
    return [
        {"name": "reach_half", "family": "precision", **ARM,
         "waypoints": [{"gx": 0.3, "gf": 0.8}]},
        {"name": "reach_far", "family": "precision", **ARM,
         "waypoints": [{"gx": 0.5, "gf": 0.8}]},
        {"name": "long_horizon_3wp", "family": "horizon", **ARM,
         "waypoints": [{"gx": 0.15, "gf": 0.8}, {"gx": 0.3, "gf": 0.8}, {"gx": 0.45, "gf": 0.8}]},
        {"name": "long_horizon_5wp", "family": "horizon", **ARM,
         "waypoints": [{"gx": 0.1, "gf": 0.8}, {"gx": 0.2, "gf": 0.8}, {"gx": 0.3, "gf": 0.8},
                       {"gx": 0.4, "gf": 0.8}, {"gx": 0.5, "gf": 0.8}]},
        {"name": "return_repeat", "family": "repeat", **ARM,
         "waypoints": [{"gx": 0.4, "gf": 0.8}, {"gx": 0.1, "gf": 0.8}, {"gx": 0.4, "gf": 0.8}],
         "repeat": [0, 2]},
        {"name": "timed_reach", "family": "timed", **ARM,
         "waypoints": [{"gx": 0.5, "gf": 0.8}], "ticks_per_wp": 60},
        {"name": "disturb_recover", "family": "disturbance", **ARM,
         "waypoints": [{"gx": 0.4, "gf": 0.8}],
         "load_windows": [[0.0, 0.5]], "dof_index": 0, "torque": 3.0},
        {"name": "hold_under_disturbance", "family": "disturbance", **ARM,
         "waypoints": [{"gx": 0.4, "gf": 0.8}],
         "load_windows": [[0.5, 1.0]], "dof_index": 0, "torque": 2.0},
        {"name": "biped_reach", "family": "cross_body", **BIPED,
         "waypoints": [{"gx": 0.3, "gf": 0.8}]},
        {"name": "biped_long_horizon", "family": "cross_body", **BIPED,
         "waypoints": [{"gx": 0.15, "gf": 0.8}, {"gx": 0.3, "gf": 0.8}]},
    ]


FAMILIES = ["precision", "horizon", "repeat", "timed", "disturbance", "cross_body"]


def linear_steady_predictor(onnx_path: str, mapping_len: int):
    """线性策略稳态解析预测器——从 ONNX 权重自描述导出（非硬编码增益）。

    稳态条件：q* = act*（伺服收敛）、dq* = 0、tau 观测块 = 0（生成器契约：纯
    goal_x 耦合）、IMU 零耦合。act* = W_q·q* + W_gx·gx + b
    ⇒ q* = (I − W_q)⁻¹ (W_gx·gx + b)。
    非纯耦合（dq/tau/IMU 列非零）= 诚实拒绝（预测器不猜）。
    """
    m = onnx.load(onnx_path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    W = np.asarray(inits["W"], dtype=np.float64)
    b = np.asarray(inits["b"], dtype=np.float64)
    obs_dim = W.shape[1]
    L = mapping_len
    if obs_dim != 3 * L + 10 + 4:
        raise SystemExit(f"线性预测不支持: obs_dim {obs_dim} ≠ 3L+10+4 (L={L})")
    W_q = W[:, :L]
    if np.any(W[:, L:obs_dim - 4] != 0) or np.any(W[:, obs_dim - 3:] != 0):
        raise SystemExit("线性稳态预测仅支持纯 goal_x 耦合策略（其余观测列须零耦合）")
    W_gx = W[:, obs_dim - 4]

    def predict(gx: float) -> np.ndarray:
        return np.linalg.solve(np.eye(L) - W_q, b + W_gx * gx)

    return predict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=None, help="任务名子串过滤")
    ap.add_argument("--ticks-per-wp", type=int, default=DEFAULTS["ticks_per_wp"])
    ap.add_argument("--tol", type=float, default=DEFAULTS["tol"])
    args = ap.parse_args()
    p = {**DEFAULTS, "ticks_per_wp": args.ticks_per_wp, "tol": args.tol}

    tasks = [t for t in default_tasks() if not args.tasks or args.tasks in t["name"]]
    if not tasks:
        raise SystemExit("任务过滤后为空")

    sessions: dict[tuple, tuple] = {}
    results = []
    for t in tasks:
        key = (t["package"], t["model"])
        if key not in sessions:
            pkg = Path(t["package"])
            cfg = tomllib.loads((pkg / "body-package.toml").read_text(encoding="utf-8"))
            rows = [r for r in cfg["policies"]["rows"] if r.get("goal_conditioned")]
            if not rows:
                raise SystemExit(f"{t['package']} 无 goal_conditioned 行——无执行对象")
            row = rows[0]
            mapping = cfg["protocol"]["joint_mapping"]["ordered"]
            neutral = np.array([float(cfg["calibration"]["neutral"].get(n, 0.0)) for n in mapping])
            lims = {n: (float(j["limits"]["min"]), float(j["limits"]["max"]))
                    for n, j in cfg["profile"]["joints"].items()}
            margin = float(cfg.get("safety", {}).get("soft_limit_margin_rad", 0.05))
            kp = float(cfg["profile"]["dynamics"]["kp"])
            kd = float(cfg["profile"]["dynamics"]["kd"])
            m = MjModel(t["model"])
            channels = m.channels(mapping)
            n_sub = substep_ratio(500.0, float(row["frequency_hz"]), m.timestep)
            sess = ort.InferenceSession(str(pkg / row["weights"]), providers=["CPUExecutionProvider"])
            predict = linear_steady_predictor(str(pkg / row["weights"]), len(mapping))
            sessions[key] = (cfg, row, mapping, neutral, lims, margin, kp, kd, m, channels,
                             n_sub, sess, predict)

        (cfg, row, mapping, neutral, lims, margin, kp, kd, m, channels, n_sub, sess,
         predict) = sessions[key]
        n = len(mapping)
        predicted_all = [predict(w["gx"]) for w in t["waypoints"]]
        m.reset()
        total = p["ticks_per_wp"] * len(t["waypoints"])
        applied: set[int] = set()
        da = channels[t.get("dof_index", 0)][1]
        torque = t.get("torque", 3.0)
        load_windows = t.get("load_windows", [])
        q_hist: list[np.ndarray] = []
        tau_max = 0.0
        goals = [np.array([w["gx"], 0.0, 0.5, w.get("gf", 0.8)], dtype=np.float32)
                 for w in t["waypoints"]]
        for i in range(total):
            frac = i / max(1, total)
            on = any(f0 <= frac < f1 for f0, f1 in load_windows)
            if on and da not in applied:
                m.set_external(da, torque)
                applied.add(da)
            elif not on and da in applied:
                m.set_external(da, 0.0)
                applied.discard(da)
            goal = goals[min(i // p["ticks_per_wp"], len(goals) - 1)]
            obs = assemble_obs(m, channels, goal=[float(v) for v in goal])
            act = sess.run(None, {"obs": obs[None, :]})[0][0]
            m.step([float(v) for v in act], channels, kp, kd, n_sub)
            q = np.array([m.data.qpos[qa] for qa, _ in channels], dtype=np.float32)
            q_hist.append(q)
            tau = np.abs(np.array([m.data.qfrc_applied[da2] for _, da2 in channels]))
            tau_max = max(tau_max, float(tau.max()))
        for da2 in list(applied):
            m.set_external(da2, 0.0)

        wps_eval = []
        for k, w in enumerate(t["waypoints"]):
            w0, w1 = k * p["ticks_per_wp"], (k + 1) * p["ticks_per_wp"]
            win = q_hist[max(w1 - DEFAULTS["settle_window"], w0):w1]
            arr = np.array(win)
            q_steady = np.mean(arr, axis=0)
            settled = float(np.max(np.abs(np.diff(arr, axis=0)))) if len(arr) > 1 else 0.0
            residual = float(np.max(np.abs(q_steady - np.array(predicted_all[k]))))
            violations = sum(
                1 for i2, name in enumerate(mapping)
                if not (lims[name][0] + margin - 1e-6 <= q_steady[i2] <= lims[name][1] - margin + 1e-6)
            )
            unsafe = violations > 0 or tau_max >= p["tau_bound"]
            reached = residual <= p["tol"] and settled <= p["settle_tol"] and not unsafe
            score = 0.0 if unsafe else max(0.0, min(1.0, 1.0 - residual / p["tol"]))
            wps_eval.append({"gx": w["gx"],
                             "predicted": [round(float(v), 4) for v in predicted_all[k]],
                             "q_steady": [round(float(v), 4) for v in q_steady],
                             "residual": round(residual, 4), "settled": round(settled, 5),
                             "tau_max": round(tau_max, 1), "limit_violations": violations,
                             "reached": reached, "score": round(score, 3)})

        repeat_ok = True
        repeat_diff = None
        if "repeat" in t:
            i0, i1 = t["repeat"]
            s0 = np.array(wps_eval[i0]["q_steady"])
            s1 = np.array(wps_eval[i1]["q_steady"])
            repeat_diff = float(np.max(np.abs(s0 - s1)))
            repeat_ok = repeat_diff <= p["tol"]

        reached = sum(1 for w in wps_eval if w["reached"])
        score = round(sum(w["score"] for w in wps_eval) / len(wps_eval), 3)
        ok = reached == len(wps_eval) and repeat_ok
        status = "SUCCESS" if ok else ("PARTIAL" if reached > 0 or score > 0 else "FAILED")
        results.append({"task": t["name"], "family": t["family"], "status": status,
                        "waypoints_reached": reached, "waypoints_total": len(wps_eval),
                        "score": score, "repeat_ok": repeat_ok,
                        "repeat_diff": None if repeat_diff is None else round(repeat_diff, 4),
                        "tau_max": round(tau_max, 1),
                        "traj_sha": hashlib.sha256(
                            b"".join(q.tobytes() for q in q_hist)).hexdigest()[:16],
                        "waypoints": wps_eval})
        print(f"TASK {t['name']} [{t['family']}]: {status} ({reached}/{len(wps_eval)} 路标, "
              f"score={score})")

    dimensions = {f: round(sum(r["score"] for r in results if r["family"] == f)
                           / max(1, sum(1 for r in results if r["family"] == f)), 3)
                  for f in FAMILIES if any(r["family"] == f for r in results)}
    system_score = round(100 * sum(r["score"] for r in results) / max(1, len(results)), 1)
    unsafe_any = any(r["tau_max"] >= p["tau_bound"]
                     or any(w["limit_violations"] > 0 for w in r["waypoints"])
                     for r in results)

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "SUCCESS")
    report = {
        "suite": "task_success",
        "mode": "bridge_policy_loop",
        "tasks": results,
        "dimensions": dimensions,
        "system_score": None if unsafe_any else system_score,
        "unsafe_any": unsafe_any,
        "summary": {"total": total, "passed": passed, "failed": total - passed,
                    "success_rate": round(passed / total, 3)},
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_dir = Path(__file__).resolve().parents[2] / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"task_success_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    dims = "  ".join(f"{k}={v}" for k, v in dimensions.items())
    print(f"SCORE 灵智系统仿真得分: {report['system_score']}/100"
          + ("" if not unsafe_any else "（安全违规——得分记 0）"))
    print(f"DIMENSIONS {dims}")
    print(f"TASK_SUCCESS {passed}/{total} 任务 SUCCESS → {out.name}")
    if passed < total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
