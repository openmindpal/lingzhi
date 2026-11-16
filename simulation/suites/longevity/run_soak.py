#!/usr/bin/env python3
"""longevity 长跑套件——时间加速 + 机械断言（技术文档 §3.3，引擎缺席版）。

断言（全部机械数值）：
  L1 确定性回归   固定种子：重放首段轨迹 hash 逐位一致（§2.3 确定性契约）
  L2 资源无泄漏   RSS 曲线 warmup 后增长 < 30MB（长时曲线断言）
  L3 限位长期尊重 全 soak 逐拍稳态 violation=0（balance 闭环持位）
  L4 力矩长期有界 全 soak max|tau| < bound
  L5 时间加速     speed>1 下 wall_time < sim_time（headless 批量 >实时）

引擎在环生命周期断言（觉醒/checkpoint 续跑/预算续做/挂起续跑/TTL）= executor
"engine_in_loop"，接 Linux E2E——本套件只覆盖物理/资源面，不冒充认知面。
用法: python -m suites.longevity.run_soak --package packages/sim-biped-29dof \
          --model assets/models/biped_29dof.xml [--ticks 2000] [--speed 10]
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
import onnxruntime as ort

from bridge.mj_model import MjModel, substep_ratio
from bridge.smoke_biped import assemble_obs
from bridge.time_control import TickClock

WARMUP_TICKS = 200
HASH_TICKS = 300
RSS_GROWTH_BOUND_MB = 30.0
TAU_BOUND = 80.0


def rss_mb() -> float:
    """进程 RSS（MB）——win ctypes / posix resource，单进程内自检足够。"""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        # 伪句柄 = (HANDLE)-1——GetCurrentProcess 默认 restype 截断为非法句柄（err=6）
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.c_void_p(-1), ctypes.byref(pmc), pmc.cb)
        if not ok:
            raise OSError(f"GetProcessMemoryInfo failed err={ctypes.GetLastError()}")
        if pmc.WorkingSetSize <= 0:
            raise OSError("WorkingSetSize 空值")
        return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def traj_hash(m: MjModel, channels, sess, kp, kd, n_sub, L, ticks) -> str:
    """固定种子闭环轨迹 hash（每拍 q 量化拼接——确定性回归判据）。"""
    m.reset()
    h = hashlib.sha256()
    for _ in range(ticks):
        act = sess.run(None, {"obs": assemble_obs(m, channels)[None, :]})[0][0]
        m.step([float(v) for v in act], channels, kp, kd, n_sub)
        q = np.array([m.data.qpos[qa] for qa, _ in channels], dtype=np.float32)
        h.update(q.tobytes())
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pkg = Path(args.package)
    cfg = tomllib.loads((pkg / "body-package.toml").read_text(encoding="utf-8"))
    mapping = cfg["protocol"]["joint_mapping"]["ordered"]
    neutral_map = cfg["calibration"]["neutral"]
    kp = float(cfg["profile"]["dynamics"]["kp"])
    kd = float(cfg["profile"]["dynamics"]["kd"])
    L = len(mapping)
    neutral = np.array([float(neutral_map[n]) for n in mapping], dtype=np.float32)
    margin = float(cfg.get("safety", {}).get("soft_limit_margin_rad", 0.05))
    lims = [(float(cfg["profile"]["joints"][n]["limits"]["min"]) + margin,
             float(cfg["profile"]["joints"][n]["limits"]["max"]) - margin) for n in mapping]
    row = next(r for r in cfg["policies"]["rows"] if r["family"] == "balance")

    m = MjModel(args.model, seed=args.seed)
    channels = m.channels(mapping)
    n_sub = substep_ratio(500.0, float(row["frequency_hz"]), m.timestep)
    sess = ort.InferenceSession(str(pkg / row["weights"]), providers=["CPUExecutionProvider"])

    # L1 确定性回归（重放首段）
    h1 = traj_hash(m, channels, sess, kp, kd, n_sub, L, HASH_TICKS)
    h2 = traj_hash(m, channels, sess, kp, kd, n_sub, L, HASH_TICKS)
    assert h1 == h2, f"L1 FAIL 确定性破坏 {h1}≠{h2}"
    print(f"L1 PASS 确定性回归 hash={h1}")

    m.reset()
    clock = TickClock(period_s=1.0 / float(row["frequency_hz"]), speed=args.speed)
    rss_curve: list[tuple[int, float]] = []
    viol_total = 0
    tau_max = 0.0
    t0 = time.monotonic()
    for t in range(args.ticks):
        act = sess.run(None, {"obs": assemble_obs(m, channels)[None, :]})[0][0]
        m.step([float(v) for v in act], channels, kp, kd, n_sub)
        q = np.array([m.data.qpos[qa] for qa, _ in channels], dtype=np.float32)
        tau = np.abs(np.array([m.data.qfrc_applied[da] for _, da in channels]))
        tau_max = max(tau_max, float(tau.max()))
        viol_total += sum(1 for i in range(L)
                          if not (lims[i][0] - 1e-6 <= q[i] <= lims[i][1] + 1e-6))
        if t % max(args.ticks // 10, 1) == 0:
            rss_curve.append((t, round(rss_mb(), 1)))
        clock.wait_tick()
    wall = time.monotonic() - t0
    warm_rss = rss_curve[0][1]
    rss_growth = rss_curve[-1][1] - warm_rss
    assert rss_growth < RSS_GROWTH_BOUND_MB, \
        f"L2 FAIL RSS 增长 {rss_growth:.1f}MB ≥ {RSS_GROWTH_BOUND_MB}（泄漏嫌疑）"
    assert viol_total == 0, f"L3 FAIL 限位 violation={viol_total}"
    assert tau_max < TAU_BOUND, f"L4 FAIL max|tau|={tau_max:.1f}"
    assert wall < clock.sim_time, f"L5 FAIL wall={wall:.1f}s ≥ sim={clock.sim_time:.1f}s"
    print(f"L2 PASS RSS {warm_rss:.0f}→{rss_curve[-1][1]:.0f}MB（增长 {rss_growth:.1f}）")
    print(f"L3 PASS 限位 violation=0（{args.ticks} 拍）")
    print(f"L4 PASS max|tau|={tau_max:.1f} N·m（<{TAU_BOUND}）")
    print(f"L5 PASS 时间加速 wall={wall:.1f}s < sim={clock.sim_time:.1f}s（speed={args.speed}）")

    out_dir = Path(__file__).resolve().parents[2] / "reports"
    out_dir.mkdir(exist_ok=True)
    report = {
        "suite": "longevity_soak", "mode": "bridge_policy_loop",
        "package": cfg["core"]["package_name"], "ticks": args.ticks,
        "determinism_hash": h1, "rss_curve": rss_curve,
        "rss_growth_mb": rss_growth, "limit_violations": viol_total,
        "tau_max": round(tau_max, 1), "wall_s": round(wall, 1),
        "sim_s": round(clock.sim_time, 1), "speed": args.speed,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / f"longevity_{cfg['core']['package_name']}_{int(time.time())}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"LONGEVITY PASS 全绿（L1~L5）→ reports/")


if __name__ == "__main__":
    main()
