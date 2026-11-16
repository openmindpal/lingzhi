#!/usr/bin/env python3
"""runner——总编排器：一条命令跑全套（technical spec §5.2 复现流程第 3 步）。

套件序（依赖序）：冒烟（本地/DDS/双足）→ 扫掠（双包）→ 长跑 → 故障注入。
聚合报告 → simulation/reports/run_all_<ts>.json；任一 FAIL = 退出码 1。
engine_in_loop 套件（runtime在环 tri 族/生命周期）经 engine_launcher 接 Linux E2E——
本编排器只跑 bridge_policy_loop 面，不冒充全链。

用法:
  python -m runner.run_all                    # 全套
  python -m runner.run_all --only sweeps      # 过滤（名含子串）
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SUITE_PLAN: list[dict] = [
    {"name": "smoke_local", "module": "bridge.smoke_local", "args": []},
    {"name": "smoke_dds", "module": "bridge.smoke_dds", "args": ["--domain", "43"]},
    {"name": "smoke_biped", "module": "bridge.smoke_biped", "args": []},
    {"name": "goal_sweep_arm", "module": "suites.sweeps.goal_sweep",
     "args": ["--package", "packages/sim-arm-6dof", "--model", "assets/models/arm_6dof.xml"]},
    {"name": "goal_sweep_biped", "module": "suites.sweeps.goal_sweep",
     "args": ["--package", "packages/sim-biped-29dof", "--model", "assets/models/biped_29dof.xml"]},
    {"name": "longevity_biped", "module": "suites.longevity.run_soak",
     "args": ["--package", "packages/sim-biped-29dof", "--model", "assets/models/biped_29dof.xml",
              "--ticks", "2000", "--speed", "10"]},
    {"name": "fault_injection_arm", "module": "suites.fault_injection.run_faults",
     "args": ["--package", "packages/sim-arm-6dof", "--model", "assets/models/arm_6dof.xml"]},
    {"name": "task_success", "module": "suites.task_success.run_tasks", "args": []},
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="套件名子串过滤")
    args = ap.parse_args()

    plan = [s for s in SUITE_PLAN if not args.only or args.only in s["name"]]
    rows = []
    failed = 0
    t0 = time.monotonic()
    for s in plan:
        t = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "-m", s["module"], *s["args"]],
            cwd=ROOT, capture_output=True, text=True, timeout=900,
            encoding="utf-8", errors="replace",   # 套件输出含 UTF-8 中文——GBK 控制台默认解码会炸（runner 必须比被验者更稳）
        )
        ok = proc.returncode == 0
        failed += 0 if ok else 1
        rows.append({"name": s["name"], "module": s["module"], "pass": ok,
                     "elapsed_s": round(time.monotonic() - t, 1),
                     "tail": (proc.stdout.strip().splitlines() or [""])[-1]
                             + (("\n| stderr: " + proc.stderr.strip().splitlines()[-1])
                                if proc.returncode and proc.stderr.strip() else "")})
        print(f"[{'PASS' if ok else 'FAIL'}] {s['name']:22s} {rows[-1]['elapsed_s']:>5}s  "
              f"{rows[-1]['tail'].splitlines()[0][:80]}")

    total = len(rows)
    passed = total - failed
    report = {
        "runner": "run_all", "mode": "bridge_policy_loop",
        "summary": {"total": total, "passed": passed, "failed": failed,
                    "wall_s": round(time.monotonic() - t0, 1)},
        "suites": rows,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"run_all_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRUN_ALL {passed}/{total} 通过"
          f"（wall {report['summary']['wall_s']}s）→ {out.relative_to(ROOT)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
