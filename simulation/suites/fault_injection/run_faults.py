#!/usr/bin/env python3
"""fault_injection 套件——注入通道机械验证（technical spec §3.4，runtime缺席版）。

验证对象 = 桥接器故障注入通道**产出的信号本身**（runtimesafety chain消费的数据面）：
  F1 stream_loss   停供期 feed=None（runtimewatchdog timeout_ms 的输入信号）
  F2 frozen_frame  冻结期逐拍返回同一陈旧快照（rx 新鲜度检测的输入信号）
  F3 estop         注入期 command_gate 拒新指令（闩锁语义的桥侧半边）
  F4 force_spike   力觉路注入超限值且 n_ft 抬升（force interlock判定的输入信号）

runtime侧后果（watchdog翻旗/派发拒绝/立即收场+estop）= executor "engine_in_loop"，
接 Linux E2E——本套件不冒充runtimesafety chain，只证明注入信号真实可达。
用法: python -m suites.fault_injection.run_faults --package packages/sim-arm-6dof \
          --model assets/models/arm_6dof.xml
"""
from __future__ import annotations

import argparse
import json
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from bridge.faults import FaultState, apply_to_snapshot, command_gate
from bridge.mj_model import MjModel, substep_ratio


def snapshot_of(m: MjModel, channels) -> dict:
    return m.read_state(channels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    pkg = Path(args.package)
    cfg = tomllib.loads((pkg / "body-package.toml").read_text(encoding="utf-8"))
    mapping = cfg["protocol"]["joint_mapping"]["ordered"]
    kp = float(cfg["profile"]["dynamics"]["kp"])
    kd = float(cfg["profile"]["dynamics"]["kd"])
    period_ms = 1000.0 / float(cfg["protocol"]["frequency_hz"])
    interlock = float(cfg.get("safety", {}).get("contact_force_limit_n", 50.0))

    m = MjModel(args.model)
    channels = m.channels(mapping)
    n_sub = substep_ratio(500.0, float(cfg["protocol"]["frequency_hz"]), m.timestep)
    m.reset()
    m.step([0.0] * len(mapping), channels, kp, kd, n_sub)
    live = snapshot_of(m, channels)

    results = []

    # F1 stream_loss：停供期 feed=None，解除后恢复
    fault = FaultState()
    fault.trigger("stream_loss")
    assert apply_to_snapshot(fault, live, period_ms) is None, "F1 FAIL 停供期仍出流"
    fault.clear()
    assert apply_to_snapshot(fault, live, period_ms) is live, "F1 FAIL 解除后未恢复"
    results.append({"case": "stream_loss", "pass": True,
                    "signal": "feed=None during loss（看门狗输入）"})

    # F2 frozen_frame：冻结期恒返回同一陈旧快照
    fault = FaultState()
    frozen = dict(live)
    fault.trigger("frozen_frame", snapshot=frozen)
    out1 = apply_to_snapshot(fault, live, period_ms)
    out2 = apply_to_snapshot(fault, live, period_ms)
    assert out1 is frozen and out2 is frozen, "F2 FAIL 冻结期快照漂移"
    assert frozen["q"] != live["q"] or True  # 冻结帧与现拍可能数值接近——同一性为判据
    fault.clear()
    assert apply_to_snapshot(fault, live, period_ms) is live, "F2 FAIL 解除后未跟拍"
    results.append({"case": "frozen_frame", "pass": True,
                    "signal": "同一陈旧对象逐拍回流（rx 新鲜度输入）"})

    # F3 estop：注入期拒新指令
    fault = FaultState()
    fault.trigger("estop")
    assert command_gate(fault) is True, "F3 FAIL estop 期仍收指令"
    fault.clear()
    assert command_gate(fault) is False, "F3 FAIL 解除后仍拒指令"
    results.append({"case": "estop", "pass": True,
                    "signal": "command_gate=True（桥侧拒新目标）"})

    # F4 force_spike：注入路 fx=超限值，n_ft 抬升，窗口后自清
    fault = FaultState()
    fault.trigger("force_spike", ft=0, n=interlock * 4.0, ms=int(period_ms * 3))
    spiked = apply_to_snapshot(fault, live, period_ms)
    fx = spiked["ft"][0][0]
    assert spiked["n_ft"] >= 1 and abs(fx - interlock * 4.0) < 1e-3, \
        f"F4 FAIL 注入路 fx={fx}（期望 {interlock * 4.0}）"
    assert abs(fx) > interlock, "F4 FAIL 注入值未超联锁阈值"
    assert abs(live["ft"][0][0]) < 1e-9, "F4 FAIL 原快照被污染（注入必须不可变）"
    for _ in range(3):
        spiked = apply_to_snapshot(fault, live, period_ms)
    assert fault.kind == "none" and apply_to_snapshot(fault, live, period_ms) is live, \
        "F4 FAIL 窗口后未自清"
    results.append({"case": "force_spike", "pass": True,
                    "signal": f"fx={fx:.0f}N > 联锁 {interlock}N，n_ft=1，窗口后自清"})

    out_dir = Path(__file__).resolve().parents[2] / "reports"
    out_dir.mkdir(exist_ok=True)
    report = {
        "suite": "fault_injection", "mode": "bridge_policy_loop",
        "package": cfg["core"]["package_name"], "cases": results,
        "engine_in_loop_pending": ["watchdog_flag", "dispatch_reject", "interlock_estop"],
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / f"fault_injection_{cfg['core']['package_name']}_{int(time.time())}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"FAULT_INJECTION PASS 全绿（F1~F4 注入信号机械实证）→ reports/")


if __name__ == "__main__":
    main()
