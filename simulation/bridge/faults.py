"""故障注入器——安全分层验证的数据通道（technical spec §3.4，判据与physical robot同语义）。

注入通道全部在桥接器（虚拟机器人）侧实施——runtime与physical robot同路径感知故障：
  - stream_loss    本进程停止供流 → runtimewatchdog → safety.stream_loss
  - frozen_frame   连接在数据死 → runtime rx 新鲜度/快照检测捕获
  - estop          桥侧忽略指令（物理悬停）——runtime侧急停闩锁独立验证
  - force_spike    力觉注入超限值 → runtimeforce interlock（立即收场+estop）

注入由 YAML 套件（suites/fault_injection/*.yaml）声明参数，经 --fault CLI 或
HTTP 注入端点触发；本模块只承载机制，场景 = 数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FaultState:
    """当前注入状态（server 主循环每拍消费）。"""

    kind: str = "none"            # none | stream_loss | frozen_frame | estop | force_spike
    # frozen_frame：沿用此刻快照不再更新
    frozen: dict | None = None
    # force_spike：在第 spike_ft 路 fx 注入 spike_n（牛顿），持续 spike_ms 毫秒
    spike_ft: int = 0
    spike_n: float = 0.0
    spike_until_tick: int = 0
    tick: int = 0
    log: list[str] = field(default_factory=list)

    def trigger(self, kind: str, **kw) -> str:
        self.kind = kind
        self.tick = 0
        if kind == "frozen_frame":
            self.frozen = kw.get("snapshot")
        elif kind == "force_spike":
            self.spike_ft = int(kw.get("ft", 0))
            self.spike_n = float(kw.get("n", 0.0))
            self.spike_until_tick = int(kw.get("ms", 500))
        self.log.append(f"trigger {kind} {kw}")
        return kind

    def clear(self) -> str:
        prev = self.kind
        self.kind = "none"
        self.frozen = None
        self.log.append(f"clear {prev}")
        return prev


def apply_to_snapshot(fault: FaultState, snap: dict, period_ms: float) -> dict | None:
    """按注入状态处理传感快照：返回 None = 本拍停供（断流语义）。"""
    fault.tick += 1
    if fault.kind == "stream_loss":
        return None
    if fault.kind == "frozen_frame":
        return fault.frozen               # 沿用旧帧——新鲜度检测捕获
    if fault.kind == "force_spike":
        if fault.tick * period_ms <= fault.spike_until_tick:
            snap = dict(snap)
            ft = [list(row) for row in snap["ft"]]
            if 0 <= fault.spike_ft < 8:
                ft[fault.spike_ft][0] = fault.spike_n      # fx 注入
                snap["n_ft"] = max(snap["n_ft"], fault.spike_ft + 1)
            snap["ft"] = ft
            return snap
        fault.clear()
        return snap
    return snap                          # none / estop：状态照常回流（estop 只吃指令）


def command_gate(fault: FaultState) -> bool:
    """estop 注入 = 桥侧忽略新指令（ True=忽略）。runtime侧闩锁语义独立验证。"""
    return fault.kind == "estop"
