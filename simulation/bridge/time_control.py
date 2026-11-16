"""时间控制——确定性回归与时间加速（技术文档 §2.3 频率与确定性契约）。

  - 固定种子：物理侧噪声（若有）与注入序列可复现——同输入序列同仿真轨迹
  - 实时节拍：headless 默认按墙钟对齐 policy 拍（speed=1.0）
  - 时间加速：speed>1.0 时拍间不睡眠（仿真时间 > 实时——长时任务分钟级验证）
"""
from __future__ import annotations

import time


class TickClock:
    """策略拍节拍器：speed=1.0 按墙钟对齐；speed>1.0 全速跑（时间加速）。"""

    def __init__(self, period_s: float, speed: float = 1.0):
        self.period_s = period_s
        self.speed = speed
        self._next = time.monotonic()
        self.sim_time = 0.0
        self.ticks = 0

    def wait_tick(self) -> None:
        """等待下一拍（时间加速 = 不等待）；返回即一拍到点。"""
        if self.speed >= 1.0:
            self._next += self.period_s / self.speed
            now = time.monotonic()
            if self._next > now:
                time.sleep(self._next - now)
            else:
                # 落后于墙钟（处理慢于节拍）——重对齐，不追赶积压
                self._next = now
        self.sim_time += self.period_s
        self.ticks += 1
