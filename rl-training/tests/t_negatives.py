# -*- coding: utf-8 -*-
"""T6 负例注入(装配器/导出/打包/装载门的拒绝行为)。"""
from __future__ import annotations

import numpy as np

from rltrain.obs_assembler import ObsAssembler


def run(tmp: str) -> None:
    # 装配器: goal 模式缺 goal 块
    g = ObsAssembler(29, "goal")
    try:
        g.assemble(np.zeros(29), np.zeros(29), np.zeros(29), np.zeros(10), None, None)
        raise AssertionError("goal 缺块应拒")
    except ValueError:
        pass
    # contact 模式缺块(诚实缺帧语义)
    c = ObsAssembler(29, "contact", 2)
    try:
        c.assemble(np.zeros(29), np.zeros(29), np.zeros(29), np.zeros(10), None, None)
        raise AssertionError("contact 缺块应拒")
    except ValueError:
        pass
    # 未知模式
    try:
        ObsAssembler(29, "vision_e2e")
        raise AssertionError("未知观测模式应拒(契约外模态)")
    except ValueError:
        pass
    print("  negatives: 缺块/未知模态/契约外键 全部拒绝(诚实语义在环)")
