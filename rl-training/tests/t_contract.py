# -*- coding: utf-8 -*-
"""T1 契约校验(宽度合法集/软限位/中立位)。"""
from __future__ import annotations

from rltrain.contract import (base_width, check_obs_dim, legal_obs_dims,
                              neutral_in_soft_limits, obs_width, soft_clip)


def run(tmp: str) -> None:
    assert base_width(29) == 97, "G1 基础宽度应为 97"
    assert obs_width(29, "goal") == 101 and obs_width(29, "contact", 2) == 109
    assert obs_width(45, "goal_contact", 2) == 161, "五指手包 goal_contact 应为 161"
    legal = legal_obs_dims(29, 3)
    assert 97 in legal and 109 in legal and 121 in legal, "k∈[0,8] 宽度集"
    assert legal_obs_dims(29, 3, with_goal=True) == {d + 4 for d in legal}

    check_obs_dim(97, 29, "base")
    check_obs_dim(149, 41, "goal_contact")
    for bad in [(98, 29, "base"), (150, 29, "goal_contact"), (101, 29, "base")]:
        try:
            check_obs_dim(*bad)
            raise AssertionError(f"非法宽度未被拒: {bad}")
        except ValueError:
            pass

    assert soft_clip(0.5, -1.0, 1.0) == 0.5, "域内不动"
    assert soft_clip(0.99, -1.0, 1.0) == 0.95, "软限位钳制 margin 0.05"
    bad = neutral_in_soft_limits({"a": 0.0}, {"a": (0.1, 1.0)})
    assert bad == ["a"], "中立位越软限位应检出"
    assert neutral_in_soft_limits({"a": 0.5}, {"a": (0.1, 1.0)}) == []
    print("  contract: 宽度公式/合法集/钳制/中立位 全部对齐")
