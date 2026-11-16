# -*- coding: utf-8 -*-
"""T2 观测装配器(逐位宽度/白名单/非有限拒绝)。"""
from __future__ import annotations

import numpy as np

from rltrain.obs_assembler import ObsAssembler


def _good(j: int):
    return (np.zeros(j, dtype=np.float32), np.zeros(j, dtype=np.float32),
            np.zeros(j, dtype=np.float32), np.zeros(10, dtype=np.float32))


def run(tmp: str) -> None:
    a = ObsAssembler(29, "base")
    o = a.assemble(*_good(29))
    assert o.shape == (97,) and o.dtype == np.float32

    assert ObsAssembler(29, "goal").dim == 101
    assert ObsAssembler(29, "contact", 2).dim == 109
    assert ObsAssembler(45, "goal_contact", 2).dim == 161

    g = ObsAssembler(29, "goal")
    q, dq, tau, imu = _good(29)
    assert g.assemble(q, dq, tau, imu, None, np.zeros(4)).shape == (101,)

    # 白名单: 特权信息拒绝
    try:
        a.assemble_from_dict({"q": np.zeros(29), "dq": np.zeros(29), "tau": np.zeros(29),
                              "imu": np.zeros(10), "obj_true_pos": np.zeros(3)})
        raise AssertionError("特权信息未被拒")
    except ValueError as e:
        assert "特权" in str(e) or "白名单" in str(e)

    # 非有限值拒绝(传感故障诚实暴露)
    bad = _good(29)
    bad[3][0] = np.nan
    try:
        a.assemble(*bad)
        raise AssertionError("非有限值未被拒")
    except ValueError:
        pass

    # 宽度不符拒绝
    try:
        a.assemble(*_good(28))
        raise AssertionError("宽度不符未被拒")
    except ValueError:
        pass
    print("  assembler: 4 模式宽度/白名单/非有限/宽度不符 全部拦截")
