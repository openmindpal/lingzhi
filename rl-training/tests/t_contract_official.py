# -*- coding: utf-8 -*-
"""T2b 官方规格同源机械断言——body-package.toml ↔ 官方 MJCF(g1_29dof_rev_1_0)。

权威分层中包声明关节名/序/限位为单一来源,本测试把"与官方模型同源"固化为
机械断言(限位按官方原值回填后,恒等关系成立):
  1. 关节集一致(29,无多无少)
  2. 逐关节限位精确相等(容差 1e-3)
  3. 中立位全部落在软限位内(margin 0.05,镜像 G1Cfg._validate)
纯文本/toml 解析,零 mujoco 依赖——基础契约门的一部分。
"""
from __future__ import annotations

import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rltrain._compat import tomllib
from rltrain.g1_cfg import PKG_TOML

OFFICIAL = os.path.join(os.path.dirname(_HERE), "envs", "g1", "assets", "official", "g1.xml")
MARGIN = 0.05
TOL = 1e-3


def run(tmp: str) -> None:
    _ = tmp
    assert os.path.isfile(OFFICIAL), f"官方资产缺失: {OFFICIAL}(base=official 的前提)"
    xml = open(OFFICIAL, encoding="utf-8").read()
    mj = {m[0]: (float(m[1]), float(m[2])) for m in re.findall(
        r'<joint name="([a-z_]+)"[^>]*?range="([-0-9.e]+) ([-0-9.e]+)"', xml)}
    assert len(mj) == 29, f"官方模型铰链数 {len(mj)} ≠ 29"

    pkg = tomllib.load(open(PKG_TOML, "rb"))
    joints = pkg["profile"]["joints"]
    neutral = pkg["calibration"]["neutral"]

    # 1. 关节集一致
    missing = sorted(set(mj) - set(joints))
    extra = sorted(set(joints) - set(mj))
    assert not missing and not extra, f"关节集不一致: 缺{missing} 多{extra}"

    # 2. 逐关节限位精确相等(官方原值回填)
    diffs = []
    for n, (lo, hi) in mj.items():
        p_lo, p_hi = float(joints[n]["limits"]["min"]), float(joints[n]["limits"]["max"])
        if abs(p_lo - lo) > TOL or abs(p_hi - hi) > TOL:
            diffs.append((n, (p_lo, p_hi), (lo, hi)))
    assert not diffs, f"限位与官方模型不符: {diffs[:4]}"

    # 3. 中立位 ∈ 软限位(镜像 G1Cfg._validate——契约装载语义)
    bad = [(n, v) for n, v in neutral.items()
           if not (mj[n][0] + MARGIN <= float(v) <= mj[n][1] - MARGIN)]
    assert not bad, f"中立位越出软限位: {bad}"

    print(f"  contract_official: 关节集/限位/中立位 与官方模型逐项一致 "
          f"(29 关节, margin={MARGIN})")


if __name__ == "__main__":
    run(os.path.join(_HERE, ".tmp"))
