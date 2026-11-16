# -*- coding: utf-8 -*-
"""T3 本体配置与参数化生成器(同源解析/自由底座/五指手/物体/地形)。"""
from __future__ import annotations

import mujoco

from rltrain.g1_cfg import G1Cfg, build_free_base_mjcf
from rltrain.mjcf_gen import generate


def _loads(xml: str) -> None:
    m = mujoco.MjModel.from_xml_path(xml)   # 解析失败即抛


def run(tmp: str) -> None:
    cfg = G1Cfg()
    assert cfg.j == 29 and len(cfg.joints) == 29
    assert cfg.kp > 0 and cfg.kd > 0
    from rltrain.contract import neutral_in_soft_limits
    assert not neutral_in_soft_limits(cfg.neutral, cfg.limits), "包中立位须全在软限位内"

    free = build_free_base_mjcf()
    _loads(free)
    assert "freejoint" in open(free, encoding="utf-8").read()

    p1, e1 = generate()
    _loads(p1)
    assert e1["j"] == 29

    p2, e2 = generate(hands=True, objects=True, press_wall=True, terrain=True)
    _loads(p2)
    assert e2["j"] == 45 and e2["hands"] and e2["terrain"], "五指手包 J=45"
    content = open(p2, encoding="utf-8").read()
    m2 = mujoco.MjModel.from_xml_path(p2)
    names = [mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m2.njnt)]
    finger = [n for n in names if n and "_hand_" in n]
    assert len(finger) == 16, f"双手 16 指关节,实际 {len(finger)}"
    assert "obj_can_geom" in content and "press_wall" in content and "terrain_0" in content
    print(f"  cfg/gen: 29 关节同源 + 自由底座 + 五指手(J=45)/物体/墙/地形 全部生成装载")
