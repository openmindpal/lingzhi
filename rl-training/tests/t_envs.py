# -*- coding: utf-8 -*-
"""T3b 六任务环境(契约宽度/有限奖励/延迟与课程机制)。"""
from __future__ import annotations

import numpy as np

from rltrain.g1_cfg import G1Cfg
from rltrain.tasks import load_task
from rltrain.env_mj import G1VecEnv

EXPECT = {"family_balance": (29, 97), "family_reach": (29, 101),
          "family_reach_contact": (29, 109), "family_deform": (29, 97),
          "family_expression": (29, 97), "family_grasp": (45, 161),
          "family_athletic": (29, 97), "family_dodge": (29, 101),
          "family_recover": (29, 97), "family_soft": (29, 109),
          "family_pour": (29, 101), "family_needle": (29, 101),
          "family_bimanual": (29, 113), "family_inhand": (45, 161),
          "family_intercept": (29, 101), "family_strike": (29, 101)}


def run(tmp: str) -> None:
    cfg = G1Cfg()
    for tn, (j, dim) in EXPECT.items():
        t = load_task(tn)
        env = G1VecEnv(cfg, t, 2, seed=1)
        assert env.j == j and env.assembler.dim == dim, f"{tn}: J={env.j} dim={env.assembler.dim}"
        o = env.reset()
        assert o.shape == (2, dim) and np.isfinite(o).all()
        a = np.zeros((2, env.j))
        o2, r, d, infos = env.step(a)
        assert np.isfinite(r).all() and o2.shape == (2, dim)
        # 电机延迟机制在环(20ms @50Hz = 1 拍——规格域上界)
        env2 = G1VecEnv(cfg, {**t, "randomize": {**t.get("randomize", {}), "motor_delay_ms": [20, 20]}},
                        1, seed=2)
        env2.reset()
        assert env2.ep_delay[0] == 1, f"20ms 延迟应为 1 拍,实际 {env2.ep_delay[0]}"
        # 扰动课程
        env2.set_curriculum(0.5)
        assert env2.curriculum_frac == 0.5
    print("  envs: 6 任务宽度逐位对齐 + 延迟/课程机制在环")
