# -*- coding: utf-8 -*-
"""T8 GPU 契约(无 GPU 可跑)——奖励/终止共享层 + isaaclab 后端诚实门。

覆盖:
  1. rewards.build_ctx 全任务 yaml(16 策略任务 + gait)可构建
  2. 合成测量 → compute_reward 全任务有限性(批数组与标量双口径)
  3. check_termination 终止真值表(超时/摔倒高度/摔倒倾角/airborne/威胁/坠落)
  4. sample_goal 界内与可复现
  5. isaaclab_env 模块无 Isaac 环境可导入;构造期 SystemExit 诚实报错(不降级)
  6. env_mj 测量面 ↔ 共享层接线(真实环境一步:测量键齐全 + 终止一致)
"""
from __future__ import annotations

import importlib

import numpy as np

from rltrain.g1_cfg import G1Cfg
from rltrain.tasks import load_task
from rltrain import rewards

TASKS = ["family_balance", "family_gait", "family_reach", "family_reach_contact",
         "family_deform", "family_expression", "family_grasp", "family_athletic",
         "family_dodge", "family_recover", "family_soft", "family_pour",
         "family_needle", "family_bimanual", "family_inhand", "family_intercept",
         "family_strike"]


def _full_measure(j: int, n: int | None = None) -> dict:
    """合成测量面:全部机制键齐备(缺失键 = 机制不激活,此处测全激活路径)。"""
    def v(fill=0.5):
        return np.full(n, fill) if n else fill
    q = np.full((n, j), 0.1) if n else np.full(j, 0.1)
    return {"action_rate": v(0.02), "tau_norm": v(0.01), "t": v(3), "q": q,
            "upright": v(1.0), "root_z": v(1.0), "quat_w": v(1.0),
            "vel_err": v(0.1), "peak_z": v(1.2), "upright_end": v(1.0),
            "ee_err": v(0.05), "prev_err": v(0.06), "obj_lift": v(0.02), "grasp_n": v(1),
            "fmag": v(3.0), "ep_f": v(0.5), "water_escaped": v(0),
            "iball_dist": v(0.2), "timing_err": v(0.03), "f_left_err": v(1.0),
            "stable_frac": v(0.9), "ball_speed": v(1.0), "ball_contacts": v(1),
            "pen_rot": v(0.5), "pen_held": v(1.0), "rot_target": v(2.0), "pen_z": v(1.0),
            "insert_depth": v(0.001), "travel": v(0.05),
            "morph_target": q.copy(), "style": v(0.5), "threat_hit": v(0)}


def run(tmp: str) -> None:
    cfg = G1Cfg()

    # ── 1/2: 全任务 ctx 构建 + 奖励有限性(标量与批数组双口径) ──
    ctxs = {}
    for tn in TASKS:
        t = load_task(tn)
        ctxs[tn] = rewards.build_ctx(t, cfg.neutral_array(), float(t["control_hz"]))
        ctxs[tn]["max_ep_steps"] = int(t["episode_s"] * t["control_hz"])
        r_scalar, terms = rewards.compute_reward(ctxs[tn], _full_measure(cfg.j))
        assert np.isfinite(r_scalar).all(), f"{tn}: 标量奖励非有限 {r_scalar}"
        r_batch, _ = rewards.compute_reward(ctxs[tn], _full_measure(cfg.j, n=4))
        assert np.shape(r_batch) == (4,) and np.isfinite(r_batch).all(), f"{tn}: 批奖励口径错误"
        assert "action_rate" in terms

    # ── 3: 终止真值表 ──
    ctx = ctxs["family_balance"]
    m = _full_measure(cfg.j)
    done, flags = rewards.check_termination(ctx, m)
    assert not done and not flags["fall"], "正常测量不应终止"

    m_t = _full_measure(cfg.j); m_t["t"] = ctx["max_ep_steps"]
    done, flags = rewards.check_termination(ctx, m_t)
    assert done and flags["timeout"], "超时应终止"

    m_z = _full_measure(cfg.j); m_z["root_z"] = 0.5
    done, flags = rewards.check_termination(ctx, m_z)
    assert done and flags["fall"] and not flags["timeout"], "骨盆低于 FALL_Z 应判摔"

    m_w = _full_measure(cfg.j); m_w["quat_w"] = 0.2
    done, flags = rewards.check_termination(ctx, m_w)
    assert done and flags["fall"], "四元数 w 低于倾角阈应判摔"

    m_air = _full_measure(cfg.j); m_air["root_z"] = 0.4; m_air["quat_w"] = 0.1
    done, flags = rewards.check_termination(ctxs["family_athletic"], m_air)
    assert not flags["fall"], "airborne_ok 任务腾空不判摔"

    done, flags = rewards.check_termination(ctx, {**_full_measure(cfg.j), "root_z": None})
    assert not flags["fall"], "无自由底座(root_z=None)不判摔"

    m_hit = _full_measure(cfg.j); m_hit["threat_hit"] = 1
    done, flags = rewards.check_termination(ctxs["family_dodge"], m_hit)
    assert done and flags["threat_hit"], "威胁命中应终止"

    m_drop = _full_measure(cfg.j); m_drop["pen_z"] = 0.5
    done, flags = rewards.check_termination(ctxs["family_inhand"], m_drop)
    assert done and flags["dropped"], "手持物坠落应终止"

    # ── 4: goal 采样界内/可复现 ──
    t = load_task("family_reach")
    r1, r2 = np.random.RandomState(7), np.random.RandomState(7)
    g1 = rewards.sample_goal(t["goal_box"], t.get("goal_f_range", [0.0, 1.0]), r1)
    g2 = rewards.sample_goal(t["goal_box"], t.get("goal_f_range", [0.0, 1.0]), r2)
    assert g1.shape == (4,) and np.allclose(g1, g2), "同种子 goal 采样应可复现"
    bx = np.asarray(t["goal_box"], dtype=np.float64)
    for i in range(3):
        assert bx[i][0] <= g1[i] <= bx[i][1], f"goal[{i}] 越界"
    assert 0.0 <= g1[3] <= 1.0

    # ── 5: isaaclab 诚实门(无 Isaac 环境 = 构造期 SystemExit,不静默降级) ──
    mod = importlib.import_module("rltrain.isaaclab_env")   # 模块本体必须无 Isaac 可导入
    try:
        import isaaclab  # noqa: F401
        has_isaac = True
    except Exception:
        has_isaac = False
    if not has_isaac:
        try:
            mod.IsaacLabVecEnv(None, {"obs_mode": "base", "family": "balance"}, 2)
            raise AssertionError("无 Isaac 环境构造应 SystemExit(诚实报错)")
        except SystemExit:
            pass

    # ── 6: env_mj 测量面 ↔ 共享层接线(真实环境一步;记录子类避免二次测量失真) ──
    from rltrain.env_mj import G1VecEnv

    class _RecordingEnv(G1VecEnv):
        def _measure(self, e, a, info):
            self._last_m = super()._measure(e, a, info)
            return self._last_m

    for tn in ("family_balance", "family_recover", "family_reach"):
        env = _RecordingEnv(cfg, load_task(tn), 1, seed=3)
        env.reset()
        _o, r, done, _infos = env.step(np.zeros((1, env.j)))
        m = env._last_m
        for key in ("action_rate", "tau_norm", "t", "q", "upright", "root_z", "quat_w"):
            assert key in m, f"{tn}: 测量面缺 {key}"
        r2, _ = rewards.compute_reward(env._ctx, m)
        assert np.isfinite(r2) and abs(float(r2) - float(r[0])) < 1e-6, \
            f"{tn}: 共享层奖励与环境输出不一致(接线断裂;差 {abs(float(r2)-float(r[0])):.2e})"
        d2, _ = rewards.check_termination(env._ctx, m)
        assert bool(np.asarray(d2)) == bool(done[0]), f"{tn}: 共享层终止与环境输出不一致"

    print("  gpu: 17 任务共享奖励/终止层全构建 + 终止真值表 + isaaclab 诚实门 + env 接线一致")
