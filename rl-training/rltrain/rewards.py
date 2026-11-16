# -*- coding: utf-8 -*-
"""任务奖励/终止/课程共享层——后端无关(MuJoCo 与 Isaac Lab 同函数同权重)。

口径:
  - build_ctx(task, neutral, control_hz) → 不可变任务上下文(权重/族/机制开关/波形参数)
  - compute_reward(ctx, m) → (reward, terms): m = 测量面字典(后端各自从仿真状态提取,
    键缺失 = 对应机制不激活——与原 env_mj 内联实现的分支语义逐项等价)
  - check_termination(ctx, m) → (done, flags): 摔倒/威胁命中/坠落/超时四路终止
  - sample_goal / rising push 课程采样同源

向量化: 全部逐元素 numpy 运算——m 传 (N,) 批数组即 GPU 批式求值(Isaac 后端),
       传标量即单环境求值(MuJoCo 后端);两后端奖励语义逐位同源,Sim2Sim 可对拍。
"""
from __future__ import annotations

import numpy as np

# 摔倒终止阈值(自由底座): 骨盆高度 / 四元数 w 分量(cos 半倾角)
FALL_Z = 0.65
UPRIGHT_COS_FALL = 0.35


def build_ctx(task: dict, neutral, control_hz: float) -> dict:
    """任务上下文(一次构建,整个训练只读)——权重/族/机制开关集中于此。"""
    inh = task.get("inhand") or {}
    return {
        "task": task,
        "w": task.get("reward", {}),
        "family": task["family"],
        "neutral": np.asarray(neutral, dtype=np.float64),
        "control_hz": float(control_hz),
        "gait": task.get("gait"),
        "airborne_ok": bool(task.get("airborne_ok")),
        "grasp": bool(task.get("grasp")),
        "goal_obs": "goal" in task["obs_mode"],
        "contact_obs": "contact" in task["obs_mode"],
        "pour": task.get("pour"),
        "intercept": task.get("intercept"),
        "bimanual": task.get("bimanual"),
        "strike": task.get("strike"),
        "inhand": task.get("inhand"),
        "needle": task.get("needle"),
        "wipe": task.get("wipe"),
        "dodge": task.get("dodge"),
        "morph": task.get("morph"),
        "base_amp": float(task.get("base_amp", 0.2)),
        "wave_hz": float(task.get("wave_hz", 1.0)),
        "wave_joint_idx": np.asarray(task["wave_joint_idx"], dtype=int)
                          if "wave_joint_idx" in task else None,
        "drop_z": float(inh.get("drop_z", 0.9)),
        "f_force_n": float(task.get("f_force_n", 20.0)),
        "max_ep_steps": 0,   # 由后端按 episode_s×control_hz 填
    }


def sample_goal(goal_box, f_range, rng) -> np.ndarray:
    """goal 课程采样: xyz ∈ 作业域包络均匀撒点,f ∈ 全域覆盖(§4.2)。"""
    bx = np.asarray(goal_box, dtype=np.float64)
    return np.array([rng.uniform(*bx[i]) for i in range(3)] +
                    [rng.uniform(*f_range)])


def _joints_sel(q, idx):
    """关节子集切片——标量 (J,) 取 q[idx],批 (N,J) 取 q[:, idx]。"""
    return q[idx] if np.ndim(q) == 1 else q[:, idx]


def compute_reward(ctx: dict, m: dict):
    """奖励 = 通用惩罚 + 族/机制项(与任务 yaml reward 段权重同源)。

    分支结构与原 env_mj 内联实现逐项等价(家族分支互斥、机制项独立叠加);
    m 键缺失 = 对应测量本回合不产生,该机制项跳过。
    """
    w = ctx["w"]
    rate, tau_n = m["action_rate"], m["tau_norm"]
    terms = {"action_rate": rate, "tau": tau_n}
    r = -w.get("action_rate", 0.0) * rate - w.get("tau", 0.0) * tau_n
    fam = ctx["family"]

    if ctx["gait"]:
        upright = m["upright"]
        terms["upright"] = upright
        r += w.get("alive", 0.5) + w.get("upright", 1.0) * upright
        ve = m["vel_err"]
        terms["vel_err"] = ve
        r += w.get("vel", 1.0) * (-ve)
    elif ctx["airborne_ok"]:
        terms["peak_z"] = m["peak_z"]
        terms["upright_end"] = m["upright_end"]
        r += w.get("peak", 2.0) * np.maximum(m["peak_z"] - 1.0, 0.0) \
            + w.get("upright", 1.0) * m["upright_end"]
    elif fam == "balance":
        upright = m["upright"]
        terms["upright"] = upright
        r += w.get("alive", 0.5) + w.get("upright", 1.0) * upright \
            + w.get("posture", 0.2) * (-np.mean(np.abs(m["q"] - ctx["neutral"]), axis=-1))
    elif fam == "manipulation":
        err = m.get("ee_err", 0.0)
        terms["ee_err"] = err
        r += w.get("ee_err", 1.0) * (-err) + w.get("progress", 5.0) * (m.get("prev_err", 0.0) - err)
        if ctx["grasp"]:
            lift = m["obj_lift"]
            gn = m["grasp_n"]
            terms["obj_lift"] = lift
            terms["grasp_n"] = gn
            r += w.get("lift", 10.0) * np.maximum(lift, 0.0) + w.get("grasp", 0.5) * np.minimum(gn, 3)
        if ctx["contact_obs"]:
            fmag = m["fmag"]
            f_ref = m["ep_f"] * ctx["f_force_n"]
            terms["f_err"] = np.abs(fmag - f_ref)
            r -= w.get("force", 0.01) * terms["f_err"]
    if ctx["pour"]:
        esc = m["water_escaped"]
        terms["water_escaped"] = esc
        r -= w.get("spill", 3.0) * esc
        err = m.get("ee_err", 0.0)
        r += w.get("ee_err", 0.5) * (-err)
    if ctx["intercept"]:
        dist = m["iball_dist"]
        terms["iball_dist"] = dist
        r += w.get("chase", 1.0) * (-dist)
        if m.get("timing_err") is not None:
            terms["timing_err"] = m["timing_err"]
            r += w.get("timing", 3.0) * (-m["timing_err"])
    if ctx["bimanual"]:
        ferr = m["f_left_err"]
        frac = m["stable_frac"]
        terms["f_left_err"] = ferr
        r += w.get("force", 0.05) * (-np.abs(ferr)) + w.get("stable", 1.0) * frac
    if ctx["strike"]:
        spd = m["ball_speed"]
        hits = m["ball_contacts"]
        terms["ball_speed"] = spd
        terms["ball_contacts"] = hits
        r += w.get("strike", 2.0) * np.minimum(spd, 3.0) + w.get("contact", 0.5) * np.minimum(hits, 2)
    if ctx["inhand"]:
        rot = m["pen_rot"]
        held = m["pen_held"]
        tgt = m["rot_target"]
        terms["pen_rot"] = rot
        r += w.get("rot", 1.5) * np.minimum(np.abs(rot), np.abs(tgt)) + w.get("held", 0.8) * held
    if ctx["needle"]:
        dep = m["insert_depth"]
        terms["insert_depth"] = dep
        r += w.get("depth", 20.0) * dep
    if ctx["wipe"]:
        tv = m["travel"]
        terms["travel"] = tv
        r += w.get("travel", 2.0) * tv
    elif fam == "deform":
        morph_err = np.mean(np.abs(m["q"] - m["morph_target"]), axis=-1)
        terms["morph_err"] = morph_err
        r += w.get("morph", 1.0) * (-morph_err) + w.get("hold", 0.2)
    elif fam == "expression":
        s = m["style"]
        amp = ctx["base_amp"] * (0.5 + s)
        ph = 2 * np.pi * ctx["wave_hz"] * m["t"] / ctx["control_hz"]
        idx = ctx["wave_joint_idx"]
        ref = ctx["neutral"].copy()
        ref[idx] += amp * np.sin(ph)
        style_err = np.mean(np.abs(_joints_sel(m["q"], idx) - ref[idx]), axis=-1)
        terms["style_err"] = style_err
        r += w.get("style", 1.0) * (-style_err)
    if ctx["dodge"]:
        terms["survive"] = 1.0
        r += w.get("survive", 1.0)
        if m.get("threat_hit") is not None:
            r -= w.get("hit_penalty", 5.0) * np.asarray(m["threat_hit"], dtype=np.float64)
    return r, terms


def check_termination(ctx: dict, m: dict):
    """终止四路: 超时 / 摔倒(高度或倾角) / 威胁命中 / 手持物坠落。

    root_z/quat_w = None 表示后端无自由底座(骨盆固定)——摔倒路不判。
    airborne_ok 任务腾空不判摔(翻转/跳跃)。返回 (done, flags 字典)。
    """
    done = np.asarray(m["t"]) >= ctx["max_ep_steps"]
    flags = {"timeout": done}
    fall = np.zeros_like(done, dtype=bool)
    if m.get("root_z") is not None and not ctx["airborne_ok"]:
        fall = (np.asarray(m["root_z"]) < FALL_Z) | (np.abs(np.asarray(m["quat_w"])) < UPRIGHT_COS_FALL)
    flags["fall"] = fall
    hit = np.zeros_like(done, dtype=bool)
    if m.get("threat_hit") is not None:
        hit = np.asarray(m["threat_hit"], dtype=bool)
    flags["threat_hit"] = hit
    dropped = np.zeros_like(done, dtype=bool)
    if ctx["inhand"]:
        dropped = np.asarray(m["pen_z"]) < ctx["drop_z"]
    flags["dropped"] = dropped
    return done | fall | hit | dropped, flags
