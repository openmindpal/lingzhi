# -*- coding: utf-8 -*-
"""契约常量与校验——训练侧镜像(唯一权威 = 灵智系统引擎契约 §2/§3/§4,随合作交付)。

锚点: 灵智系统引擎契约(随合作交付)
"""
from __future__ import annotations

import math

# ── 观测布局 ────────────────────────────────────────────────
IMU_DIM = 10          # quat_wxyz(4) + gyro(3) + accel(3)
GOAL_DIM = 4          # [x, y, z, f]
CONTACT_WRENCH = 6    # 每路 [fx,fy,fz,tx,ty,tz]
CONTACT_MAX_ROUTES = 8
SOFT_LIMIT_MARGIN_RAD = 0.05   # safety.soft_limit_margin_rad 缺省

FAMILIES = ("balance", "manipulation", "deform", "expression")


def base_width(j: int, b: int = 3) -> int:
    """基础观测宽度 = b·J + imu(10)。G1: 3·29+10 = 97。"""
    return b * j + IMU_DIM


def contact_width(k: int) -> int:
    return CONTACT_WRENCH * k


def obs_width(j: int, mode: str, k: int = 0, b: int = 3) -> int:
    """按观测模式计算宽度。mode ∈ base|goal|contact|goal_contact。"""
    w = base_width(j, b)
    if "contact" in mode:
        w += contact_width(k)
    if "goal" in mode:
        w += GOAL_DIM
    return w


def legal_obs_dims(j: int, b: int = 3, with_goal: bool = False, with_contact: bool = False) -> set[int]:
    """合法宽度集 = { b·J + 10 + 6k }(k∈[0,8]) 再按行声明 +goal(4)。"""
    dims = set()
    for k in range(0, CONTACT_MAX_ROUTES + 1):
        dims.add(base_width(j, b) + contact_width(k))
    if with_goal:
        dims = {d + GOAL_DIM for d in dims}
    return dims


def check_obs_dim(obs_dim: int, j: int, mode: str, k: int = 0, b: int = 3) -> None:
    """装载级校验:obs_dim 必须命中合法宽度集,否则拒收。"""
    with_goal = mode in ("goal", "goal_contact")
    with_contact = mode in ("contact", "goal_contact")
    legal = legal_obs_dims(j, b, with_goal=with_goal, with_contact=with_contact)
    if obs_dim not in legal:
        raise ValueError(f"obs_dim={obs_dim} 不在合法宽度集 {sorted(legal)} 内")


def soft_clip(x: float, lo: float, hi: float, margin: float = SOFT_LIMIT_MARGIN_RAD) -> float:
    """软限位钳制域 [lo+margin, hi−margin]。"""
    return min(max(x, lo + margin), hi - margin)


def neutral_in_soft_limits(neutral: dict[str, float], limits: dict[str, tuple[float, float]],
                           margin: float = SOFT_LIMIT_MARGIN_RAD) -> list[str]:
    """装载校验镜像:中立位 ∈ 软限位。返回违例通道列表(空 = 通过)。"""
    bad = []
    for name, q0 in neutral.items():
        lim = limits.get(name)
        if lim is None:
            continue
        lo, hi = lim
        if not (lo + margin <= q0 <= hi - margin):
            bad.append(name)
    return bad


def finite_goal(goal) -> bool:
    """引擎派发校验镜像:goal 全有限。"""
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in goal)
