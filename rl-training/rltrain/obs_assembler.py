# -*- coding: utf-8 -*-
"""观测装配器——按契约布局逐位拼装(白名单制,特权信息进不来)。

契约: contracts/obs_layout.md(唯一权威 = 灵智系统引擎契约 §2,随合作交付)。
输入只接受布局内的物理传感来源: q/dq/tau/imu/contact/goal——其余键一律拒绝。
"""
from __future__ import annotations

import numpy as np

from rltrain.contract import GOAL_DIM, IMU_DIM, CONTACT_WRENCH, obs_width


class ObsAssembler:
    """mode ∈ base | goal | contact(goal/contact 可叠加 → 'goal_contact')。

    k = 接触力路数(G1 双腕 = 2: 0=左腕 1=右腕,路序包文档化)。
    """

    def __init__(self, j: int, mode: str = "base", k: int = 0, b: int = 3):
        if mode not in ("base", "goal", "contact", "goal_contact"):
            raise ValueError(f"未知观测模式: {mode}")
        self.j, self.mode, self.k, self.b = j, mode, k, b
        self.dim = obs_width(j, mode, k, b)

    # ── 拼装(逐位 = 契约块序) ───────────────────────────────
    def assemble(self, q, dq, tau, imu, contact=None, goal=None) -> np.ndarray:
        blocks = [self._block("q", q, self.j), self._block("dq", dq, self.j),
                  self._block("tau", tau, self.j), self._block("imu", imu, IMU_DIM)]
        if "contact" in self.mode:
            if contact is None:
                raise ValueError("contact 模式必须提供接触力块(诚实缺帧语义 = 环境收场,不补零冒充)")
            blocks.append(self._block("contact", contact, CONTACT_WRENCH * self.k))
        if "goal" in self.mode:
            if goal is None:
                raise ValueError("goal 模式必须提供 goal 块")
            blocks.append(self._block("goal", goal, GOAL_DIM))
        obs = np.concatenate(blocks).astype(np.float32)
        if obs.shape[0] != self.dim:
            raise RuntimeError(f"装配宽度 {obs.shape[0]} ≠ 契约宽度 {self.dim}")
        return obs

    @staticmethod
    def _block(name: str, x, expected: int) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.shape[0] != expected:
            raise ValueError(f"观测块 {name} 宽度 {arr.shape[0]} ≠ {expected}")
        if not np.isfinite(arr).all():
            raise ValueError(f"观测块 {name} 含非有限值(传感故障须诚实暴露)")
        return arr

    # ── 特权信息白名单(装配器只接受这些键) ──────────────────
    ALLOWED_KEYS = {"q", "dq", "tau", "imu", "contact", "goal"}

    def assemble_from_dict(self, d: dict) -> np.ndarray:
        extra = set(d) - self.ALLOWED_KEYS
        if extra:
            raise ValueError(f"特权信息拒绝进观测: {sorted(extra)}(契约白名单外)")
        return self.assemble(d.get("q"), d.get("dq"), d.get("tau"), d.get("imu"),
                             d.get("contact"), d.get("goal"))
