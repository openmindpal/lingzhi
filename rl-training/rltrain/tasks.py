# -*- coding: utf-8 -*-
"""任务定义装载与校验(tasks/*.yaml)。"""
from __future__ import annotations

import os

import yaml

from rltrain.contract import FAMILIES

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tasks")


def load_task(name: str) -> dict:
    """按名装载 tasks/<name>.yaml,校验必填字段与契约一致性。"""
    path = os.path.join(TASKS_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"任务定义不存在: {path}")
    with open(path, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    _validate(t, path)
    return t


def _validate(t: dict, path: str) -> None:
    for key in ("kind", "family", "obs_mode", "control_hz", "episode_s"):
        if key not in t:
            raise ValueError(f"{path} 缺必填字段: {key}")
    kind = t["kind"]
    if kind == "fastbrain":
        if t.get("goal_stream") is None:
            raise ValueError(f"{path} fastbrain 任务缺 goal_stream")
        return
    if t["family"] not in FAMILIES:
        raise ValueError(f"{path} family={t['family']} 不在引擎枚举 {FAMILIES}")
    if t["obs_mode"] not in ("base", "goal", "contact", "goal_contact"):
        raise ValueError(f"{path} obs_mode 非法: {t['obs_mode']}")
    if t["obs_mode"] in ("goal", "goal_contact") and "goal_box" not in t:
        raise ValueError(f"{path} goal 条件化任务缺 goal_box(课程采样域)")
    rw = t.get("reward", {})
    if not rw:
        raise ValueError(f"{path} 缺 reward 权重表")


def policy_row_from_task(t: dict, weights: str, obs_dim: int, j: int, b: int = 3) -> dict:
    """由任务定义生成 [policies.rows] 行字段(与引擎 PolicyRowV1 对齐)。"""
    mode = t["obs_mode"]
    return {
        "name": t["row_name"],
        "family": t["family"],
        "weights": weights,
        "frequency_hz": float(t["control_hz"]),
        "priority": t.get("priority", "task"),
        "obs_dim": int(obs_dim),
        "goal_conditioned": "goal" in mode,
        "obs_contact": "contact" in mode,
    }
