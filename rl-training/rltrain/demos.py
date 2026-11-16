# -*- coding: utf-8 -*-
"""演示数据录制/加载(il_pretrain 的数据面)。

格式(demos/<name>.npz):
  obs   [T, obs_dim]  float32——契约观测(装配器产物)
  act   [T, J]        float32——tanh 域动作(环境端 neutral+a·scale)
  task  str           任务名
  src   str           来源(policy=导出 ONNX / scripted=脚本专家)
用途: 模仿学习预训练(algos/il_pretrain.yaml)与快脑蒸馏的数据面。
"""
from __future__ import annotations

import os
import time

import numpy as np


def record(env, policy_fn, steps: int, out_path: str, src: str = "policy") -> str:
    """录制 (obs, act) 轨迹对。policy_fn(obs[N,D]) -> act[N,J](batch 逐行亦可)。"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    O, A = [], []
    obs = env.reset()
    for _ in range(steps):
        a = np.asarray(policy_fn(obs), dtype=np.float32)
        if a.shape != obs.shape[:1] + a.shape[1:]:
            raise ValueError(f"策略输出 {a.shape} 与环境批 {obs.shape[0]} 不一致")
        O.append(obs.copy())
        A.append(a.copy())
        obs, _, done, infos = env.step(a)
        for e, inf in enumerate(infos):
            if done[e] and "final_obs" in inf:
                obs[e] = inf["final_obs"]
    np.savez_compressed(out_path, obs=np.concatenate(O, 0).astype(np.float32),
                        act=np.concatenate(A, 0).astype(np.float32),
                        task=np.asarray(env.task["name"]), src=np.asarray(src),
                        ts=np.asarray(time.strftime("%Y%m%d-%H%M%S")))
    n = sum(o.shape[0] for o in O)
    print(f"[demos] 录制 {n} 对 → {out_path}")
    return out_path


def load(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    return {"obs": d["obs"], "act": d["act"],
            "task": str(d["task"]), "src": str(d["src"])}
