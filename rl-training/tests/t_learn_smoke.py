# -*- coding: utf-8 -*-
"""T4 学习冒烟(PPO 最小迭代 / SAC 最小步 / 快脑 BC)——产物齐全即过。"""
from __future__ import annotations

import argparse
import os

from rltrain.g1_cfg import G1Cfg
from rltrain.tasks import load_task


def _args(**kw) -> argparse.Namespace:
    d = dict(task="", algo="ppo", backend="mujoco", device="cpu", envs=2, iters=1,
             steps=6, lr="3e-4", hidden=[32, 32], seed=0, eval_every=1,
             push_curriculum=False, sac_warmup=8, batch=32, bc_samples=64, bc_epochs=2,
             out="")
    d.update(kw)
    return argparse.Namespace(**d)


def run(tmp: str) -> None:
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "train_mod", os.path.join(os.path.dirname(here), "scripts", "train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    cfg = G1Cfg()
    # PPO balance(最优保存+manifest)
    rd = os.path.join(tmp, "t_balance")
    os.makedirs(rd, exist_ok=True)
    train.train_policy(load_task("family_balance"), cfg,
                       _args(task="family_balance", out=rd), rd)
    assert os.path.exists(os.path.join(rd, "ckpt.pt"))
    assert os.path.exists(os.path.join(rd, "ckpt_best.pt")), "最优检查点应保存"
    assert os.path.exists(os.path.join(rd, "manifest.json"))
    import json
    mf = json.load(open(os.path.join(rd, "manifest.json"), encoding="utf-8"))
    assert mf["j"] == 29 and mf["obs_dim"] == 97 and mf.get("best")

    # SAC reach(极小步)
    rd2 = os.path.join(tmp, "t_sac")
    os.makedirs(rd2, exist_ok=True)
    train.train_sac(load_task("family_reach"), cfg,
                    _args(task="family_reach", algo="sac", steps=24, out=rd2), rd2)
    assert os.path.exists(os.path.join(rd2, "ckpt.pt"))

    # 快脑 BC
    rd3 = os.path.join(tmp, "t_fb")
    os.makedirs(rd3, exist_ok=True)
    train.train_fastbrain(load_task("fastbrain_tactician"), cfg,
                          _args(task="fastbrain_tactician", out=rd3), rd3)
    assert os.path.exists(os.path.join(rd3, "manifest.json"))
    print("  learn: PPO(含 best)/SAC/快脑BC 产物齐全")
