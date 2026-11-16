#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""演示数据录制 CLI(il_pretrain 数据面)。

用法:
  # 从已导出策略录制
  python scripts/record_demo.py --task family_reach --policy runs/<id>/exports/reach_goal.onnx --steps 200 --out demos/reach.npz
  # 无策略 = 中性保持脚本(占位数据面——真示教来自遥操作接入点)
  python scripts/record_demo.py --task family_reach --steps 100 --out demos/hold.npz
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np

from rltrain.demos import record
from rltrain.g1_cfg import G1Cfg
from rltrain.tasks import load_task
from rltrain.env_mj import G1VecEnv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True)
    ap.add_argument("--policy", default=None, help="导出 ONNX 路径;缺省 = 中性保持脚本")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--envs", type=int, default=2)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    task = load_task(args.task)
    cfg = G1Cfg()
    env = G1VecEnv(cfg, task, args.envs, seed=0)

    if args.policy:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name

        def policy_fn(obs):
            # 契约 IO = [1, dim]——逐行推理
            return np.stack([sess.run(None, {iname: obs[i:i + 1].astype(np.float32)})[0][0]
                             for i in range(obs.shape[0])])
        src = "policy"
    else:
        def policy_fn(obs):
            return np.zeros((obs.shape[0], env.j), dtype=np.float32)
        src = "scripted"

    record(env, policy_fn, args.steps, args.out, src=src)


if __name__ == "__main__":
    main()
