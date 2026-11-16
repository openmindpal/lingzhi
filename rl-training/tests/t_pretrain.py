# -*- coding: utf-8 -*-
"""T10 il_pretrain BC 预训练适配——演示数据 → 策略热启动端到端。

流程: 中性保持示教录制(record 数据面) → BC 预训练(train --pretrain-demos)
→ 断言: 产物齐全 + BC loss 收敛(中性保持平凡可学,loss 应显著下降)。
诚实边界: 本测试验证数据面与适配链路;示教质量对策略质量的决定性
(真机遥操作示教)属生产范畴。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def run(tmp: str) -> None:
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "train_mod", os.path.join(os.path.dirname(here), "scripts", "train.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    from rltrain.g1_cfg import G1Cfg
    from rltrain.tasks import load_task
    from rltrain.env_mj import G1VecEnv
    from rltrain import demos

    cfg = G1Cfg()
    task = load_task("family_balance")
    env = G1VecEnv(cfg, task, 1, seed=5)

    # 中性保持示教(act=0 = neutral+a·0)——平凡可学,验证 BC 链路
    demo_path = os.path.join(tmp, "pretrain_demos.npz")
    demos.record(env, lambda obs: np.zeros((obs.shape[0], env.j), dtype=np.float32),
                 steps=120, out_path=demo_path, src="scripted")
    dm = demos.load(demo_path)
    assert dm["obs"].shape[1] == env.assembler.dim, "演示 obs 维度=契约宽度"
    assert dm["act"].shape[1] == env.j

    # BC 预训练 + 短 RL(复用 train_policy 全流程)
    rd = os.path.join(tmp, "t_pretrain_run")
    os.makedirs(rd, exist_ok=True)

    class A:
        task = "family_balance"; algo = "ppo"; backend = "mujoco"; device = "cpu"
        envs = 1; iters = 1; steps = 8; lr = "3e-4"; hidden = [32, 32]; seed = 5
        eval_every = 0; push_curriculum = False; pretrain_demos = demo_path
        bc_pretrain_epochs = 8; out = rd
    args = A()

    train.train_policy(task, cfg, args, rd)
    mf = json.load(open(os.path.join(rd, "manifest.json"), encoding="utf-8"))
    assert "pretrain" in mf, "manifest 缺 pretrain 段"
    pt = mf["pretrain"]
    assert pt["samples"] == 120 and pt["epochs"] == 8
    assert pt["bc_loss_last"] < 0.05, f"中性保持 BC 应收敛: {pt}"
    assert pt["bc_loss_last"] <= pt["bc_loss_first"] * 1.1 + 1e-6, "BC loss 应单调下降趋势"
    assert os.path.exists(os.path.join(rd, "ckpt.pt"))
    print(f"  pretrain: BC 预训练链路端到端 OK (samples={pt['samples']} "
          f"loss {pt['bc_loss_first']}→{pt['bc_loss_last']})")


if __name__ == "__main__":
    run(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp"))
