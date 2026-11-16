# -*- coding: utf-8 -*-
"""T9 训练质量门(收敛回归测量)——标准任务固定种子冒烟训练,奖励不得低于基线 floor。

门的语义(诚实边界): 冒烟级——防"改奖励/环境/算法把训练质量改崩"的回归,
不是"训得好"的证明。floor 方向: 正奖励 floor=基线×0.6(允许 40% 劣化),
负奖励 floor=基线×1.5(劣化方向=更负)。基线刷新: python tests/t_quality.py --rebase
(在参照机上重测并覆写——跨机器浮点漂移时刷新)。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

BASELINE = os.path.join(_HERE, "quality_baseline.json")


def _smoke_eval(task_name: str, seed: int = 0, iters: int = 2, steps: int = 24) -> float:
    """确定性冒烟训练: 固定种子/结构,返回平均回合奖励(训练质量代理量)。"""
    import torch
    torch.set_num_threads(1)   # 确定性:CPU 多线程浮点求和顺序不确定,被训练动力学放大成不可复现

    from rltrain.g1_cfg import G1Cfg
    from rltrain.env_mj import G1VecEnv
    from rltrain.tasks import load_task
    from rltrain import ppo as P

    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = G1Cfg()
    env = G1VecEnv(cfg, load_task(task_name), 2, seed=seed)
    model = P.ActorCritic(env.assembler.dim, env.j, hidden=(64, 64))
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    norm = P.RunningNorm(env.assembler.dim)
    obs = env.reset()
    rets = []
    for _ in range(iters):
        ep = np.zeros(2)
        for _t in range(steps):
            with torch.no_grad():
                mu, _ = model.forward(torch.as_tensor(norm.normalize(obs), dtype=torch.float32))
                a = torch.tanh(mu).numpy()
            obs, r, done, infos = env.step(a)
            norm.update(obs)
            ep += r
            for e, inf in enumerate(infos):
                if done[e] and "final_obs" in inf:
                    obs[e] = inf["final_obs"]
        rets.append(ep.mean())
    return float(np.mean(rets))


def run(tmp: str) -> None:
    rebase = "--rebase" in sys.argv   # 参照机上重测并覆写基线

    import torch
    torch.set_num_threads(1)   # 确定性:CPU 多线程浮点求和顺序不确定,被训练动力学放大成不可复现

    from rltrain.g1_cfg import G1Cfg
    from rltrain.env_mj import G1VecEnv
    from rltrain.tasks import load_task
    from rltrain import ppo as P

    base = json.load(open(BASELINE, encoding="utf-8"))
    cfg = G1Cfg()
    results = {}
    for tn, spec in base.items():
        if tn.startswith("_") or not isinstance(spec, dict):
            continue   # 元数据键(_comment/machine)
        torch.manual_seed(spec["seed"])
        np.random.seed(spec["seed"])
        env = G1VecEnv(cfg, load_task(tn), 2, seed=spec["seed"])
        model = P.ActorCritic(env.assembler.dim, env.j, hidden=(64, 64))
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
        norm = P.RunningNorm(env.assembler.dim)
        obs = env.reset()
        ep = np.zeros(2)
        for _it in range(spec["iters"]):
            for _t in range(spec["steps"]):
                with torch.no_grad():
                    mu, _ = model.forward(torch.as_tensor(norm.normalize(obs), dtype=torch.float32))
                    a = torch.tanh(mu).numpy()
                obs, r, done, infos = env.step(a)
                norm.update(obs)
                ep += r
                for e, inf in enumerate(infos):
                    if done[e] and "final_obs" in inf:
                        obs[e] = inf["final_obs"]
        measured = float(ep.mean())
        floor = spec["floor"]
        ok = measured >= floor
        results[tn] = {"measured": round(measured, 3), "floor": floor, "pass": ok}
        print(f"  [{'PASS' if ok else 'FAIL'}] {tn}: measured={measured:.2f} ≥ floor={floor} "
              f"({'基线 ' + str(spec['ep_ret_mean'])})")

    if rebase:
        for tn, r in results.items():
            base[tn]["ep_ret_mean"] = r["measured"]
            r0 = r["measured"]
            base[tn]["floor"] = round(r0 * (0.6 if r0 >= 0 else 1.5), 3)
        json.dump(base, open(BASELINE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  基线已刷新 → {BASELINE}")
        return

    bad = [tn for tn, r in results.items() if not r["pass"]]
    assert not bad, f"训练质量门未过: {bad}——奖励崩坏回归(检查奖励函数/环境/算法改动)"
    print("  quality: 训练质量门全过(冒烟级收敛基线内)")


if __name__ == "__main__":
    run(os.path.join(_HERE, ".tmp"))
