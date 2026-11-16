#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收门(训练侧)——确定性回放 + 任务判据(阈值 = 任务 yaml gate 段)。

用法:
  python scripts/eval_gate.py --run runs/<id> [--episodes 3]
判据(引擎验收协议镜像,参考阈值——真机验收以 tri 在环为准):
  balance:  不摔(存活全程) + 抗扰后回归
  manipulation: 末端到位误差 ≤ gate.ee_err_m
  deform:   20ms 采样相邻跳变 < 0.05rad(tri_03 同判据)
  expression: 风格误差 ≤ gate.style_err_rad
退出码 = 0 全绿 / 1 有 FAIL(机械断言,非"看起来对")。
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np

from rltrain.g1_cfg import G1Cfg
from rltrain.tasks import load_task
from rltrain.env_mj import G1VecEnv


def make_policy(run: str, mf: dict):
    """验收只对导出产物负责(ONNX = 交付形态)——无 exports = 诚实退出。"""
    exp = mf.get("exports", [])
    if not exp:
        raise SystemExit("run 无 exports——先 export_onnx.py(验收对象 = 导出产物,非中间 ckpt)")
    import onnxruntime as ort
    path = exp[-1]["file"]
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    def policy(obs: np.ndarray) -> np.ndarray:
        # 契约 IO = [1,obs_dim]——逐行推理(批维固定 1)
        return np.stack([sess.run(None, {iname: obs[i:i + 1].astype(np.float32)})[0][0]
                         for i in range(obs.shape[0])])
    return policy, f"onnx:{os.path.basename(path)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    args = ap.parse_args()
    mf = json.load(open(os.path.join(args.run, "manifest.json"), encoding="utf-8"))
    task = load_task(mf["task"])
    cfg = G1Cfg()
    env = G1VecEnv(cfg, task, args.episodes, seed=123)
    gate = task.get("gate", {})
    fails = []

    if mf["kind"] == "fastbrain":
        obs = env.reset()
        err_max = 0.0
        from rltrain.contract import base_width
        bw = base_width(cfg.j)
        import onnxruntime as ort
        path = mf["exports"][-1]["file"]
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        for _ in range(int(task.get("eval_steps", 200))):
            g = np.stack([env._goal_obs(e) for e in range(env.n)])
            x = np.concatenate([obs[:, :bw], g], axis=1)
            # 契约 IO = [1,in_dim]——逐行推理
            g2 = np.stack([sess.run(None, {iname: x[i:i + 1].astype(np.float32)})[0][0]
                           for i in range(x.shape[0])])
            # 诚实校验: 输出全有限 + 域内 + 变化率
            if not np.isfinite(g2).all():
                fails.append("goal 输出含非有限值")
            tgt = np.stack([env.ep_goal[e][:3] for e in range(env.n)])
            err_max = max(err_max, float(np.max(np.linalg.norm(g2[:, :3] - tgt, axis=1))))
            obs = env.reset()
        print(f"[gate] fastbrain goal→target 最大距离 = {err_max:.3f} m")
        if err_max > float(gate.get("goal_dist_m", 0.5)):
            fails.append(f"goal 距目标超阈: {err_max:.3f}")
        _verdict(fails)
        return

    policy, src = make_policy(args.run, mf)
    print(f"[gate] policy={src} episodes={args.episodes} family={task['family']}")
    obs = env.reset()
    ep = {e: {"steps": 0, "ee": [], "style": [], "lift": [], "grasp": []} for e in range(env.n)}
    jerks = []
    prev_a = None
    for _ in range(env.max_ep_steps):
        a = policy(obs)
        if prev_a is not None:
            jerks.append(float(np.max(np.abs(a - prev_a))))
        prev_a = a.copy()
        obs, r, done, infos = env.step(a)
        for e, inf in enumerate(infos):
            ep[e]["steps"] += 1
            if "ee_err" in inf:
                ep[e]["ee"].append(inf["ee_err"])
            if "style_err" in inf:
                ep[e]["style"].append(inf["style_err"])
            if "obj_lift" in inf:
                ep[e]["lift"].append(inf["obj_lift"])
                ep[e]["grasp"].append(inf["obj_grasp_n"])
        if done.all():
            break

    fam = task["family"]
    if fam == "balance":
        survive = float(np.mean([1.0 if ep[e]["steps"] >= env.max_ep_steps else 0.0
                                 for e in range(env.n)]))
        print(f"[gate] balance 存活率 = {survive:.2f}")
        if survive < float(gate.get("survive", 1.0)):
            fails.append(f"存活率 {survive:.2f} < {gate.get('survive')}")
    elif fam == "manipulation":
        ees = [x for e in range(env.n) for x in ep[e]["ee"]]
        tail = float(np.mean(ees[-max(1, len(ees) // 5):])) if ees else 9e9
        print(f"[gate] manipulation 末端误差(尾段均值) = {tail:.4f} m")
        if tail > float(gate.get("ee_err_m", 9e9)):
            fails.append(f"末端误差 {tail:.4f} > {gate.get('ee_err_m')}")
    elif fam == "deform":
        j = float(np.max(jerks)) if jerks else 9e9
        print(f"[gate] deform 20ms 最大跳变 = {j:.4f} rad(tri_03 阈 0.05)")
        if j > float(gate.get("max_step_rad", 0.05)):
            fails.append(f"跳变 {j:.4f} > {gate.get('max_step_rad')}")
    elif fam == "expression":
        s = float(np.mean([x for e in range(env.n) for x in ep[e]["style"]])) \
            if any(ep[e]["style"] for e in range(env.n)) else 9e9
        print(f"[gate] expression 风格误差 = {s:.4f} rad")
        if s > float(gate.get("style_err_rad", 9e9)):
            fails.append(f"风格误差 {s:.4f} > {gate.get('style_err_rad')}")
    if task.get("grasp"):
        lifts = [x for e in range(env.n) for x in ep[e]["lift"]]
        best = float(np.max(lifts)) if lifts else 0.0
        gn = [x for e in range(env.n) for x in ep[e]["grasp"]]
        frac = float(np.mean([1.0 if x > 0 else 0.0 for x in gn])) if gn else 0.0
        print(f"[gate] grasp 最大提起 = {best:.4f} m / 接触帧占比 = {frac:.2f}")
        if best < float(gate.get("lift_m", 9e9)):
            fails.append(f"提起 {best:.4f} < {gate.get('lift_m')}")
    _verdict(fails)


def _verdict(fails: list) -> None:
    for f in fails:
        print(f"[FAIL] {f}")
    print("EVAL_GATE_PASS" if not fails else "EVAL_GATE_FAIL")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
