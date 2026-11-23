# -*- coding: utf-8 -*-
"""T5b ONNX 在环——导出产物与环境入口的语义一致性断言。

导出图与训练环境是两套动作语义域: 契约(action_contract)规定 ONNX 输出 =
绝对关节目标(rad),环境入口为 tanh 域(neutral + a·scale)。本测试锁住二者的
对齐契约:
  1. 微型训练(1 迭代)→ export_policy 导出
  2. onnxruntime 装载 → 契约 IO 断言(单输入 [1,obs_dim] / 首输出 [1,J])
  3. 按契约语义逆变换回 tanh 域下发(与 eval_gate 同一变换——两处语义共同锁定)
  4. 环境 N 步在环: 动作/观测全程有限 + 步进无异常
诚实边界: 微型策略不收敛——本测试断言"链路语义",不断言存活/收敛
(那是 eval_gate 验收门的职责)。
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))


def run(tmp: str) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(
            name[:-3] + "_mod", os.path.join(root, "scripts", name))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    train = _load("train.py")
    export_mod = _load("export_onnx.py")

    from rltrain.g1_cfg import G1Cfg
    from rltrain.tasks import load_task
    from rltrain.env_mj import G1VecEnv

    cfg = G1Cfg()
    task = load_task("family_balance")
    rd = os.path.join(tmp, "t_onnx_in_env_run")
    os.makedirs(rd, exist_ok=True)

    class A:
        task = "family_balance"; algo = "ppo"; backend = "mujoco"; device = "cpu"
        envs = 1; iters = 1; steps = 8; lr = "3e-4"; hidden = [32, 32]; seed = 5
        eval_every = 0; push_curriculum = False; pretrain_demos = None
        bc_pretrain_epochs = 0; out = rd
    train.train_policy(task, cfg, A(), rd)
    export_mod.export_policy(rd)

    mf = json.load(open(os.path.join(rd, "manifest.json"), encoding="utf-8"))
    fname = mf["exports"][-1]["file"].replace("\\", "/").split("/")[-1]
    path = os.path.join(rd, "exports", fname)
    assert os.path.isfile(path), "导出产物缺失"

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    assert len(sess.get_inputs()) == 1, "契约禁止多输入图"
    in_dim = sess.get_inputs()[0].shape[1]
    assert in_dim == mf["obs_dim"], f"输入宽度 {in_dim} ≠ manifest {mf['obs_dim']}"

    env = G1VecEnv(cfg, task, 2, seed=5)
    obs = env.reset()
    assert obs.shape[1] == in_dim, "环境观测宽度与导出图不一致"

    # 契约语义在环: ONNX 输出 = 绝对关节目标 → 逆变换回 tanh 域下发
    # (镜像 eval_gate 的逆变换——两处语义必须一致,本测试同时锁住它们)
    n_steps = 30
    for _ in range(n_steps):
        a = np.stack([sess.run(None, {sess.get_inputs()[0].name: obs[i:i + 1].astype(np.float32)})[0][0]
                      for i in range(obs.shape[0])])
        assert a.shape == (obs.shape[0], env.j) and np.isfinite(a).all(), \
            f"ONNX 输出形状/有限性违规: {a.shape}"
        a_env = (a - env.neutral[None, :]) / env.action_scale_vec[None, :]
        obs, r, done, infos = env.step(a_env)
        assert np.isfinite(obs).all(), "在环观测含非有限值"
    print(f"  onnx_in_env: 契约 IO + 逆变换 + {n_steps} 步在环 全部有限 OK "
          f"(obs_dim={in_dim}, J={env.j})")


if __name__ == "__main__":
    run(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp"))
