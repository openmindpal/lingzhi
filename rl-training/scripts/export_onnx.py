#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ONNX 导出——契约三重校验(布局/特权/限位),归一化烘焙进计算图。

用法:
  python scripts/export_onnx.py --run runs/<id>
产物:
  runs/<id>/exports/<row_name>.onnx   (策略: [1,obs_dim]→[1,J] 绝对关节目标,软限位内)
  manifest.json 的 exports[] 更新(权重相对路径 + md5)
"""
import argparse
import hashlib
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
import torch
import torch.nn as nn

from rltrain.contract import check_obs_dim, SOFT_LIMIT_MARGIN_RAD


class PolicyExport(nn.Module):
    """导出图:归一化烘焙 → actor → tanh → 绝对目标(软限位钳制头)。

    输出 = 策略出线语义(引擎契约:绝对关节目标,引擎侧再钳一次——双保险)。
    """

    def __init__(self, model: nn.Module, mean, std, neutral, soft_lo, soft_hi, scale):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("neutral", torch.as_tensor(neutral, dtype=torch.float32))
        self.register_buffer("soft_lo", torch.as_tensor(soft_lo, dtype=torch.float32))
        self.register_buffer("soft_hi", torch.as_tensor(soft_hi, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, obs):
        x = (obs - self.mean) / self.std
        mu, _ = self.model.forward(x)
        a = torch.tanh(mu)
        return torch.clamp(self.neutral + a * self.scale, self.soft_lo, self.soft_hi)


class FastbrainExport(nn.Module):
    """快脑导出图:goal' = clamp(net(x), 域) + nan_to_num(引擎要求全有限)。"""

    def __init__(self, net, box_lo, box_hi):
        super().__init__()
        self.net = net
        self.register_buffer("box_lo", torch.as_tensor(box_lo, dtype=torch.float32))
        self.register_buffer("box_hi", torch.as_tensor(box_hi, dtype=torch.float32))

    def forward(self, x):
        g = self.net(x)
        g = torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0)
        xyz = torch.clamp(g[:, :3], self.box_lo[:3], self.box_hi[:3])
        f = torch.clamp(g[:, 3:4], self.box_lo[3:4], self.box_hi[3:4])
        return torch.cat([xyz, f], dim=1)


def export_policy(run: str) -> None:
    from rltrain.contract import check_obs_dim
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location="cpu", weights_only=False)
    mf = json.load(open(os.path.join(run, "manifest.json"), encoding="utf-8"))
    joint_names = mf.get("joint_names") or []
    check_obs_dim(mf["obs_dim"], mf["j"], mf["obs_mode"])
    from rltrain.ppo import ActorCritic
    model = ActorCritic(mf["obs_dim"], mf["j"], hidden=tuple(mf["hidden"]))
    model.load_state_dict(ck["model"])
    std = np.sqrt(np.asarray(ck["obs_var"]) + 1e-6)
    soft_lo = np.asarray([mf["soft_limits"][n][0] + SOFT_LIMIT_MARGIN_RAD for n in joint_names])
    soft_hi = np.asarray([mf["soft_limits"][n][1] - SOFT_LIMIT_MARGIN_RAD for n in joint_names])
    neutral = np.asarray([mf["neutral"][n] for n in joint_names])
    net = PolicyExport(model, ck["obs_mean"], std, neutral, soft_lo, soft_hi,
                       np.asarray(mf["action_scale"]))
    net.eval()
    dummy = torch.zeros(1, mf["obs_dim"], dtype=torch.float32)
    out_dir = os.path.join(run, "exports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{mf['row_name']}.onnx")
    torch.onnx.export(net, (dummy,), path, opset_version=17,
                      input_names=["obs"], output_names=["action"], dynamo=False)
    _validate(path, mf["obs_dim"], mf["j"])
    _register(run, mf, mf["weights_rel"], path)


def export_fastbrain(run: str) -> None:
    ck = torch.load(os.path.join(run, "ckpt.pt"), map_location="cpu", weights_only=False)
    mf = json.load(open(os.path.join(run, "manifest.json"), encoding="utf-8"))
    net = torch.nn.Sequential(torch.nn.Linear(mf["input_dim"], 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 64), torch.nn.Tanh(),
                              torch.nn.Linear(64, 4))
    net.load_state_dict(ck["net"])
    box_lo, box_hi = _fastbrain_box()
    export = FastbrainExport(net, box_lo, box_hi)
    export.eval()
    dummy = torch.zeros(1, mf["input_dim"], dtype=torch.float32)
    out_dir = os.path.join(run, "exports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tactician.onnx")
    torch.onnx.export(export, (dummy,), path, opset_version=17,
                      input_names=["obs"], output_names=["goal"], dynamo=False)
    _validate(path, mf["input_dim"], 4)
    _register(run, mf, mf["weights_rel"], path)


def _fastbrain_box():
    """goal 域(xyz=作业包络,f∈[0,1])——与任务 yaml 同源。"""
    from rltrain.tasks import load_task
    t = load_task("fastbrain_tactician")
    box = np.asarray(t["goal_box"], dtype=np.float64)
    lo = box[:, 0].tolist() + [0.0]
    hi = box[:, 1].tolist() + [1.0]
    return np.asarray(lo, dtype=np.float32), np.asarray(hi, dtype=np.float32)


def _validate(path: str, in_dim: int, out_dim: int) -> None:
    import onnx
    m = onnx.load(path)
    onnx.checker.check_model(m)
    ins = m.graph.input
    outs = m.graph.output
    assert len(ins) == 1, f"契约禁止多输入图(实际 {len(ins)} 输入)"
    assert list(ins[0].type.tensor_type.shape.dim[i].dim_value for i in range(2)) == [1, in_dim]
    got = list(outs[0].type.tensor_type.shape.dim[i].dim_value for i in range(2))
    assert got == [1, out_dim], f"输出形状 {got} ≠ [1,{out_dim}]"
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, in_dim).astype(np.float32)
    y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    assert y.shape == (1, out_dim), f"运行期输出 {y.shape} ≠ (1,{out_dim})"
    assert np.isfinite(y).all(), "输出含非有限值"
    print(f"[export] OK {os.path.basename(path)} in=[1,{in_dim}] out=[1,{out_dim}] opset=17")


def _register(run: str, mf: dict, weights_rel: str, path: str) -> None:
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    entry = {"weights": weights_rel, "file": os.path.abspath(path), "md5": md5}
    exports = [e for e in mf.get("exports", []) if e["weights"] != weights_rel]
    exports.append(entry)
    mf["exports"] = exports
    with open(os.path.join(run, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(mf, f, ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    mf = json.load(open(os.path.join(args.run, "manifest.json"), encoding="utf-8"))
    if mf["kind"] == "fastbrain":
        export_fastbrain(args.run)
    else:
        export_policy(args.run)


if __name__ == "__main__":
    main()
