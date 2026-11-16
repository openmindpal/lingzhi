#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接触成功评分器——数据集合成 + 训练 + ONNX 导出（学习版接触模型 v1）。

定位：把抓取"成功判据"从引擎内手写几何代理换成从经验学出的分布——本脚本
生成 (候选接触构型 → 成功标签) 数据集并训练评分网络，产物 = 引擎
LearnedScorer 的 ONNX 权重（声明于本体包 end_effector.scorer_weights）。

监督源（v1 = 合成解析真值；GPU 域 = Isaac 物理闭合+提起标签，接口不变）：
  场景 = 桌面 + 随机尺寸/位置盒；候选 = 双接触点对（含有效对向对与失效对：
  同面/跨度失配/离物）；标签 = 解析真值（对向跨盒 + 跨度匹配 + 净空）。
  v1 学的是"从候选几何复现成功判据"——管线冒烟 + 竞赛基线；语义/堆积/柔性
  情形的覆盖随 GPU 物理标签扩展，接口不动。

ONNX 契约（与引擎 LearnedScorer 逐字对齐，改必同步）：
  输入  cand  [n, 17] f32  [质心xyz | span | 接触数 | 净空 | 高度比 | 展开度 | 闭合轴向xyz | 云质心xyz | 云半延展xyz]
  输出  score [n, 1] f32  成功概率 0~1（sigmoid 在训练侧固化进图）

用法:
  python scripts/gen_contact_scorer.py [--out 路径]   # 缺省写 tri-family 夹具
"""
import argparse
import os

import numpy as np
import onnx
from onnx import helper, TensorProto

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "..", "..", "apps", "device-agent-rs",
                           "fixtures", "tri-family", "contact", "contact_scorer.onnx")

SEED = 20260112
CAND_DIM = 17         # 与引擎 SCORER_CAND_DIM 对齐（8 候选几何 + 3 闭合轴向 + 6 云统计）
N_SCENES = 600
N_CAND_PER_SCENE = 24
EPOCHS = 400
LR = 0.008
HIDDEN = 32


def sample_scene(rng: np.random.Generator):
    """随机盒场景：返回 (盒中心/半边长, 表面点云, 桌面点云)。"""
    half = rng.uniform(0.02, 0.06, size=3)          # 盒半边长 4~12cm
    center = np.array([rng.uniform(0.30, 0.50), rng.uniform(-0.25, 0.25),
                       half[2] + 0.75])              # 桌面 z=0.75 上
    cloud = []
    for _ in range(200):
        face = rng.integers(0, 6)
        p = center.copy()
        axis = face // 2
        p[axis] += half[axis] * (1.0 if face % 2 == 0 else -1.0)
        others = [i for i in range(3) if i != axis]
        p[others[0]] += rng.uniform(-half[others[0]], half[others[0]])
        p[others[1]] += rng.uniform(-half[others[1]], half[others[1]])
        cloud.append(p)
    return center, half, np.array(cloud)             # 云 = 盒面点（roi 裁剪语义）


def sample_candidate(rng: np.random.Generator, center: np.ndarray, half: np.ndarray,
                     cloud: np.ndarray):
    """分层候选采样：40% 真对向对（轴向随机、切向抖动）+ 60% 明确失效对
    （同面 / 跨度失配 / 随机两点）——决策边界厚，冒烟任务可分。"""
    if rng.random() < 0.4:
        axis = int(rng.integers(0, 3))
        jitter = half.copy() * 0.0
        others = [i for i in range(3) if i != axis]
        pa = center.copy(); pb = center.copy()
        pa[axis] -= half[axis]; pb[axis] += half[axis]
        for o in others:
            j = rng.uniform(-0.4, 0.4) * half[o]
            pa[o] += j; pb[o] += j
        span_jitter = rng.uniform(-0.08, 0.08) * 2.0 * half[axis]   # 相对量——落在 ±25% 窗内
        pa[axis] -= span_jitter / 2.0; pb[axis] += span_jitter / 2.0
        return pa, pb
    mode = int(rng.integers(0, 3))
    if mode == 0:                                    # 同面两点
        face = int(rng.integers(0, 6)); axis = face // 2
        sign = 1.0 if face % 2 == 0 else -1.0
        pa = center.copy(); pb = center.copy()
        pa[axis] += sign * half[axis]; pb[axis] += sign * half[axis]
        for o in (i for i in range(3) if i != axis):
            pa[o] += rng.uniform(-half[o], half[o])
            pb[o] += rng.uniform(-half[o], half[o])
        return pa, pb
    if mode == 1:                                    # 对向但跨度失配
        axis = int(rng.integers(0, 3))
        factor = rng.uniform(0.2, 2.2)
        if abs(factor - 1.0) < 0.3:
            factor += 0.6
        pa = center.copy(); pb = center.copy()
        pa[axis] -= half[axis] * factor; pb[axis] += half[axis] * factor
        return pa, pb
    idx = rng.choice(len(cloud), size=2, replace=False)   # 随机两点
    return cloud[idx[0]].copy(), cloud[idx[1]].copy()


def encode_candidate(pa: np.ndarray, pb: np.ndarray, cloud: np.ndarray) -> np.ndarray:
    """引擎候选编码（逐字对齐 LearnedScorer::encode，14 维）。
    云统计 = roi 裁剪云的观测量（引擎可算，非语义泄漏）。"""
    c = (pa + pb) / 2.0
    span = float(np.linalg.norm(pa - pb))
    min_z = float(cloud[:, 2].min())
    cc = cloud.mean(axis=0)
    half_ext = (cloud.max(axis=0) - cloud.min(axis=0)) / 2.0
    d = pb - pa
    norm = float(np.linalg.norm(d))
    d_hat = d / norm if norm > 1e-12 else np.zeros(3)
    return np.array([c[0], c[1], c[2], span, 2.0,
                     c[2] - min_z,
                     min(1.0, max(0.0, (c[2] - min_z) / 0.15)),
                     span,
                     d_hat[0], d_hat[1], d_hat[2],
                     cc[0], cc[1], cc[2],
                     half_ext[0], half_ext[1], half_ext[2]], dtype=np.float32)


def label_pair(pa: np.ndarray, pb: np.ndarray, center: np.ndarray,
               half: np.ndarray, cloud: np.ndarray) -> float:
    """解析真值（编码可表达的三条件）：对向跨盒 + 跨度匹配 + 质心入盒。"""
    d = pb - pa
    span = float(np.linalg.norm(d))
    axis = int(np.argmax(np.abs(d)))
    sa = 1.0 if pa[axis] > center[axis] else -1.0
    sb = 1.0 if pb[axis] > center[axis] else -1.0
    if sa == sb:
        return 0.0                                     # 同侧 = 非对向
    if not (0.75 * 2 * half[axis] <= span <= 1.25 * 2 * half[axis]):
        return 0.0                                     # 跨度与盒该维失配
    mid = (pa + pb) / 2.0
    if np.any(np.abs(mid - center) > half + 0.005):
        return 0.0                                     # 质心出盒 = 非跨盒对
    _ = cloud
    return 1.0


def build_dataset(rng: np.random.Generator):
    xs, ys = [], []
    for _ in range(N_SCENES):
        center, half, cloud = sample_scene(rng)
        for _ in range(N_CAND_PER_SCENE):
            pa, pb = sample_candidate(rng, center, half, cloud)
            xs.append(encode_candidate(pa, pb, cloud))
            ys.append(label_pair(pa, pb, center, half, cloud))
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32).reshape(-1, 1)
    return xs, ys


def train(xs: np.ndarray, ys: np.ndarray, rng: np.random.Generator):
    """numpy MLP（8→32→32→1，BCE + 动量 SGD）——确定性种子，学习冒烟量级。"""
    w1 = rng.normal(0, 0.3, size=(CAND_DIM, HIDDEN)).astype(np.float32)
    b1 = np.zeros(HIDDEN, dtype=np.float32)
    w2 = rng.normal(0, 0.3, size=(HIDDEN, HIDDEN)).astype(np.float32)
    b2 = np.zeros(HIDDEN, dtype=np.float32)
    w3 = rng.normal(0, 0.3, size=(HIDDEN, 1)).astype(np.float32)
    b3 = np.zeros(1, dtype=np.float32)
    params = [w1, b1, w2, b2, w3, b3]
    vel = [np.zeros_like(p) for p in params]
    n = len(xs)
    # float64 训练（数值稳）——导出时转 float32（引擎 ORT 契约）
    xs64 = xs.astype(np.float64)
    ys64 = ys.astype(np.float64)
    for _ in range(EPOCHS):
        order = rng.permutation(n)
        for i0 in range(0, n, 256):
            idx = order[i0:i0 + 256]
            x, y = xs64[idx], ys64[idx]
            # 前向
            h1 = np.maximum(x @ w1 + b1, 0)
            h2 = np.maximum(h1 @ w2 + b2, 0)
            logit = h2 @ w3 + b3
            p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
            # BCE 梯度（softmax 交叉熵对 logit 的梯度 = p − y）
            g = (p - y) / len(idx)
            gw3 = h2.T @ g
            gb3 = g.sum(axis=0)
            dh2 = g @ w3.T
            gw2 = h1.T @ dh2
            gb2 = dh2.sum(axis=0)
            dh1 = dh2 @ w2.T
            dh1x = dh1 * (h1 > 0)
            gw1 = x.T @ dh1x
            gb1 = dh1x.sum(axis=0)
            # 梯度范数裁剪（全局 5）——发散防护
            gnorm = float(np.sqrt(sum(float(np.sum(g_ * g_)) for g_ in [gw1, gb1, gw2, gb2, gw3, gb3])))
            clip = min(1.0, 5.0 / max(gnorm, 1e-12))
            for pi, g_ in enumerate([gw1, gb1, gw2, gb2, gw3, gb3]):
                vel[pi] = 0.9 * vel[pi] + g_ * clip
                params[pi] -= LR * vel[pi]
    return params


def forward(params, xs: np.ndarray) -> np.ndarray:
    w1, b1, w2, b2, w3, b3 = params
    h1 = np.maximum(xs @ w1 + b1, 0)
    h2 = np.maximum(h1 @ w2 + b2, 0)
    return 1.0 / (1.0 + np.exp(-np.clip(h2 @ w3 + b3, -30.0, 30.0)))


def export_onnx(params, mean: np.ndarray, std: np.ndarray, out_path: str) -> None:
    """图：cand [n,17] → 标准化（常量烘进图——引擎契约保持原始特征输入）
    → MatMul/Relu ×2 → Sigmoid → score [n,1]。"""
    w1, b1, w2, b2, w3, b3 = params
    inits = [
        helper.make_tensor("SC_MEAN", TensorProto.FLOAT, [CAND_DIM], mean.astype(np.float32).flatten().tolist()),
        helper.make_tensor("SC_STD", TensorProto.FLOAT, [CAND_DIM], std.astype(np.float32).flatten().tolist()),
        helper.make_tensor("W1", TensorProto.FLOAT, list(w1.shape), w1.astype(np.float32).flatten().tolist()),
        helper.make_tensor("B1", TensorProto.FLOAT, list(b1.shape), b1.astype(np.float32).flatten().tolist()),
        helper.make_tensor("W2", TensorProto.FLOAT, list(w2.shape), w2.astype(np.float32).flatten().tolist()),
        helper.make_tensor("B2", TensorProto.FLOAT, list(b2.shape), b2.astype(np.float32).flatten().tolist()),
        helper.make_tensor("W3", TensorProto.FLOAT, list(w3.shape), w3.astype(np.float32).flatten().tolist()),
        helper.make_tensor("B3", TensorProto.FLOAT, list(b3.shape), b3.astype(np.float32).flatten().tolist()),
    ]
    graph = helper.make_graph(
        [
            helper.make_node("Sub", ["cand", "SC_MEAN"], ["n0"]),
            helper.make_node("Div", ["n0", "SC_STD"], ["n1"]),
            helper.make_node("MatMul", ["n1", "W1"], ["h1"]),
            helper.make_node("Add", ["h1", "B1"], ["h1b"]),
            helper.make_node("Relu", ["h1b"], ["h1r"]),
            helper.make_node("MatMul", ["h1r", "W2"], ["h2"]),
            helper.make_node("Add", ["h2", "B2"], ["h2b"]),
            helper.make_node("Relu", ["h2b"], ["h2r"]),
            helper.make_node("MatMul", ["h2r", "W3"], ["logit"]),
            helper.make_node("Add", ["logit", "B3"], ["logitb"]),
            helper.make_node("Sigmoid", ["logitb"], ["score"]),
        ],
        "contact_scorer",
        [helper.make_tensor_value_info("cand", TensorProto.FLOAT, ["n", CAND_DIM])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, ["n", 1])],
        initializer=inits,
    )
    model = helper.make_model(graph, producer_name="lingzhi-contact-scorer",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", 13))
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    onnx.save(model, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    xs, ys = build_dataset(rng)
    split = int(len(xs) * 0.8)
    mean = xs[:split].mean(axis=0)
    std = xs[:split].std(axis=0) + 1e-6
    xs_n = (xs - mean) / std
    params = train(xs_n[:split], ys[:split], rng)
    pred = forward(params, xs_n[split:])
    acc = float(((pred > 0.5) == (ys[split:] > 0.5)).mean())
    pos_rate = float(ys.mean())
    print(f"DATASET n={len(xs)} 正例率={pos_rate:.2f} | 测试准确率={acc:.3f}")
    assert acc >= 0.95, f"学习冒烟准确率不达线: {acc:.3f} (<0.95)"
    export_onnx(params, mean, std, args.out)
    print(f"CONTACT_SCORER_OK → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
