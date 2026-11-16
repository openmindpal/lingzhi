#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""接触成功评分器验证——提交物（夹具 ONNX）契约与判别力机械断言。

断言：
  1. 产物存在且 ONNX checker 通过
  2. 图契约与引擎 LearnedScorer 逐字对齐（输入 cand [n,17] / 输出 score [n,1]）
  3. 判别力：新鲜合成场景（独立种子 = 分布内新样本）上准确率 ≥ 0.90
  4. 有序性：同一场景内"对向有效对"得分 > "同面失效对"得分（评分器语义）
"""
import importlib.util
import os
import sys

import numpy as np
import onnx

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))          # rl-training
SCRIPT = os.path.join(_ROOT, "rl-training", "scripts", "gen_contact_scorer.py")
FIXTURE = os.path.join(_ROOT, "apps", "device-agent-rs",
                       "fixtures", "tri-family", "contact", "contact_scorer.onnx")


def _load_gen():
    spec = importlib.util.spec_from_file_location("gen_contact_scorer", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pair_candidates(center, half, cloud):
    """构造一对对照候选：有效对向对 vs 同面失效对（同特征域、异标签）。"""
    axis = int(np.argmax(half))                          # 最长轴 = 对向有解方向
    pa = center.copy(); pb = center.copy()
    pa[axis] -= half[axis]; pb[axis] += half[axis]
    good = (pa.copy(), pb.copy())
    face = axis * 2                                      # 同面失效对
    sign = 1.0
    fa = center.copy(); fb = center.copy()
    fa[face // 2] += sign * half[face // 2]
    fb[face // 2] += sign * half[face // 2]
    bad = (fa, fb)
    return good, bad


def run(tmp: str) -> None:
    _ = tmp
    assert os.path.isfile(FIXTURE), f"评分器产物缺失: {FIXTURE}"
    model = onnx.load(FIXTURE)
    onnx.checker.check_model(model)

    # 图契约：输入 cand [n,17]，输出 score [n,1]
    ins = {i.name: [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]
           for i in model.graph.input}
    outs = {o.name: [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]
            for o in model.graph.output}
    assert ins == {"cand": ["n", 17]}, f"输入契约不符: {ins}"
    assert outs == {"score": ["n", 1]}, f"输出契约不符: {outs}"

    g = _load_gen()
    import onnxruntime as ort
    sess = ort.InferenceSession(FIXTURE, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(987654321)               # 独立种子 = 分布内新样本
    xs, ys = [], []
    pairs_good, pairs_bad = [], []
    for _ in range(120):
        center, half, cloud = g.sample_scene(rng)
        (pa, pb), (fa, fb) = _pair_candidates(center, half, cloud)
        xs.append(g.encode_candidate(pa, pb, cloud))
        ys.append(g.label_pair(pa, pb, center, half, cloud))
        xs.append(g.encode_candidate(fa, fb, cloud))
        ys.append(g.label_pair(fa, fb, center, half, cloud))
        pairs_good.append((pa, pb, center, half, cloud))
        pairs_bad.append((fa, fb, center, half, cloud))
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    scores = sess.run(["score"], {"cand": xs})[0].reshape(-1)
    acc = float(((scores > 0.5) == (ys > 0.5)).mean())
    assert acc >= 0.90, f"判别力不达线: {acc:.3f} (<0.90)"

    # 有序性：同场景内"对向有效对"得分须高于"同面失效对"（≥90% 场景成立）
    wins = 0
    n_cmp = 0
    for i in range(len(pairs_good)):
        pa, pb, center, half, cloud = pairs_good[i]
        fa, fb = pairs_bad[i][:2]
        sg = float(sess.run(["score"], {"cand": g.encode_candidate(pa, pb, cloud)[None, :]})[0][0, 0])
        sb = float(sess.run(["score"], {"cand": g.encode_candidate(fa, fb, cloud)[None, :]})[0][0, 0])
        n_cmp += 1
        wins += 1 if sg > sb else 0
    order_rate = wins / max(1, n_cmp)
    assert order_rate >= 0.90, f"有序性不达线: {order_rate:.3f} (<0.90)"

    print(f"CONTACT_SCORER_VERIFIED acc={acc:.3f} order={order_rate:.3f} n={len(xs)}")


if __name__ == "__main__":
    run(os.path.join(_HERE, ".tmp"))
