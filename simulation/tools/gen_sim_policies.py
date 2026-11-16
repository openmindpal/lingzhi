#!/usr/bin/env python3
"""仿真包测试权重生成器（通用版）——按包 policies.rows 逐族生成单 Gemm 可解析预测权重。

行为契约（验证套件的断言依据——行为可预测才可机械断言，对照 e2e/tri-family/gen_policies.py）：
  balance     act = (1+k)·neutral − k·q        → 扰动后指数回归 neutral（tri_01 语义）
  manipulation(goal_conditioned)
              act = neutral + g·gx·sign_i      → goal_x 变化 → 稳态目标可预测偏移（tri_02 语义）
  deform      act = q + v_i                    → 每拍恒速漂移（tri_03 语义；钳制兜底）
  expression  act = neutral                    → 保持（冒烟占位——自然度=训练域，不属断言面）
  fastbrain   输入 = 被绑定策略观测原样（逐位同源）→ goal(4)
              gx_out = 0.3·q0 + 0.3            → 活反馈环（q0 变 → goal 变 → 肌肉响应）

obs 布局：[q(L)|dq(L)|tau(L)|imu(10)(|goal(4))]；goal_x = obs[base_dim]。
用法: python -m tools.gen_sim_policies <package_dir>
"""
import sys
import tomllib
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

K_RECOVER = 0.6        # balance 回归增益
G_GAIN = 0.5           # reach 目标增益
V_STEP = 0.004         # deform 每拍漂移（rad @50Hz → 0.2 rad/s）
FB_Q0_GAIN = 0.3       # tactician 反馈增益


def sign_pattern(i: int) -> float:
    return 1.0 if i % 2 == 0 else -1.0


def make_gemm(name: str, w: np.ndarray, b: np.ndarray) -> bytes:
    obs_dim, act_dim = w.shape[1], w.shape[0]
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["obs", "W", "b"], ["act"],
                          alpha=1.0, beta=1.0, transB=1, name=f"{name}_gemm")],
        name,
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, obs_dim])],
        [helper.make_tensor_value_info("act", TensorProto.FLOAT, [1, act_dim])],
        initializer=[
            helper.make_tensor("W", TensorProto.FLOAT, list(w.shape),
                               w.astype(np.float32).flatten().tolist()),
            helper.make_tensor("b", TensorProto.FLOAT, [act_dim],
                               b.astype(np.float32).flatten().tolist()),
        ],
    )
    model = helper.make_model(graph, producer_name="lingzhi-sim-bridge",
                              opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    del model.opset_import[:]
    model.opset_import.append(helper.make_opsetid("", 13))
    onnx.checker.check_model(model)
    return model.SerializeToString()


def weight_for(row: dict, L: int, base_dim: int, neutral: np.ndarray,
               soft: list[tuple[float, float]], gain: float):
    """按族产出 (W, b)——行为契约见模块头注。gain = 包络推导（main 计算）。"""
    fam = row["family"]
    goal = bool(row.get("goal_conditioned", False))
    obs_dim = base_dim + (4 if goal else 0)
    swing = gain * 0.9                     # 目标域上限 = gain × goal_x 域上界
    if fam == "balance":
        if goal:
            raise SystemExit(f"balance 行 {row['name']} 不应 goal_conditioned")
        w = np.zeros((L, obs_dim))
        for i in range(L):
            w[i, i] = -K_RECOVER
        return w, (1.0 + K_RECOVER) * neutral
    if fam == "manipulation":
        if not goal:
            raise SystemExit(f"manipulation 行 {row['name']} 须 goal_conditioned"
                             "（操作语义 = goal 条件块承载——四族指南 §0）")
        w = np.zeros((L, obs_dim))
        for i in range(L):
            if not (soft[i][0] - 1e-9 <= float(neutral[i]) - swing
                    and float(neutral[i]) + swing <= soft[i][1] + 1e-9):
                raise SystemExit(
                    f"行 {row['name']} 关节 {i} 软域 [{soft[i][0]:.3f},{soft[i][1]:.3f}] "
                    f"容不下 swing±{swing:.3f}——gain 推导缺陷（构造性域断言拒绝）")
            w[i, base_dim] = gain * (1.0 if i % 2 == 0 else -1.0)
        return w, neutral.copy()
    if fam == "deform":
        if goal:
            raise SystemExit(f"deform 行 {row['name']} 不应 goal_conditioned")
        w = np.zeros((L, obs_dim))
        for i in range(L):
            w[i, i] = 1.0
        v = np.array([V_STEP * sign_pattern(i) for i in range(L)])
        return w, v
    if fam == "expression":
        if goal:
            raise SystemExit(f"expression 行 {row['name']} 冒烟生成器不支持 goal_conditioned")
        return np.zeros((L, obs_dim)), neutral.copy()
    raise SystemExit(f"未知族 {fam}（{row['name']}）——仅 balance/manipulation/deform/expression")


def main(pkg_dir: str) -> None:
    pkg = Path(pkg_dir)
    cfg = tomllib.loads((pkg / "body-package.toml").read_text(encoding="utf-8"))
    mapping = cfg["protocol"]["joint_mapping"]["ordered"]
    neutral_map = cfg["calibration"]["neutral"]
    L = len(mapping)
    neutral = np.array([float(neutral_map.get(n, 0.0)) for n in mapping])
    base_dim = 3 * L + 10                     # q|dq|tau|imu
    margin = float(cfg.get("safety", {}).get("soft_limit_margin_rad", 0.05))
    soft = [(float(cfg["profile"]["joints"][n]["limits"]["min"]) + margin,
             float(cfg["profile"]["joints"][n]["limits"]["max"]) - margin)
            for n in mapping]
    # 包络推导 gain：目标域入软域之外再让出伺服垂坠余量（稳态 q = target ± sag）
    SAG_RESERVE = 0.1
    feasible = min(min(s[1] - float(neutral[i]), float(neutral[i]) - s[0]) - SAG_RESERVE
                   for i, s in enumerate(soft)) / 0.9
    gain = min(G_GAIN, feasible)
    if gain <= 0.05:
        raise SystemExit(f"包络推导 gain={gain:.3f} 过小（软域−垂坠余量后不足）——先修包")
    print(f"  envelope gain={gain:.3f}（G_GAIN 上限 {G_GAIN}，可行 {feasible:.3f}，垂坠余量 {SAG_RESERVE}）")
    obs_by_name = {}
    (pkg / "policies").mkdir(exist_ok=True)
    for row in cfg.get("policies", {}).get("rows", []):
        w, b = weight_for(row, L, base_dim, neutral, soft, gain)
        obs_by_name[row["name"]] = w.shape[1]
        if int(row["obs_dim"]) != w.shape[1]:
            raise SystemExit(f"行 {row['name']} obs_dim={row['obs_dim']} ≠ 生成器 {w.shape[1]}"
                             "（宽度集公式 §6.3）")
        out = pkg / row["weights"]          # 直接按行声明的 weights 路径落盘
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(make_gemm(row["name"], w, b))
        print(f"  policy {row['name']}: obs={w.shape[1]} act={L} → {row['weights']}")

    (pkg / "fastbrains").mkdir(exist_ok=True)
    for fb in cfg.get("fastbrains", {}).get("rows", []):
        bound = fb.get("outputs", {}).get("goal_stream")
        if not bound or bound not in obs_by_name:
            raise SystemExit(f"快脑 {fb['name']} goal_stream={bound} 未命中策略行")
        obs_dim = obs_by_name[bound]
        wfb = np.zeros((4, obs_dim))
        wfb[0, 0] = FB_Q0_GAIN
        bfb = np.array([0.3, 0.0, 0.0, 1.0])
        out = pkg / fb["weights"]
        out.write_bytes(make_gemm(fb["name"], wfb, bfb))
        print(f"  fastbrain {fb['name']}: in={obs_dim} out=4 → {fb['weights']}")

    print(f"OK L={L} base={base_dim} → {pkg}")


if __name__ == "__main__":
    main(sys.argv[1])
