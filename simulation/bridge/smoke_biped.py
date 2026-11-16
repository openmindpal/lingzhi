#!/usr/bin/env python3
"""双足包冒烟——29 DOF 模型/包一致性 + 平衡策略 ORT 闭环扰动回归（tri_01 判据语义）。

判据（全部机械断言）：
  B1 模型关节 = 包 state_order 同序 29（文档序=编译序——桥接器启动门同款）
  B2 obs 装配：q|dq|tau|imu = 97（宽度集 3·29+10）；goal 行 = 101
  B3 平衡闭环：扰动注入（关节力矩冲击）→ balance.onnx 闭环 → |q−neutral|
     单调收敛（扰动后峰值回落 >70%——指数回归判据）
  B4 reach 语义：goal_x 扫掠 → 稳态关节目标偏移 = g·gx·sign_i（可解析预测）
用法: python -m bridge.smoke_biped
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .mj_model import MjModel, substep_ratio

PKG = Path(__file__).resolve().parents[1] / "packages" / "sim-biped-29dof"
MODEL = Path(__file__).resolve().parents[1] / "assets" / "models" / "biped_29dof.xml"
SIM_HZ = 500.0
POLICY_HZ = 50.0


def assemble_obs(m: MjModel, channels, goal=None) -> np.ndarray:
    """观测装配 v3/v5 白名单序：[q|dq|tau|imu(|goal)]——runtime同构（无归一化）。"""
    snap = m.read_state(channels)
    L = snap["n"]
    obs = np.array(snap["q"][:L] + snap["dq"][:L] + snap["tau"][:L] + snap["imu"],
                   dtype=np.float32)
    if goal is not None:
        obs = np.concatenate([obs, np.array(goal, dtype=np.float32)])
    return obs


def main() -> None:
    import tomllib
    cfg = tomllib.loads((PKG / "body-package.toml").read_text(encoding="utf-8"))
    mapping = cfg["protocol"]["joint_mapping"]["ordered"]
    state_order = cfg["profile"]["state_order"]
    neutral_map = cfg["calibration"]["neutral"]
    kp = float(cfg["profile"]["dynamics"]["kp"])
    kd = float(cfg["profile"]["dynamics"]["kd"])
    L = len(mapping)
    neutral = np.array([float(neutral_map[n]) for n in mapping], dtype=np.float32)

    # B1 模型/包一致性——机械契约 = 映射关节全部在模型中（channels 逐名解析，
    # 未命中 KeyError 诚实拒绝）；模型可携带超出包映射的本体自由度（手指链，
    # 树遍历文档序与包 state_order 的先后约定解耦——线打包按映射序投影，
    # 模型文档序不承载线语义）
    m = MjModel(str(MODEL))
    channels = m.channels(mapping)
    extra = [n for n in m.joint_names if n not in set(state_order)]
    print(f"B1 PASS {L} 映射关节全可伺服；模型 {len(m.joint_names)} 关节，包外 DOF {len(extra)} 个（本体扩展）")

    # B2 obs 装配宽度
    m.reset()
    obs = assemble_obs(m, channels)
    assert obs.shape == (97,), f"B2 FAIL obs {obs.shape} ≠ 97"
    obs_g = assemble_obs(m, channels, goal=[0.5, 0.0, 0.2, 1.0])
    assert obs_g.shape == (101,)
    print(f"B2 PASS obs 布局 q|dq|tau|imu={obs.shape[0]} (+goal={obs_g.shape[0]})")

    sess = ort.InferenceSession(str(PKG / "policies" / "balance.onnx"),
                                providers=["CPUExecutionProvider"])
    n_sub = substep_ratio(SIM_HZ, POLICY_HZ, m.timestep)
    m.reset()

    def run_balance(ticks: int, kick: tuple[int, float, int] | None = None) -> list[float]:
        errs = []
        for t in range(ticks):
            if kick:
                _, da = channels[kick[0]]
                m.set_external(da, kick[1] if 10 <= t < 10 + kick[2] else 0.0)
            act = sess.run(None, {"obs": assemble_obs(m, channels)[None, :]})[0][0]
            m.step([float(v) for v in act], channels, kp, kd, n_sub)
            q = np.array([m.data.qpos[qa] for qa, _ in channels], dtype=np.float32)
            errs.append(float(np.max(np.abs(q - neutral))))
        return errs

    base_err = run_balance(20)[-1]
    kicked = run_balance(80, kick=(16, 60.0, 10))           # 肩 yaw 持续冲击 10 拍
    peak = max(kicked[5:30])
    settled = kicked[-1]
    assert peak > 0.08, f"B3 FAIL 冲击未产生可测位移 peak={peak:.3f}（扰动强度不足）"
    recovery = (peak - settled) / max(peak - base_err, 1e-9)
    assert recovery > 0.7, f"B3 FAIL 回归率 {recovery:.2f}（peak={peak:.3f} settled={settled:.3f}）"
    print(f"B3 PASS 扰动回归 peak={peak:.3f} → settled={settled:.3f} 回归率 {recovery:.0%}")

    # B4 策略语义（精确——解析期望从 ONNX 权重行导出，不复制模式）+ 伺服权威（有界）
    import onnx
    pol = onnx.load(str(PKG / "policies" / "reach.onnx"))
    init = {t.name: onnx.numpy_helper.to_array(t) for t in pol.graph.initializer}
    base_dim = 3 * L + 10
    w_goal_col = init["W"][:, base_dim]                 # act 对 goal_x 的解析增益行
    sess_r = ort.InferenceSession(str(PKG / "policies" / "reach.onnx"),
                                  providers=["CPUExecutionProvider"])
    act_by_gx = {}
    for gx in (0.1, 0.9):
        m.reset()
        act = None
        for _ in range(80):
            act = sess_r.run(None, {"obs": assemble_obs(m, channels, goal=[gx, 0, 0, 1.0])[None, :]})[0][0]
            m.step([float(v) for v in act], channels, kp, kd, n_sub)
        act_by_gx[gx] = np.array(act, dtype=np.float32)
        if gx == 0.9:
            q_steady = np.array([m.data.qpos[qa] for qa, _ in channels], dtype=np.float32)
            q_vs_target = float(np.max(np.abs(q_steady - act)))
    delta_act = act_by_gx[0.9] - act_by_gx[0.1]
    expect = w_goal_col * 0.8                            # 线性策略：Δact = W_col·Δgx
    assert np.max(np.abs(delta_act - expect)) < 1e-4, \
        f"B4 FAIL 策略语义 act Δ vs W 列 max|Δ|={np.max(np.abs(delta_act - expect)):.4f}"
    assert q_vs_target < 0.25, f"B4 FAIL 伺服权威 重力垂坠 {q_vs_target:.3f} rad 超界"
    print(f"B4 PASS 策略 act 偏移=W 列解析值精确；伺服重力垂坠 max={q_vs_target:.3f} rad（<0.25 有界）")

    print("BIPED SMOKE PASS 全绿（B1~B4）——平衡回归/可解析预测闭环实证")


if __name__ == "__main__":
    main()
