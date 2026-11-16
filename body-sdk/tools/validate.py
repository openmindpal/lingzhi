# -*- coding: utf-8 -*-
"""本体包校验器(对接套件独立版,零外部依赖)——装载前机械自检。

用法:
  python tools/validate.py <包目录>          # 全部检查
  python tools/validate.py <包目录> --dev    # 宽松档(签名未签 = WARN 不拒)
退出码 = 唯一裁决(0 全绿)。

检查项(V1~V5):
  V1 TOML 可解析 + 必填段在场(protocol/profile/calibration/acquisition)
  V2 版本门(word_schema_version/abi_version/runtime_min)
  V3 策略行:family 枚举/obs_dim ∈ 合法宽度集(J 由包内 joint_mapping 自 derivation)/权重在场
  V4 快脑行:kind=onnx/goal_stream 交叉校验
  V5 中立位 ∈ 软限位(margin 0.05)+ 关节映射唯一性
  V6 末端执行器交叉校验(重名/actuator 关节∈映射且目标在限位域/力参数域/scorer 权重在场)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib

MARGIN = 0.05
FAMILIES = ("balance", "manipulation", "deform", "expression")
WORD_SCHEMA_SUPPORTED = [1]
REQUIRED_SECTIONS = ("protocol", "profile", "calibration", "acquisition")
OPTIONAL_SECTIONS = ("words", "policies", "fastbrains", "output", "safety", "reactions")

FAILS: list[str] = []
WARNS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def _obs_legal(j: int, b: int, goal: bool, contact: bool) -> set[int]:
    """合法宽度集——与引擎 validate.rs 同源：IMU 块可有可无（双分支），
    接触块 6k 仅 obs_contact 行入集，goal 行 +4。"""
    ks = range(0, 9) if contact else [0]
    dims = set()
    for bb in range(1, b + 1):        # 块数 = 上限——1..=b 全扫（引擎 validate.rs 同源）
        for k in ks:
            c = 6 * k
            dims |= {bb * j + c, bb * j + 10 + c}
    if goal:
        dims = {d + 4 for d in dims}
    return dims


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkg")
    ap.add_argument("--dev", action="store_true", help="宽松档(未签 = WARN)")
    args = ap.parse_args()
    path = os.path.join(args.pkg, "body-package.toml")

    # V1 结构
    check(os.path.exists(path), "V1 body-package.toml 在场")
    if not os.path.exists(path):
        sys.exit(1)
    raw = open(path, encoding="utf-8").read()
    pkg = tomllib.loads(raw)
    signed = raw.lstrip().startswith('signature = "') and not raw.lstrip().startswith('signature = ""')
    if signed:
        check(True, "V1 签名在场")
    elif args.dev:
        WARNS.append("未签署(--dev 宽松档放行;交付必须签)")
    else:
        check(False, "V1 签名在场")

    for sec in REQUIRED_SECTIONS:
        check(sec in pkg, f"V1 必填段 {sec}")
    for sec in OPTIONAL_SECTIONS:
        if sec not in pkg:
            WARNS.append(f"可缺省段 {sec} 缺失(对应能力空白——诚实缺省)")

    # V2 版本门
    core = pkg.get("core", {})
    check(core.get("word_schema_version") in WORD_SCHEMA_SUPPORTED,
          "V2 word_schema_version", str(core.get("word_schema_version")))
    check(isinstance(core.get("abi_version"), int), "V2 abi_version")
    check(isinstance(core.get("runtime_min"), str), "V2 runtime_min")

    # J 与 b 由包内自 derivation(不依赖外部配置)
    jm = pkg.get("protocol", {}).get("joint_mapping", {}).get("ordered", [])
    check(len(jm) > 0 and len(set(jm)) == len(jm), "V5 关节映射非空且唯一", f"J={len(jm)}")
    j = len(jm)
    mf = pkg.get("protocol", {}).get("sensor_topics", {}).get("joint_state", {}).get("motor_fields")
    b = 1 + len([f for f in (mf or []) if ":dq" in str(f) or ":tau" in str(f)]) if mf else 3

    # V3 策略行
    rows = pkg.get("policies", {}).get("rows", [])
    if not rows:
        # 策略段可缺省（规范 §2）——微码桥厂商动作走微码原语，无策略行合法
        print("[WARN] V3 策略段缺失(可缺省——微码桥路线动作走微码原语)")

    for r in rows:
        check(r.get("family") in FAMILIES, f"V3 family 枚举 {r.get('name')}", str(r.get("family")))
        legal = _obs_legal(j, b, bool(r.get("goal_conditioned")), bool(r.get("obs_contact")))
        check(int(r.get("obs_dim", -1)) in legal,
              f"V3 obs_dim 合法宽度集 {r.get('name')}",
              f"obs_dim={r.get('obs_dim')} j={j} b={b}")
        w = os.path.join(args.pkg, r.get("weights", ""))
        check(os.path.exists(w), f"V3 权重在场 {r.get('name')}", r.get("weights", ""))
        check(float(r.get("frequency_hz", 0)) > 0, f"V3 frequency_hz>0 {r.get('name')}")

    # V4 快脑行
    names = {r.get("name"): r for r in rows}
    for fb in pkg.get("fastbrains", {}).get("rows", []):
        check(fb.get("kind") == "onnx", f"V4 kind=onnx {fb.get('name')}")
        gs = (fb.get("outputs") or {}).get("goal_stream")
        tgt = names.get(gs)
        check(tgt is not None and bool(tgt.get("goal_conditioned")),
              f"V4 goal_stream 交叉校验 {fb.get('name')}", str(gs))
        check(float(fb.get("rate_hz", 0)) > 0 and int(fb.get("budget_ms", 0)) > 0,
              f"V4 rate/budget {fb.get('name')}")

    # V5 中立位 ∈ 软限位
    prof = pkg.get("profile", {}).get("joints", {})
    neutral = pkg.get("calibration", {}).get("neutral", {})
    bad = []
    for name, q0 in neutral.items():
        lim = prof.get(name, {}).get("limits")
        if lim and not (float(lim["min"]) + MARGIN <= float(q0) <= float(lim["max"]) - MARGIN):
            bad.append(name)
    check(not bad, "V5 中立位 ∈ 软限位(margin 0.05)", ",".join(bad) or "全部通过")

    # V6 末端执行器交叉校验（镜像引擎装载校验——声明制诚实语义：段缺省 = 合法）
    ee = pkg.get("end_effector")
    if not ee:
        check(True, "V6 end_effector 未声明(合法——无末端不注册抓取能力)")
    else:
        specs = [ee.get("primary")] + list(ee.get("auxiliary", []))
        seen_ee = set()
        for sp in specs:
            if not sp:
                check(False, "V6 末端实例结构缺失")
                continue
            nm = sp.get("name", "")
            check(bool(nm) and nm not in seen_ee, f"V6 实例名非空唯一 {nm}")
            seen_ee.add(nm)
            check(bool(sp.get("modality")), f"V6 modality 非空 {nm}")
            check(sp.get("attach") in ("right", "left"), f"V6 attach 枚举 {nm}", str(sp.get("attach")))
            if sp.get("scorer_weights"):
                w = os.path.join(args.pkg, sp["scorer_weights"])
                check(os.path.exists(w), f"V6 评分器权重在场 {nm}", sp["scorer_weights"])
            act = sp.get("actuator")
            if not act:
                check(True, f"V6 actuator 未声明 {nm}(合法——闭合规格无执行翻译则 effector_close 诚实拒绝)")
                continue
            check(act.get("kind") == "joints", f"V6 actuator.kind=joints {nm}", str(act.get("kind")))
            for jn in act.get("joints", []):
                check(jn in jm, f"V6 闭合关节 ∈ 映射序 {nm}:{jn}")
                lim = prof.get(jn, {}).get("limits")
                for label in ("open_pos", "closed_pos"):
                    v = act.get(label)
                    check(isinstance(v, (int, float)), f"V6 {label} 有限 {nm}", str(v))
                    if isinstance(v, (int, float)) and lim:
                        check(float(lim["min"]) <= float(v) <= float(lim["max"]),
                              f"V6 {label} ∈ 限位域 {nm}:{jn}", f"{v} vs {lim}")
            ctl = act.get("close_tau_limit")
            check(ctl is None or (isinstance(ctl, (int, float)) and float(ctl) > 0),
                  f"V6 close_tau_limit 正有限 {nm}", str(ctl))
            cs = act.get("close_step")
            check(cs is None or (isinstance(cs, (int, float)) and float(cs) > 0),
                  f"V6 close_step 正有限 {nm}", str(cs))
            ctm = act.get("close_ticks_max")
            check(ctm is None or (isinstance(ctm, int) and int(ctm) >= 1),
                  f"V6 close_ticks_max ≥ 1 {nm}", str(ctm))

    for w in WARNS:
        print(f"[WARN] {w}")
    print("VALIDATE_PASS" if not FAILS else f"VALIDATE_FAIL({len(FAILS)})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
