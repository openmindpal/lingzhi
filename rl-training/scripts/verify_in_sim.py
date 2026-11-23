#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验收门(包装载级,机械断言)——本地门 + docker 集成门。

用法:
  python scripts/verify_in_sim.py --package runs/<id>/package
  python scripts/verify_in_sim.py --docker        # 集成门:仿真平台 verifier 全套件

本地门(退出码 = 唯一裁决):
  V1 TOML 可解析 + 签名行在场(未签 = WARN 不挡本地门,交付必签)
  V2 策略行:family 枚举 / obs_dim ∈ 合法宽度集 / 权重文件在场
  V3 快脑行:kind=onnx / goal_stream 交叉校验(须存在且 goal_conditioned)
  V4 ONNX 逐个装载(onnxruntime):单输入/形状/有限输出
  V5 中立位 ∈ 软限位(装载校验镜像)
"""
import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

import numpy as np
from rltrain._compat import tomllib

from rltrain.contract import legal_obs_dims, neutral_in_soft_limits
from rltrain.g1_cfg import G1Cfg

FAILS: list[str] = []
WARNS: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def verify_local(pkg: str) -> None:
    cfg = G1Cfg()
    path = os.path.join(pkg, "body-package.toml")
    check(os.path.exists(path), "V1 body-package.toml 在场")
    raw = open(path, encoding="utf-8").read()
    pkgd = tomllib.loads(raw)
    check(raw.lstrip().startswith("signature = \""), "V1 签名行在场")
    if raw.lstrip().startswith('signature = ""'):
        WARNS.append("签名为占位(未签署)——交付前须 sign_package.py")
    # 关节数: 交付段 "# j = N" 标注(手包 J≠29);缺省 = 基座 29
    import re
    mj = re.search(r'# j = (\d+)', raw)
    j = int(mj.group(1)) if mj else cfg.j

    rows = pkgd.get("policies", {}).get("rows", [])
    check(len(rows) > 0, "V2 策略行在场", f"{len(rows)} 行")
    for r in rows:
        check(r.get("family") in ("balance", "manipulation", "deform", "expression"),
              f"V2 family 枚举 {r.get('name')}", str(r.get("family")))
        legal = legal_obs_dims(j, 3, with_goal=bool(r.get("goal_conditioned")),
                               with_contact=bool(r.get("obs_contact")))
        check(int(r.get("obs_dim", -1)) in legal,
              f"V2 obs_dim 合法宽度集 {r.get('name')}", f"obs_dim={r.get('obs_dim')} j={j}")
        w = os.path.join(pkg, r.get("weights", ""))
        check(os.path.exists(w), f"V2 权重在场 {r.get('name')}", r.get("weights", ""))

    fb = pkgd.get("fastbrains", {}).get("rows", [])
    policy_names = {r.get("name"): r for r in rows}
    for f in fb:
        check(f.get("kind") == "onnx", f"V3 kind=onnx {f.get('name')}")
        gs = (f.get("outputs") or {}).get("goal_stream")
        target = policy_names.get(gs)
        check(target is not None and bool(target.get("goal_conditioned")),
              f"V3 goal_stream 交叉校验 {f.get('name')}", str(gs))

    try:
        import onnxruntime as ort
        have_ort = True
    except Exception:
        have_ort = False
        WARNS.append("onnxruntime 缺——V4 跳过(本地门弱化,交付环境必须装)")
    if have_ort:
        import glob as g
        for w in g.glob(os.path.join(pkg, "**", "*.onnx"), recursive=True):
            rel = os.path.relpath(w, pkg)
            try:
                sess = ort.InferenceSession(w, providers=["CPUExecutionProvider"])
                check(len(sess.get_inputs()) == 1, f"V4 单输入 {rel}")
                x = np.random.randn(1, sess.get_inputs()[0].shape[1]).astype(np.float32)
                y = sess.run(None, {sess.get_inputs()[0].name: x})[0]
                check(np.isfinite(y).all() and y.shape[0] == 1, f"V4 前向有限 {rel}",
                      f"out={y.shape}")
            except Exception as e:
                check(False, f"V4 ONNX 装载 {rel}", str(e))

    bad = neutral_in_soft_limits(cfg.neutral, cfg.limits)
    check(not bad, "V5 中立位 ∈ 软限位(基座 29 关节)", ",".join(bad) or "全部通过")
    if j != cfg.j:
        WARNS.append(f"扩展本体包(j={j})——V5 覆盖基座 29 关节;扩展通道由 manifest 软限位自检承担")


def verify_docker() -> None:
    """集成门:仿真平台 verifier 全套件(需镜像已 build)。"""
    repo = os.path.dirname(os.path.dirname(_HERE))
    compose = os.path.join(repo, "simulation", "docker", "compose.yaml")
    cmd = ["docker", "compose", "-f", compose, "run", "--rm", "verifier"]
    print("[docker] ", " ".join(cmd))
    sys.exit(subprocess.call(cmd, cwd=repo))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default=None)
    ap.add_argument("--docker", action="store_true")
    args = ap.parse_args()
    if args.docker:
        verify_docker()
        return
    if not args.package:
        ap.error("--package 或 --docker 必选其一")
    verify_local(args.package)
    for w in WARNS:
        print(f"[WARN] {w}")
    print("VERIFY_PASS(本地装载门全绿)" if not FAILS else "VERIFY_FAIL")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
