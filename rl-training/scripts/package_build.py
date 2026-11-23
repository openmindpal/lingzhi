#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本体包构建——由 run 产物生成可装载的本体包(权重 + 行声明 + 签名)。

用法:
  python scripts/package_build.py --run runs/<id> [--out runs/<id>/package]
                                  [--package-name unitree-g1] [--sign-key <pem>]
流程(架构 L4):
  1) 权重落位 policies/ fastbrains/
  2) 从 unitree-g1 包继承 core/protocol/profile/calibration,重生成 policies/fastbrains 段
  3) 装载前自检(obs_dim 合法宽度集/中立位∈软限位/goal_stream 交叉校验)
  4) 签名(占位→签→回填,复用 apps/body-packages/tools/sign_package.py)
"""
import argparse
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rltrain._compat import tomllib

from rltrain.contract import check_obs_dim
from rltrain.g1_cfg import G1Cfg, PKG_TOML

POLICY_MARK = "── policies"


def gen_policy_row(mf: dict) -> str:
    mode = ("goal" if mf["goal_conditioned"] else "") + ("contact" if mf["obs_contact"] else "")
    gc = "\ngoal_conditioned = true" if mf["goal_conditioned"] else ""
    oc = "\nobs_contact = true" if mf["obs_contact"] else ""
    extra = "\n# k=2(0=左腕 1=右腕)——obs_dim=109" if mode == "contact" else ""
    return (f'[[policies.rows]]\n'
            f'name = "{mf["row_name"]}"\n'
            f'family = "{mf["family"]}"\n'
            f'weights = "{mf["weights_rel"]}"\n'
            f'frequency_hz = {mf["frequency_hz"]}\n'
            f'priority = "{mf["priority"]}"\n'
            f'obs_dim = {mf["obs_dim"]}{gc}{oc}{extra}\n')


def self_checks(mf: dict, pkg_dir: str) -> list[str]:
    """装载前自检(引擎装载校验镜像)。返回违例列表(空 = 通过)。"""
    bad = []
    cfg = G1Cfg()
    if mf["kind"] == "fastbrain":
        w = os.path.join(pkg_dir, mf["weights_rel"])
        if not os.path.exists(w):
            bad.append(f"快脑权重缺失: {w}")
        return bad
    mode = ("goal_" if mf["goal_conditioned"] else "") + ("contact" if mf["obs_contact"] else "")
    mode = mode.rstrip("_") or "base"
    try:
        check_obs_dim(mf["obs_dim"], int(mf.get("j", 29)), mode)
    except ValueError as e:
        bad.append(str(e))
    if mf["family"] not in ("balance", "manipulation", "deform", "expression"):
        bad.append(f"family 非法: {mf['family']}")
    if not os.path.exists(os.path.join(pkg_dir, mf["weights_rel"])):
        bad.append(f"权重缺失: {mf['weights_rel']}")
    from rltrain.contract import neutral_in_soft_limits
    # manifest.soft_limits 已含余量(env 软域)——此处 margin=0 直接比对
    limits = {n: tuple(v) for n, v in mf.get("soft_limits", {}).items()} or cfg.limits
    bad += [f"中立位越软限位: {n}" for n in
            neutral_in_soft_limits(mf.get("neutral", {}), limits, margin=0.0)]
    return bad


def build(runs: list[str], out: str, pkg_name: str, sign_key: str | None,
          muscle_run: str | None = None) -> None:
    """整包交付:多个 run 合成一个本体包(四族全部行 + 快脑)。

    规则:
      - 所有 policy 行共享同一本体 → j 必须一致(不一致 = 诚实拒收)
      - 快脑行的 goal_stream 须命中包内某个 policy 行(交叉校验)
      - row_name 重复 = 拒收(派发主键唯一)
    """
    mfs = []
    for r in runs:
        mf = json.load(open(os.path.join(r, "manifest.json"), encoding="utf-8"))
        if mf.get("kind") == "policy" and not mf.get("exports"):
            raise SystemExit(f"run {r} 无 exports——先 export_onnx.py")
        if mf.get("kind") == "fastbrain" and not mf.get("exports"):
            raise SystemExit(f"run {r} 无 exports——先 export_onnx.py")
        mfs.append(mf)
    if not mfs:
        raise SystemExit("无有效 run")

    policy_mfs = [m for m in mfs if m["kind"] == "policy"]
    fb_mfs = [m for m in mfs if m["kind"] == "fastbrain"]
    names = [m["row_name"] for m in policy_mfs]
    if len(names) != len(set(names)):
        raise SystemExit(f"row_name 重复(派发主键唯一): {names}")
    js = {int(m.get("j", 29)) for m in policy_mfs} or {29}
    if len(js) != 1:
        raise SystemExit(f"混合本体包须同 j(收到 {sorted(js)})——手包/基座包分开交付")
    j = js.pop()

    # 快脑交叉校验(goal_stream → 包内 policy 行;不在包内则须 --muscle-run 补)
    for fb in fb_mfs:
        if fb["goal_stream"] not in names and muscle_run:
            mmf = json.load(open(os.path.join(muscle_run, "manifest.json"), encoding="utf-8"))
            if mmf.get("kind") == "policy" and mmf.get("exports"):
                if mmf["row_name"] != fb["goal_stream"]:
                    raise SystemExit(f"--muscle-run 行名 {mmf['row_name']} ≠ goal_stream {fb['goal_stream']}")
                mfs.append(mmf); policy_mfs.append(mmf); names.append(mmf["row_name"])
    for fb in fb_mfs:
        if fb["goal_stream"] not in names:
            raise SystemExit(f"快脑 {fb['name']} 的 goal_stream={fb['goal_stream']} "
                             f"不在包内策略行 {names}——补对应 run 或 --muscle-run")

    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "policies"), exist_ok=True)
    os.makedirs(os.path.join(out, "fastbrains"), exist_ok=True)

    # 1) 权重落位
    seen = set()
    for m in mfs:
        for e in m.get("exports", []):
            if e["weights"] in seen:
                continue
            seen.add(e["weights"])
            dst = os.path.join(out, e["weights"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(e["file"], dst)

    # 2) 继承源包(去签名行/去 policies 段),重生成交付段
    src = open(PKG_TOML, encoding="utf-8").read()
    src = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("signature"))
    ix = src.find(POLICY_MARK)
    if ix < 0:
        raise SystemExit("源包缺 policies 段标记")
    base = src[:ix].rstrip() + "\n"
    toml = ('signature = ""\n' if sign_key else "") + base
    toml += f"\n# ── policies/fastbrains:RL 训练平台交付段(package_build 自动生成,整包 {len(policy_mfs)} 策略 + {len(fb_mfs)} 快脑)──\n"
    toml += f"# j = {j}\n"
    for m in policy_mfs:
        toml += gen_policy_row(m) + "\n"
    for fb in fb_mfs:
        toml += (f'[[fastbrains.rows]]\nname = "{fb["name"]}"\nkind = "onnx"\n'
                 f'weights = "{fb["weights_rel"]}"\n'
                 f'outputs = {{ goal_stream = "{fb["goal_stream"]}" }}\n'
                 f'rate_hz = {fb["rate_hz"]}\nbudget_ms = {fb["budget_ms"]}\n\n')

    pkg_toml = os.path.join(out, "body-package.toml")
    with open(pkg_toml, "w", encoding="utf-8", newline="\n") as f:
        f.write(toml)
    with open(pkg_toml, "rb") as f:
        tomllib.load(f)   # 语法门:TOML 可解析

    # 3) 自检(逐 policy 行)
    bad = []
    for m in policy_mfs + fb_mfs:
        bad += [f"[{m.get('row_name', m.get('name', '?'))}] {b}" for b in self_checks(m, out)]
    for b in bad:
        print(f"[FAIL] {b}")
    if bad:
        sys.exit(1)

    # 4) 签名(工具 = body-sdk 对接套件;回退 body-packages 历史位置)
    if sign_key:
        import importlib.util
        cand = [os.path.join(os.path.dirname(_HERE), "..", "body-sdk", "tools", "sign_package.py"),
                os.path.join(os.path.dirname(_HERE), "..", "apps", "body-packages", "tools", "sign_package.py")]
        sign_path = next((p for p in cand if os.path.exists(p)), cand[0])
        spec = importlib.util.spec_from_file_location("sign_package", sign_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.sign(out, sign_key)
        print("[sign] 已签署(占位→签→回填)")
    else:
        print("[sign] 跳过——交付前须用 apps/body-packages/tools/sign_package.py 签署")

    print(f"[package] OK {out} — {len(policy_mfs)} policies + {len(fb_mfs)} fastbrains")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True,
                    help="run 目录(可多次传入——整包交付)")
    ap.add_argument("--muscle-run", default=None, help="快脑交付的肌肉策略 run(reach 等)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--package-name", default=None)
    ap.add_argument("--sign-key", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.run[0], "package")
    build(args.run, out, args.package_name, args.sign_key, args.muscle_run)


if __name__ == "__main__":
    main()
