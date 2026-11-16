# -*- coding: utf-8 -*-
"""T5 导出→打包(单/整包合并)→装载门 往返。"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys


def _load_script(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "scripts", name)
    spec = importlib.util.spec_from_file_location(name[:-3] + "_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gen_key(tmp: str) -> str:
    key = os.path.join(tmp, "test_key.pem")
    if not os.path.exists(key):
        here = os.path.dirname(os.path.abspath(__file__))          # rl-training/tests
        repo_root = os.path.dirname(os.path.dirname(here))         # 仓库根（rl-training 的上一级）
        cand = [os.path.join(repo_root, "body-sdk", "tools", "sign_package.py"),
                os.path.join(repo_root, "apps", "body-packages", "tools", "sign_package.py")]
        sign = next((p for p in cand if os.path.exists(p)), cand[0])
        subprocess.check_call([sys.executable, sign, "--generate", key],
                              stdout=subprocess.DEVNULL)
    return key


def run(tmp: str) -> None:
    export = _load_script("export_onnx.py")
    pb = _load_script("package_build.py")
    vis = _load_script("verify_in_sim.py")
    key = _gen_key(tmp)

    # 策略导出(基座 29 + 手包 45)
    export.export_policy(os.path.join(tmp, "t_balance"))
    export.export_policy(os.path.join(tmp, "t_sac"))
    export.export_fastbrain(os.path.join(tmp, "t_fb"))
    for rd in ("t_balance", "t_sac", "t_fb"):
        mf = json.load(open(os.path.join(tmp, rd, "manifest.json"), encoding="utf-8"))
        assert mf.get("exports"), f"{rd} 缺 exports"

    # 单策略整包 → 装载门(rltrain 门 + body-sdk 独立校验器 双验)
    pkg1 = os.path.join(tmp, "pkg_balance")
    pb.build([os.path.join(tmp, "t_balance")], pkg1, None, key)
    vis.FAILS.clear(); vis.WARNS.clear()
    vis.verify_local(pkg1)
    assert not vis.FAILS, f"装载门 FAIL: {vis.FAILS}"
    here = os.path.dirname(os.path.abspath(__file__))
    apps_dir = os.path.dirname(os.path.dirname(here))
    vtool = os.path.join(apps_dir, "body-sdk", "tools", "validate.py")
    if os.path.exists(vtool):
        subprocess.check_call([sys.executable, vtool, pkg1, "--dev"],
                              stdout=subprocess.DEVNULL)   # 独立校验器也须全绿

    # 整包合并: balance + sac(两 policy)+ 快脑(goal_stream=reach_goal → t_sac 行名命中)
    pkg3 = os.path.join(tmp, "pkg_multi_fb")
    pb.build([os.path.join(tmp, "t_balance"), os.path.join(tmp, "t_sac"),
              os.path.join(tmp, "t_fb")], pkg3, None, key)
    vis.FAILS.clear(); vis.WARNS.clear()
    vis.verify_local(pkg3)
    assert not vis.FAILS, f"三件套整包 FAIL: {vis.FAILS}"
    import tomllib
    pkgd = tomllib.load(open(os.path.join(pkg3, "body-package.toml"), "rb"))
    assert len(pkgd["policies"]["rows"]) == 2 and len(pkgd["fastbrains"]["rows"]) == 1

    # 负例: 快脑 goal_stream 指向不存在的策略行 → 诚实拒收
    import shutil
    fb_bad = os.path.join(tmp, "t_fb_bad")
    shutil.copytree(os.path.join(tmp, "t_fb"), fb_bad, dirs_exist_ok=True)
    mfb = json.load(open(os.path.join(fb_bad, "manifest.json"), encoding="utf-8"))
    mfb["goal_stream"] = "no_such_policy"
    with open(os.path.join(fb_bad, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(mfb, f, ensure_ascii=False)
    try:
        pb.build([os.path.join(tmp, "t_balance"), os.path.join(fb_bad)],
                 os.path.join(tmp, "pkg_reject"), None, key)
        raise AssertionError("goal_stream 不在包内应拒收")
    except SystemExit:
        pass

    # 负例: 伪造 manifest obs_dim 越域 → self_checks 拒
    import copy
    bad = copy.deepcopy(json.load(open(os.path.join(tmp, "t_balance", "manifest.json"),
                                       encoding="utf-8")))
    bad["obs_dim"] = 98
    bads = pb.self_checks(bad, pkg1)
    assert bads, "obs_dim=98 应被自检拒绝"

    # 整包: balance + sac 两 policy 合并(同 j=29)
    pkg2 = os.path.join(tmp, "pkg_multi")
    pb.build([os.path.join(tmp, "t_balance"), os.path.join(tmp, "t_sac")], pkg2, None, key)
    vis.FAILS.clear(); vis.WARNS.clear()
    vis.verify_local(pkg2)
    assert not vis.FAILS, f"整包装载门 FAIL: {vis.FAILS}"
    import tomllib
    rows = tomllib.load(open(os.path.join(pkg2, "body-package.toml"), "rb"))["policies"]["rows"]
    assert len(rows) == 2, f"整包应 2 策略行,实际 {len(rows)}"
    print("  export/package: 单包/整包合并/装载门/负例拒收 全部通过")
