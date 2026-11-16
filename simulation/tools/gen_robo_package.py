#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gen_robo_package.py——robodojo 验证包生成（tri-family 夹具 → http_depth 深度供数）。

与 gen_robodojo_scene.py 同源权威：相机声明（device/intrinsics/mount）三处一致——
  包声明（本脚本写入）↔ 场景相机位姿（gen_robodojo_scene.py 推导）↔ shim 供数（同声明）。
产出 .tmp/robo-package（签名后 docker cp 入容器 /pkg/robodojo）。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "apps/device-agent-rs/fixtures/tri-family"
DST = REPO / ".tmp/robo-package"
KEY = REPO / ".tmp/robo_dev_key.pem"

HTTP_DEPTH_BLOCK = '''# ── 深度供数面（robodojo 绑定验证）：MuJoCo 桥真渲染深度——HTTP Z16 逐位契约。
# 位姿/内参与场景相机（gen_robodojo_scene.py）同源——单一权威 bridge/engine_frames.py
[[acquisition.cameras]]
kind = "http_depth"
device = "http://127.0.0.1:8284/depth.raw"
depth = true
intrinsics = { fx = 400.0, fy = 400.0, cx = 160.0, cy = 120.0 }
mount_xyz_m = [0.35, 0.0, 1.2]
mount_rpy_deg = [0.0, 55.0, 0.0]
'''

SYNTHETIC_BLOCK_MARK = '[[acquisition.cameras]]\nkind = "synthetic_box"'


def main() -> int:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)

    toml = DST / "body-package.toml"
    t = toml.read_text(encoding="utf-8")

    # 1) 包名改 robodojo-arm（tri-family 夹具沿袭 unitree-g1 命名——按 [core] 段锚替换）
    import re
    t, n = re.subn(r'package_name = "[^"]+"', 'package_name = "robodojo-arm"', t, count=1)
    assert n == 1, "package_name 锚未命中"

    # 2) synthetic_box 深度块 → http_depth 块（从块头到下一个段边界）
    i = t.find(SYNTHETIC_BLOCK_MARK)
    assert i >= 0, "synthetic_box 块未找到"
    j = t.find("\n[", i + 1)              # 下一顶层段
    j = len(t) if j < 0 else j + 1
    t = t[:i] + HTTP_DEPTH_BLOCK + t[j:]
    toml.write_text(t, encoding="utf-8", newline="\n")

    # 3) dev 密钥签名（run-fullstack 同款信任链）
    if not KEY.exists():
        subprocess.run([sys.executable, str(REPO / "apps/body-packages/tools/sign_package.py"),
                        "--generate", str(KEY)], check=True, capture_output=True)
    r = subprocess.run([sys.executable, str(REPO / "apps/body-packages/tools/sign_package.py"),
                        str(DST), str(KEY)], check=True, capture_output=True, text=True)
    pubkey = next((ln.split("=", 1)[1] for ln in r.stdout.splitlines()
                   if ln.startswith("PUBKEY_B64=")), "")
    assert pubkey, "签名输出缺 PUBKEY_B64"
    print(f"OK {DST} PUBKEY={pubkey[:24]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
