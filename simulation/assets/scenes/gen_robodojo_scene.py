#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""robodojo_table.xml 生成器——感知绑定在环验证场景（tri 本体 + 桌 + 目标盒 + 深度相机）。

相机位姿与引擎侧声明（robodojo 包 acquisition.mount_xyz_m/mount_rpy_deg）逐位同源：
换算权威 = bridge/engine_frames.py（macroview.rs 数值复刻）+ MuJoCo 相机四元数推导。
产出：本文件同目录 robodojo_table.xml（幂等重生成）。
"""
from __future__ import annotations

import sys
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[2]   # simulation/
sys.path.insert(0, str(SIM_ROOT))

from bridge.engine_frames import (  # noqa: E402
    engine_base_to_world, mat_vec, mujoco_camera_quat,
)

# ── 引擎侧声明（与 robodojo 包 acquisition 段同源——单一权威）──
MOUNT_XYZ = [0.35, 0.0, 1.2]
MOUNT_RPY_DEG = [0.0, 55.0, 0.0]
TABLE_TOP_Z = 0.75       # 桌面高（与 synthetic_box 验收场景同源约定）
TABLE_CENTER = [0.665, 0.0]   # 光轴与桌平面交点（供数几何对齐）
TARGET_BASE = [0.665, 0.0]   # 目标盒基点（盒底 = 桌面）
TARGET_HALF = 0.05       # 10cm 盒（与 synthetic_box 同源）
DISTRACTOR_OFFSET = [0.16, 0.12]  # 干扰盒偏移（clutter 变体）


def main() -> int:
    # 视轴自检：-Z_mj 须 = 引擎视轴 = R_e·[0,0,1]（引擎 z 前）
    r_e = engine_base_to_world(MOUNT_RPY_DEG)
    view = mat_vec(r_e, [0.0, 0.0, 1.0])
    t = (TABLE_TOP_Z - MOUNT_XYZ[2]) / view[2]
    hit = [MOUNT_XYZ[i] + view[i] * t for i in range(3)]
    assert abs(hit[0] - TABLE_CENTER[0]) < 0.02 and abs(hit[1] - TABLE_CENTER[1]) < 0.02, \
        f"光轴-桌交点 {hit[:2]} 偏离桌中心 {TABLE_CENTER}"

    w, x, y, z = mujoco_camera_quat(MOUNT_RPY_DEG)
    target_z = TABLE_TOP_Z + TARGET_HALF
    dx = [TARGET_BASE[0] + DISTRACTOR_OFFSET[0], TARGET_BASE[1] + DISTRACTOR_OFFSET[1], target_z]

    # 生成策略 = tri_body 原文最小注入（保其 <option>/<default>/资产全量——
    # 求解器稳定性不重建：integrator/阻尼缺任一即 QACC 失稳，实测）。
    tri = Path(__file__).resolve().parents[3] / "apps/device-agent-rs/fixtures/tri-family/assets/tri_body.xml"
    body = tri.read_text(encoding="utf-8")

    inject = f"""    <!-- ── robodojo 绑定验证注入（gen_robodojo_scene.py 数值生成，勿手改）── -->
    <!-- 深度相机：pos/quat = 引擎声明（mount_xyz/rpy）数值推导——权威 bridge/engine_frames.py -->
    <camera name="depth_cam" pos="{MOUNT_XYZ[0]} {MOUNT_XYZ[1]} {MOUNT_XYZ[2]}"
            quat="{w:.9f} {x:.9f} {y:.9f} {z:.9f}" fovy="39.6"/>

    <!-- 桌：顶面 {TABLE_TOP_Z}m，中心 = 光轴-桌交点 -->
    <body name="table" pos="{TABLE_CENTER[0]} {TABLE_CENTER[1]} {TABLE_TOP_Z - 0.05:.3f}">
      <geom name="table_geom" type="box" size="0.35 0.35 0.05" rgba="0.55 0.4 0.25 1"
            contype="1" conaffinity="1"/>
    </body>

    <!-- 目标盒（静态——绑定验证变体经 resident /variant 路由改 body_pos/geom_size） -->
    <body name="target_box" pos="{TARGET_BASE[0]} {TARGET_BASE[1]} {target_z:.3f}">
      <geom name="target_geom" type="box" size="{TARGET_HALF} {TARGET_HALF} {TARGET_HALF}"
            mass="0.4" rgba="0.2 0.5 0.9 1" contype="1" conaffinity="1"/>
    </body>

    <!-- 干扰盒（clutter 回合近置偏移；非 clutter 回合被移出视场角） -->
    <body name="distractor_box" pos="{dx[0]:.3f} {dx[1]:.3f} {dx[2]:.3f}">
      <geom name="distractor_geom" type="box" size="{TARGET_HALF} {TARGET_HALF} {TARGET_HALF}"
            mass="0.4" rgba="0.8 0.3 0.2 1" contype="1" conaffinity="1"/>
    </body>
"""
    anchor = body.index("<worldbody>") + len("<worldbody>")
    xml = body[:anchor] + "\n" + inject + body[anchor:]
    xml = xml.replace('<mujoco model="biped_29dof">', '<mujoco model="robodojo_table">')
    out = Path(__file__).with_name("robodojo_table.xml")
    out.write_text(xml, encoding="utf-8", newline="\n")
    print(f"OK {out} quat=({w:.4f},{x:.4f},{y:.4f},{z:.4f}) 光轴-桌交点={[round(v, 3) for v in hit[:2]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
