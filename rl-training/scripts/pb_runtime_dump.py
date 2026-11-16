#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PyBullet 运行时动力学 dump——PB 实际装载值(非 XML 声明值)逐项打印。

XML 级 diff 全绿但行为发散时用:PB 会静默替换非法惯量/以自有口径解析链,
本脚本给出 PB 侧真相,与 sim2sim_diff(编译 MJCF)对照定位。
"""
import os
import sys

import pybullet as p

_HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(os.path.dirname(_HERE), "envs", "g1", "assets", "g1_pb.urdf")

p.connect(p.DIRECT)
robot = p.loadURDF(URDF, flags=p.URDF_USE_INERTIA_FROM_FILE)
n = p.getNumJoints(robot)
print(f"links(含基座) = {n + 1}")

total_mass = 0.0
for i in range(-1, n):
    name = "pelvis(base)" if i == -1 else \
        p.getJointInfo(robot, i)[12].decode()
    di = p.getDynamicsInfo(robot, i)
    total_mass += di[0]
    print(f"[link {i:2d}] {name:14s} mass={di[0]:.6g} friction={di[1]:.3g} "
          f"inertia={tuple(round(v, 8) for v in di[2])} "
          f"inertial_pos={tuple(round(v, 6) for v in di[3])} "
          f"inertial_orn={tuple(round(v, 4) for v in di[4])}")

for j in range(n):
    info = p.getJointInfo(robot, j)
    print(f"[joint {j:2d}] {info[1].decode():32s} type={info[2]} damp={info[6]:.4g} "
          f"fric={info[7]:.4g} lim=({info[8]:.4g},{info[9]:.4g}) "
          f"axis={info[13]} parentPos={tuple(round(v, 6) for v in info[14])} "
          f"parentOrn={tuple(round(v, 4) for v in info[15])}")
print(f"total_mass = {total_mass:.6g}")
p.disconnect()
