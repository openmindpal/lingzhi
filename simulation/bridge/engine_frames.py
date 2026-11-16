"""引擎坐标系约定（sensorbus::macroview.rs 数值复刻）——仿真侧与引擎侧的单一换算权威。

两端同源的契约面：
  引擎：cam_point_to_base(p, rpy_deg, mount_xyz) = R(rpy ZYX 度)·C·p + mount_xyz
        （C = CV_TO_ALIGNED——OpenCV 相机系 x右/y下/z前 → 对齐系 x前/y左/z上）
  本模块：同一 R/C/mount 数值 + 逆变换（世界点 → 引擎基座系，GT 判据用）
          + MuJoCo 相机四元数推导（R_mj = R_e·diag(1,-1,-1)——mj 相机沿 -Z 看）。
改引擎约定 = 只改本文件与 macroview.rs 两处，gen 场景脚本与 shim 自动跟随。
"""
from __future__ import annotations

import math

CV_TO_ALIGNED = [
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
]


def rpy_deg_to_matrix(rpy_deg):
    """RPY（度，ZYX 序）→ 旋转矩阵（macroview.rs rpy_deg_to_matrix 逐位复刻）。"""
    r, p, y = (math.radians(v) for v in rpy_deg)
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def mat_vec(m, v):
    return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def transpose(m):
    return [list(row) for row in zip(*m)]


def engine_base_to_world(rpy_deg):
    """引擎基座系 → 世界系旋转（R_e = R(rpy)·C）。"""
    return mat_mul(rpy_deg_to_matrix(rpy_deg), CV_TO_ALIGNED)


def world_to_engine_base(p_w, mount_xyz, rpy_deg):
    """世界系点 → 引擎基座系点：p_e = R_eᵀ·(p_w − mount_xyz)。GT 判据权威换算。"""
    r_et = transpose(engine_base_to_world(rpy_deg))
    d = [p_w[i] - mount_xyz[i] for i in range(3)]
    return mat_vec(r_et, d)


def mat2quat(m):
    """旋转矩阵 → MuJoCo 四元数 (w, x, y, z)——Shepperd 数值稳定法。"""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def mujoco_camera_quat(mount_rpy_deg):
    """声明 RPY → MuJoCo 相机四元数（R_mj = R_e·diag(1,-1,-1)：mj 沿 -Z 看、+Y 图像上）。"""
    r_e = engine_base_to_world(mount_rpy_deg)
    flip = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    return mat2quat(mat_mul(r_e, flip))
