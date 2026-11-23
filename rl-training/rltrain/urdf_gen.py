# -*- coding: utf-8 -*-
"""URDF 生成器(Sim2Sim 策略级对拍资产)——从编译后的训练变体 MJCF 导出。

来源 = mjcf_gen 产物(自由底座/手/物体)——关节序/限位/质量惯量逐项取自编译模型,
不做二次手写(单一来源)。PyBullet 重放使用(悬空根 = 浮动基)。

转换边界(诚实声明):
  - 关节锚点 jnt_pos 必须为 0(URDF 子连杆系=关节系);非 0 需中间固定关节,拒绝导出
  - armature 无 URDF 域,不导出——留在对拍残差,由 <5° 判据覆盖
  - capsule→cylinder(接触形状近似);plane = 环境不进本体
"""
from __future__ import annotations

import os
import sys
from math import asin, atan2

import mujoco

_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rltrain.g1_cfg import build_free_base_mjcf

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs", "g1", "assets")
OUT = os.path.join(ASSETS, "g1_pb.urdf")


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _rpy(q) -> str:
    """四元数(w,x,y,z) → ZYX 内旋 rpy 字符串(MuJoCo 与 URDF 同约定)。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    roll = atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return _fmt((roll, pitch, yaw))


def generate_urdf(xml: str | None = None, out: str = OUT) -> str:
    m = mujoco.MjModel.from_xml_path(xml or build_free_base_mjcf())
    parts = ['<?xml version="1.0"?>',
             '<robot name="g1_train_pb">']

    def link(name: str) -> None:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        mass = float(m.body_mass[bid])
        inertia = m.body_inertia[bid]
        ipos = m.body_ipos[bid]
        geoms = []
        for g in range(m.body_geomadr[bid], m.body_geomadr[bid] + m.body_geomnum[bid]):
            gtype = int(m.geom_type[g])
            size = m.geom_size[g]
            pos = m.geom_pos[g]
            if gtype == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue   # 平面 = 环境,不进本体
            if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                continue   # mesh 碰撞(官方模型)不导出——对拍窗 = 无接触纯惯量段,
                           # 惯量来自编译模型精确值;视觉壳不参与动力学
            if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                shape = f'<sphere radius="{size[0]:.6g}"/>'
            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                shape = f'<box size="{_fmt(2*size)}"/>'
            elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                shape = f'<cylinder radius="{size[0]:.6g}" length="{2*size[1]:.6g}"/>'
            else:   # capsule → cylinder(PB 兼容口径)
                length = float(m.geom_rbound[g]) if size[1] <= 0 else 2 * float(size[1])
                shape = f'<cylinder radius="{size[0]:.6g}" length="{length:.6g}"/>'
            # 位姿逐项带过(pos+quat→rpy): 初始姿态非恒等的几何,姿态丢失 = 接触位形错
            geoms.append(f'      <collision><origin xyz="{_fmt(pos)}" rpy="{_rpy(m.geom_quat[g])}"/>'
                         f'<geometry>{shape}</geometry></collision>')
        inert = (f'      <inertial><origin xyz="{_fmt(ipos)}" rpy="{_rpy(m.body_iquat[bid])}"/>\n'
                 f'        <mass value="{mass:.6g}"/>\n'
                 f'        <inertia ixx="{inertia[0]:.6g}" iyy="{inertia[1]:.6g}" izz="{inertia[2]:.6g}" '
                 f'ixy="0" ixz="0" iyz="0"/></inertial>\n') if mass > 0 else \
                '      <inertial><mass value="0"/><inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/></inertial>\n'
        parts.append(f'  <link name="{name}">\n{inert}' + "\n".join(geoms) + "\n  </link>")

    def _origin(bid: int) -> str:
        """父→子变换(编译模型 body_pos/quat)= URDF joint origin。"""
        return f'<origin xyz="{_fmt(m.body_pos[bid])}" rpy="{_rpy(m.body_quat[bid])}"/>'

    def moving_joint(child: str, parent: str, jid, bid: int) -> None:
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if float(abs(m.jnt_pos[jid]).max()) > 1e-9:
            raise SystemExit(f"[urdf_gen] 关节 {jname} 锚点 jnt_pos={m.jnt_pos[jid]}≠0:"
                             "URDF 子连杆系=关节系,需中间固定关节——诚实拒绝(训练变体不应出现)")
        axis = m.jnt_axis[jid]
        kind = "prismatic" if int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_SLIDE) else "revolute"
        limited = int(m.jnt_limited[jid]) == 1
        lim = (f'<limit lower="{m.jnt_range[jid][0]:.6g}" upper="{m.jnt_range[jid][1]:.6g}" '
               f'effort="300" velocity="12"/>') if limited else ""
        dof = int(m.jnt_dofadr[jid])   # 关节动力学逐项带过(缺阻尼 = PB 侧自由振动,系统差)
        dyn = (f'<dynamics damping="{m.dof_damping[dof]:.6g}" '
               f'friction="{m.dof_frictionloss[dof]:.6g}"/>')
        parts.append(f'  <joint name="{jname}" type="{kind}">\n'
                     f'    <parent link="{parent}"/><child link="{child}"/>\n'
                     f'    {_origin(bid)}\n'
                     f'    <axis xyz="{_fmt(axis)}"/>\n    {lim}\n    {dyn}\n  </joint>')

    # 遍历运动树(world 的直接子体 = 根链接)
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
    if int(m.body_jntnum[1]) == 0:
        parts.append('  <link name="world"/>')   # 固定底座变体: 骨盆焊接父 = world(必须存在)
    for b in range(1, m.nbody):
        name = bodies[b]
        if not name:
            continue
        link(name)
        parent = bodies[int(m.body_parentid[b])]
        if int(m.body_jntnum[b]) == 0:   # 焊接体:无 URDF joint = 悬空,必须补 fixed
            parts.append(f'  <joint name="{name}_fixed" type="fixed">\n'
                         f'    <parent link="{parent}"/><child link="{name}"/>\n'
                         f'    {_origin(b)}\n  </joint>')
            continue
        jid = int(m.body_jntadr[b])
        if int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            continue   # 浮动基——URDF 根 link 悬空即等价
        moving_joint(name, parent, jid, b)
    parts.append("</robot>")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))
    return out


if __name__ == "__main__":
    print("URDF:", generate_urdf())
