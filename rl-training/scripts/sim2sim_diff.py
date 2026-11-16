#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sim2Sim 资产参数机械比对——编译 MJCF ↔ 导出 URDF 逐项 diff。

对拍门(sim2sim_check)的配套工具:行为发散时先定位"哪一项参数不一致",不靠猜。
比对全部取编译后值(单一来源 = mjcf_gen 产物经 MuJoCo 编译):

  运动树  — 父→子变换(body_pos/quat ↔ joint origin xyz/rpy)、锚点边界(jnt_pos=0)
  惯量    — 质量/主惯量/质心偏移/主轴姿态(body_iquat ↔ inertial origin rpy)
  几何    — collision 类型/尺寸/位姿(geom_pos/geom_quat ↔ collision origin)
  动力学  — 关节阻尼/摩擦损耗(dof_damping/dof_frictionloss ↔ <dynamics>)
  轴/限位 — jnt_axis / jnt_range(rad) ↔ <axis>/<limit>

已知转换口径(不算差异): capsule→cylinder、plane 跳过、根体(自由底座)跳过
origin 比对(PB 显式设基座位姿)、effort/velocity 为声明占位(力矩控制不用)。
退出码: 0 = 全一致, 1 = 存在差异。
"""
import os
import sys
import xml.etree.ElementTree as ET
from math import asin, atan2

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from rltrain.g1_cfg import build_free_base_mjcf

URDF = os.path.join(os.path.dirname(_HERE), "envs", "g1", "assets", "g1_pb.urdf")
TOL = 1e-5
_bad = 0


def _rpy_from_quat(q) -> tuple[float, float, float]:
    """四元数(w,x,y,z) → ZYX 内旋 rpy(MuJoCo 与 URDF 同约定)。"""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return (atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _close(a, b, tol=TOL) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) <= tol))


def _check(label: str, ok: bool, detail: str = "") -> None:
    global _bad
    if ok:
        print(f"  [OK] {label}")
    else:
        _bad += 1
        print(f"  [DIFF] {label}  {detail}")


def _shape_of(gtype: int, size, capsule_to_cylinder: bool):
    """MJCF geom → 预期 URDF 形状文本(复刻生成器转换口径)。"""
    import mujoco
    G = mujoco.mjtGeom
    t = int(gtype)
    if t == int(G.mjGEOM_PLANE):
        return None
    if t == int(G.mjGEOM_SPHERE):
        return ("sphere", {"radius": float(size[0])})
    if t == int(G.mjGEOM_BOX):
        return ("box", {"size": tuple(2 * np.asarray(size, float))})
    if t == int(G.mjGEOM_CYLINDER):
        return ("cylinder", {"radius": float(size[0]), "length": 2 * float(size[1])})
    if t == int(G.mjGEOM_CAPSULE) and capsule_to_cylinder:
        return ("cylinder", {"radius": float(size[0]), "length": 2 * float(size[1])})
    return ("unsupported", {})


def _shape_of_urdf(geom_el) -> tuple[str, dict]:
    box = geom_el.find("geometry")
    if box is None or len(box) == 0:
        return ("malformed(缺 <geometry> 包裹)", {})
    g = box[0]
    tag = g.tag
    if tag == "sphere":
        return ("sphere", {"radius": float(g.get("radius"))})
    if tag == "box":
        return ("box", {"size": tuple(float(v) for v in g.get("size").split())})
    if tag == "cylinder":
        return ("cylinder", {"radius": float(g.get("radius")), "length": float(g.get("length"))})
    return (tag, {})


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(build_free_base_mjcf())
    root = ET.parse(URDF).getroot()
    links = {el.get("name"): el for el in root.findall("link")}
    joints_by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    print(f"[diff] {os.path.basename(build_free_base_mjcf())} ↔ {os.path.basename(URDF)}")

    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if not name:
            continue
        print(f"── {name} ──")
        lk = links.get(name)
        if lk is None:
            _check(f"{name}: link 存在", False, "URDF 缺 link")
            continue

        # ── 运动树:父→子变换(根体/自由底座跳过) ──
        jn = int(m.body_jntnum[b])
        jid = int(m.body_jntadr[b]) if jn > 0 else -1
        is_free = jn > 0 and int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
        if is_free:
            _check(f"{name}: 自由底座(根)——URDF 应无 joint",
                   name not in joints_by_child)
        elif jn == 0:
            j = joints_by_child.get(name)
            _check(f"{name}: 无关节体→fixed joint",
                   j is not None and j.get("type") == "fixed",
                   "URDF 缺 fixed joint(焊接体)" if j is None else f"type={j.get('type')}")
        else:
            _check(f"{name}: 关节锚点 jnt_pos=0(转换边界)",
                   _close(m.jnt_pos[jid], np.zeros(3)),
                   f"jnt_pos={m.jnt_pos[jid]}≠0——URDF 需中间固定关节,生成器必须诚实报错")
            j = joints_by_child.get(name)
            _check(f"{name}: URDF joint 存在", j is not None)
            if j is not None:
                org = j.find("origin")
                want_xyz, want_rpy = m.body_pos[b], _rpy_from_quat(m.body_quat[b])
                got_xyz = np.array([float(v) for v in (org.get("xyz") or "0 0 0").split()])
                got_rpy = np.array([float(v) for v in (org.get("rpy") or "0 0 0").split()])
                _check(f"{name}: joint origin xyz=body_pos", _close(got_xyz, want_xyz),
                       f"urdf={got_xyz} vs mjcf={want_xyz}")
                _check(f"{name}: joint origin rpy=body_quat", _close(got_rpy, want_rpy),
                       f"urdf={got_rpy} vs mjcf={want_rpy}")
                axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                _check(f"{name}: 关节轴", _close(axis, m.jnt_axis[jid]))
                limited = int(m.jnt_limited[jid]) == 1
                lim = j.find("limit")
                if limited:
                    _check(f"{name}: 限位(rad)", lim is not None and
                           _close([float(lim.get("lower")), float(lim.get("upper"))],
                                  m.jnt_range[jid]))
                dof = int(m.jnt_dofadr[jid])
                dyn = j.find("dynamics")
                want_damp = float(m.dof_damping[dof])
                want_fric = float(m.dof_frictionloss[dof])
                got_damp = float(dyn.get("damping")) if dyn is not None and dyn.get("damping") else 0.0
                got_fric = float(dyn.get("friction")) if dyn is not None and dyn.get("friction") else 0.0
                _check(f"{name}: 关节阻尼({want_damp})", _close(got_damp, want_damp),
                       f"urdf={got_damp} vs mjcf={want_damp}")
                _check(f"{name}: 关节摩擦损耗({want_fric})", _close(got_fric, want_fric),
                       f"urdf={got_fric} vs mjcf={want_fric}")

        # ── 惯量 ──
        inert = lk.find("inertial")
        mass = float(m.body_mass[b])
        if mass > 0:
            im = inert.find("mass") if inert is not None else None
            _check(f"{name}: 质量({mass:.4g})", im is not None and _close(float(im.get("value")), mass))
            iorg = inert.find("origin") if inert is not None else None
            got_xyz = np.array([float(v) for v in (iorg.get("xyz") or "0 0 0").split()]) if iorg is not None else np.zeros(3)
            _check(f"{name}: 质心 ipos", _close(got_xyz, m.body_ipos[b]))
            iquat = m.body_iquat[b]
            identity = _close(iquat, [1, 0, 0, 0])
            got_rpy = np.array([float(v) for v in (iorg.get("rpy") or "0 0 0").split()]) if iorg is not None and iorg.get("rpy") else np.zeros(3)
            if identity:
                _check(f"{name}: 主轴=体轴(iquat 恒等,rpy 应 0)", _close(got_rpy, np.zeros(3)))
            else:
                _check(f"{name}: 主轴姿态(iquat→rpy)", _close(got_rpy, _rpy_from_quat(iquat)))
            it = inert.find("inertia") if inert is not None else None
            want_diag = m.body_inertia[b]
            got_diag = np.array([float(it.get(k)) for k in ("ixx", "iyy", "izz")]) if it is not None else np.zeros(3)
            got_off = np.array([float(it.get(k)) for k in ("ixy", "ixz", "iyz")]) if it is not None else np.ones(3)
            _check(f"{name}: 主惯量对角", _close(got_diag, want_diag),
                   f"urdf={got_diag} vs mjcf={want_diag}")
            _check(f"{name}: 非对角=0(主轴系)", _close(got_off, np.zeros(3)))
        else:
            _check(f"{name}: 零质量体", inert is not None)

        # ── collision 几何 ──
        mj_geoms = []
        for g in range(int(m.body_geomadr[b]), int(m.body_geomadr[b]) + int(m.body_geomnum[b])):
            if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
                continue
            mj_geoms.append(g)
        cols = lk.findall("collision")
        _check(f"{name}: collision 数({len(mj_geoms)})", len(cols) == len(mj_geoms),
               f"urdf={len(cols)} vs mjcf={len(mj_geoms)}")
        for g, col in zip(mj_geoms, cols):
            want = _shape_of(int(m.geom_type[g]), m.geom_size[g], capsule_to_cylinder=True)
            got = _shape_of_urdf(col)
            ok_shape = want[0] == got[0] and all(_close(got[1].get(k, np.nan), v) for k, v in want[1].items())
            corg = col.find("origin")
            got_xyz = np.array([float(v) for v in (corg.get("xyz") or "0 0 0").split()]) if corg is not None else np.zeros(3)
            got_rpy = np.array([float(v) for v in (corg.get("rpy") or "0 0 0").split()]) if corg is not None and corg.get("rpy") else np.zeros(3)
            want_rpy = _rpy_from_quat(m.geom_quat[g])
            ok_pos = _close(got_xyz, m.geom_pos[g]) and _close(got_rpy, want_rpy)
            _check(f"{name}: geom{g} {want[0]} 位姿/尺寸", ok_shape and ok_pos,
                   f"shape urdf={got} vs mjcf={want}; pos urdf={got_xyz} vs mjcf={m.geom_pos[g]}; "
                   f"rpy urdf={got_rpy} vs mjcf={want_rpy}")

    print(f"[diff] {'全一致' if _bad == 0 else f'{_bad} 项差异'}")
    sys.exit(0 if _bad == 0 else 1)


if __name__ == "__main__":
    main()
