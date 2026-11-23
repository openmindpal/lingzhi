#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sim2Sim 对拍编排(架构 L3→L5 交叉验证)。

模式:
  smoke  — 既有跨引擎基准(simulation/assets/models/sim2sim_pendulum.{xml,urdf},
           MuJoCo↔PyBullet,判据引用 simulation/suites/sim2sim)——验证工具链
  policy — 对拍门(需 envs/g1/assets/g1_pb.urdf,缺则由训练变体诚实生成):
           默认 = 解析探针——固定基座重力响应窗(前 PB_WIN_BEATS 拍)逐拍逐关节
           比对 <5°,配 scripts/sim2sim_diff(XML 级逐项参数比对);
           --onnx = 策略在环(参考语义,混沌放大警告,判据不设门)

诚实边界: pybullet 未安装 = 退出码 2 + 安装指引(不假装对拍)。
"""
import argparse
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

_REPO = os.path.join(os.path.dirname(_HERE), "..")
PEN_XML = os.path.join(_REPO, "simulation", "assets", "models", "sim2sim_pendulum.xml")
PEN_URDF = os.path.join(_REPO, "simulation", "assets", "models", "sim2sim_pendulum.urdf")
G1_URDF = os.path.join(os.path.dirname(_HERE), "envs", "g1", "assets", "g1_pb.urdf")
COMPARE = os.path.join(_REPO, "simulation", "suites", "sim2sim", "compare.py")


def require_pybullet() -> None:
    try:
        import pybullet  # noqa: F401
    except Exception:
        print("[sim2sim] pybullet 未安装——不假装对拍。安装: pip install pybullet")
        print("[sim2sim] 既有的平面基准对拍见 simulation/suites/sim2sim/compare.py")
        sys.exit(2)


def mode_smoke() -> None:
    require_pybullet()
    for p in (PEN_XML, PEN_URDF):
        if not os.path.exists(p):
            print(f"[sim2sim] 基准资产缺失: {p}")
            sys.exit(2)
    print(f"[sim2sim] 工具链基准(摆): {os.path.basename(PEN_XML)} ↔ {os.path.basename(PEN_URDF)}")
    sys.exit(subprocess.call([sys.executable, COMPARE]))


def mode_policy(run: str, seconds: float, onnx: str | None = None) -> None:
    require_pybullet()
    # 对拍两侧同源: MuJoCo 侧 = 固定底座变体(训练变体去 freejoint/去地板——
    # 悬空无接触重力响应段),URDF 由同一变体生成(基座由 body_spec.base 决定:
    # official = 官方资产,smoke = 冒烟拓扑)
    from rltrain.mjcf_gen import build_fixed_base_variant
    FIXED = build_fixed_base_variant()
    from rltrain.urdf_gen import generate_urdf
    generate_urdf(xml=FIXED, out=G1_URDF)   # 确定性重生成——不读缓存防基座切换后错配
    print(f"[sim2sim] URDF 由固定底座变体生成: {G1_URDF}")
    import json
    import numpy as np
    import onnxruntime as ort
    from rltrain.g1_cfg import G1Cfg

    mf = json.load(open(os.path.join(run, "manifest.json"), encoding="utf-8"))
    cfg = G1Cfg()
    # exports 文件按 run 目录相对定位——manifest 记录的生成时绝对路径不可移植
    # (换机/仓库布局变更即失效;文件名是可移植的事实,位置由 --run 决定)
    fname = mf["exports"][-1]["file"].replace("\\", "/").split("/")[-1]
    path = os.path.join(run, "exports", fname)
    if not os.path.exists(path):
        raise SystemExit(f"[sim2sim] 导出权重缺失: {path}——先跑 export_onnx")
    # 动作源二选一(对拍测的是引擎不是策略): 默认 = 解析探针(sess=None);
    # --onnx = 训练策略在环(参考语义,混沌放大警告)。
    # onnx_resolved 必须直取 onnx——混入 path 兜底会让探针分支永不生效
    sess = iname = None
    onnx_resolved = onnx
    if onnx_resolved:
        sess = ort.InferenceSession(onnx_resolved, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
    tag = f"policy={os.path.basename(onnx_resolved)}(混沌放大警告)" if onnx else "解析探针(速度脉冲)"
    print(f"[sim2sim] {tag} mujoco={os.path.basename(FIXED)}(固定基座) pb={os.path.basename(G1_URDF)}(useFixedBase)")

    # MuJoCo 在环轨迹(固定基座 = 训练变体去 freejoint/去地板——骨盆焊死,重力经支撑加载关节)
    import mujoco
    m = mujoco.MjModel.from_xml_path(FIXED)
    # 判定窗 = 无接触重力响应段: 自碰撞冲量属求解器差异,不属模型 parity
    # (与"落点接触退出判据"同原则)。MJ 默认开自碰撞而 PB 默认关——
    # 模型级禁接触使两引擎口径一致
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)
    from rltrain.obs_assembler import ObsAssembler
    from rltrain.tasks import load_task
    asm = ObsAssembler(cfg.j, mf["obs_mode"], 2)
    probe_goal = None
    if "goal" in mf["obs_mode"]:
        box = np.asarray(load_task(mf["task"])["goal_box"], dtype=np.float64)
        probe_goal = np.array([(box[i][0] + box[i][1]) / 2 for i in range(3)] + [0.5])
    hz = mf["frequency_hz"]
    n_sub = int(round((1.0 / hz) / m.opt.timestep))
    traj = []
    steps = int(seconds * hz)
    lo = np.asarray([cfg.limits[n][0] + 0.05 for n in cfg.joints])
    hi = np.asarray([cfg.limits[n][1] - 0.05 for n in cfg.joints])
    qadr, dadr = [], []
    for name in cfg.joints:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr.append(int(m.jnt_qposadr[jid]))
        dadr.append(int(m.jnt_dofadr[jid]))
    qadr, dadr = np.asarray(qadr), np.asarray(dadr)
    if sess is None:
        # 解析探针 = 无源纯惯量段: 判定窗恰为 URDF 可转换子集的刚体动力学比对
        # ——armature(URDF 契约外)与 dof_damping/frictionloss(两引擎实现语义
        # 不同)双侧清零,PD 不施力;阻尼值 parity 由 sim2sim_diff 精确承担。
        # 策略腿保留原模型+PD(参考语义)
        m.dof_armature[dadr] = 0.0
        m.dof_damping[dadr] = 0.0
        m.dof_frictionloss[dadr] = 0.0
    # 脉冲方向 = 关节域内侧(上界≤0 的关节如肘,0 在界上,取负向)
    _lo, _hi = (np.asarray(a) for a in cfg.limit_arrays())
    sgn = np.where(_hi <= 0.0, -1.0, 1.0)
    d.qvel[dadr] = PB_DQ0 * sgn
    mujoco.mj_forward(m, d)
    for _ in range(steps):
        q = d.qpos[qadr] + np.random.normal(0, 1e-4, cfg.j)
        dq = d.qvel[dadr]
        tau = d.qfrc_applied[dadr]
        obs = asm.assemble(q, dq, tau, _imu(d), None, probe_goal)
        if sess is not None:
            act = sess.run(None, {iname: obs[None].astype(np.float32)})[0][0]
        else:
            act = np.zeros(cfg.j)   # 解析探针: 中性保持
        tgt = np.clip(act, lo, hi)
        for _ in range(n_sub):
            if sess is not None:   # 策略腿: PD 在环(参考);解析探针: 无源被动(零力矩)
                d.qfrc_applied[dadr] = cfg.kp * (tgt - d.qpos[qadr]) - cfg.kd * d.qvel[dadr]
            mujoco.mj_step(m, d)
        traj.append(d.qpos[qadr].copy())
    out = os.path.join(run, "exports", "sim2sim_mj_traj.npz")
    np.savez(out, traj=np.asarray(traj), hz=hz, lo=lo, hi=hi,
             mj_dt=m.opt.timestep, n_sub=n_sub,
             damping=m.dof_damping[dadr].copy(),
             armature=m.dof_armature[dadr].copy(),
             goal=(probe_goal if probe_goal is not None else np.zeros(4)),
             obs_mode=np.asarray(mf["obs_mode"]))
    print(f"[sim2sim] MuJoCo 轨迹已存: {out}")

    # ── PyBullet 重放腿(同初态/同策略/同 PD 律) + <5° 判据 ──
    verdict = replay_pb(run, seconds, onnx_resolved)
    print(f"[sim2sim] verdict: {verdict}")
    if onnx_resolved:
        # 文件头语义: --onnx = 策略在环诊断参考(混沌放大,判据不设门)——
        # 训练策略逐拍放大引擎微小差异,门 = 解析探针模式
        print("[sim2sim] 策略在环 = 诊断参考(混沌放大,不设门)——对拍门以解析探针为准")
        sys.exit(0)
    sys.exit(0 if verdict["pass"] else 1)


def _imu(d) -> np.ndarray:
    """传感序 = xml 声明序: framequat(4)/gyro(3)/accel(3) → imu(10)。"""
    sd = d.sensordata
    return np.concatenate([sd[0:4], sd[4:7], sd[7:10]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="smoke", choices=["smoke", "policy", "policy-replay"])
    ap.add_argument("--run", default=None)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--onnx", default=None,
                    help="训练策略 onnx(混沌放大警告)——缺省 = 解析探针(中性保持)")
    args = ap.parse_args()
    if args.mode == "smoke":
        mode_smoke()
    elif args.mode == "policy-replay":
        if not args.run:
            ap.error("--mode policy-replay 需要 --run")
        replay_pb(args.run, args.seconds, args.onnx)
        sys.exit(0)
    else:
        if not args.run:
            ap.error("--mode policy 需要 --run")
        mode_policy(args.run, args.seconds, args.onnx)




# ================================================================== */
#  PyBullet 重放腿(策略级 Sim2Sim 第二引擎腿)                          */
# ================================================================== */

PB_MAX_DEG = 5.0    # 对拍判据: 判定窗内逐拍逐关节 |Δ角度| < 5°(与架构 §5 同一门)
PB_WIN_BEATS = 10   # 判定窗 = 固定基座重力响应段(pelvis 焊死,10 拍,零力矩,双侧
                    # armature/阻尼/摩擦损耗清零):重力经固定支撑加载关节,对
                    # 质量/质心/惯量/几何误差灵敏且确定。悬空探针不可用——均匀
                    # 重力下关节零力矩(等效原理);落点接触属求解器特性不属模型
                    # parity,均退出判据。阻尼值 parity 由 scripts/sim2sim_diff
                    # 承担(两引擎阻尼实现语义不同)
PB_DQ0 = 1.0        # 判定窗初始关节速度脉冲幅值(rad/s);方向取关节域内侧
                    # (上界≤0 的关节如肘,q=0 在界上,正向脉冲即撞限位——限位
                    # 求解差异属引擎实现,不属模型 parity)

def replay_pb(run: str, seconds: float, onnx: str | None = None) -> dict:
    """PyBullet 同初态/同策略/同 PD 律重放,与 MuJoCo 轨迹逐点比对。

    判定窗(前 PB_WIN_BEATS 拍) = 固定基座重力响应段(pelvis 焊死+速度脉冲,零
    力矩,双侧 armature/阻尼清零):重力载荷下 URDF 可转换子集的刚体动力学比
    对;全 horizon 轨迹入报告作诊断。模型参数经 scripts/sim2sim_diff 逐项对齐
    (阻尼值 parity 由其承担——两引擎阻尼实现语义不同;armature 经链接惯量对角
    补齐仅属策略腿)。
    跨引擎传感近似:imu 块 = 基座链路姿态/角速度/线加速度(比力,含重力反应)
    ——MuJoCo 侧为 site 传感(已知近似,诚实标注;解析探针不用 obs)。
    """
    import json
    import pybullet as p
    import pybullet_data
    import onnxruntime as ort
    from rltrain.g1_cfg import G1Cfg
    from rltrain.obs_assembler import ObsAssembler
    from rltrain.tasks import load_task

    npz = np.load(os.path.join(run, "exports", "sim2sim_mj_traj.npz"))
    mj_traj, hz = npz["traj"], float(npz["hz"])
    lo, hi = npz["lo"], npz["hi"]
    mj_dt, n_sub = float(npz["mj_dt"]), int(npz["n_sub"])
    damping, armature = npz["damping"], npz["armature"]
    goal = npz["goal"]
    obs_mode = str(npz["obs_mode"])
    mf = json.load(open(os.path.join(run, "manifest.json"), encoding="utf-8"))
    cfg = G1Cfg()
    fname = mf["exports"][-1]["file"].replace("\\", "/").split("/")[-1]
    onnx_path = os.path.join(run, "exports", fname)
    sess = iname = None
    if onnx:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name

    if not os.path.exists(G1_URDF):
        from rltrain.urdf_gen import generate_urdf
        generate_urdf(out=G1_URDF)

    cid = p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)   # 无接触重力响应段(与 MuJoCo 腿同口径)
        p.setTimeStep(mj_dt)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf", useFixedBase=True)
        # 滑摩擦显式对齐编译模型(MuJoCo 默认 1.0,几何对取 max)
        p.changeDynamics(plane, -1, linearDamping=0.0, angularDamping=0.0, lateralFriction=1.0)
        robot = p.loadURDF(G1_URDF, basePosition=[0, 0, 1.0],
                           flags=p.URDF_USE_INERTIA_FROM_FILE, useFixedBase=True)
        n_j = p.getNumJoints(robot)
        name2idx = {}
        for j in range(n_j):
            info = p.getJointInfo(robot, j)
            name2idx[info[1].decode()] = j
            # 体阻尼清零(MuJoCo 无体阻尼项);滑摩擦对齐
            p.changeDynamics(robot, j, linearDamping=0.0, angularDamping=0.0, lateralFriction=1.0)
        jidx = [name2idx[n] for n in cfg.joints]   # 缺关节 = URDF 与包不同源,诚实报错
        # 关节动力学: URDF <dynamics> 已带编译值;两引擎内阻尼实现语义不同,
        # 解析探针清零退出判定窗(值 parity 由 sim2sim_diff 承担);armature
        # 经惯量对角补齐仅属策略腿(参考语义,交叉轴为近似)
        for i, ji in enumerate(jidx):
            p.changeDynamics(robot, ji,
                             jointDamping=0.0 if sess is None else float(damping[i]))
            if sess is not None:
                inert = p.getDynamicsInfo(robot, ji)[2]
                arm = float(armature[i])
                p.changeDynamics(robot, ji,
                                 localInertiaDiagonal=(inert[0] + arm, inert[1] + arm, inert[2] + arm))
        # 关闭默认位置电机(TORQUE_CONTROL 直驱 PD)
        for ji in jidx:
            p.setJointMotorControl2(robot, ji, p.VELOCITY_CONTROL, force=0.0)
        p.resetBasePositionAndOrientation(robot, [0, 0, 1.0], [0, 0, 0, 1])
        _lo, _hi = (np.asarray(a) for a in cfg.limit_arrays())
        sgn = np.where(_hi <= 0.0, -1.0, 1.0)   # 脉冲方向 = 关节域内侧(与 MuJoCo 腿同式)
        for i, ji in enumerate(jidx):
            p.resetJointState(robot, ji, 0.0, PB_DQ0 * float(sgn[i]))

        asm = ObsAssembler(cfg.j, obs_mode, 2)

        steps = min(len(mj_traj), int(seconds * hz))
        win = min(steps, PB_WIN_BEATS)
        prev_lin = np.zeros(3)
        traj_pb = np.zeros((steps, cfg.j))
        win_deg = 0.0                  # 判定窗最大偏差(判据对象)
        max_deg = 0.0                  # 全 horizon 最大偏差(诊断)
        per_joint = np.zeros(cfg.j)    # 判定窗逐关节
        for t in range(steps):
            states = [p.getJointState(robot, ji) for ji in jidx]
            q = np.asarray([s[0] for s in states])
            dq = np.asarray([s[1] for s in states])
            tau = np.asarray([s[3] for s in states])   # applied joint motor torque
            pos, quat_xyzw = p.getBasePositionAndOrientation(robot)
            lin_w, ang_w = p.getBaseVelocity(robot)
            qwxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            R = _quat_to_mat_pb(qwxyz)
            acc_w = (np.asarray(lin_w) - prev_lin) / (1.0 / hz)
            prev_lin = np.asarray(lin_w).copy()
            acc_b = R.T @ (acc_w + np.asarray([0.0, 0.0, 9.81]))   # 比力(含重力反应)
            gyro_b = R.T @ np.asarray(ang_w)
            imu = np.concatenate([qwxyz, gyro_b, acc_b])
            obs = asm.assemble(q, dq, tau, imu, None,
                               goal if "goal" in obs_mode else None)
            if sess is not None:
                act = sess.run(None, {iname: obs[None].astype(np.float32)})[0][0]
            else:
                act = np.zeros(cfg.j)   # 解析探针: 中性保持
            tgt = np.clip(act, lo, hi)
            # 伺服时序与训练闭环同口径(env_mj._servo): 目标 50Hz 保持,PD 每子步重算。
            # 策略腿 PD 在环(参考);解析探针无源被动(零力矩,与 MuJoCo 腿同条件)
            for _ in range(n_sub):
                st = [p.getJointState(robot, ji) for ji in jidx]
                q_s = np.asarray([s[0] for s in st])
                dq_s = np.asarray([s[1] for s in st])
                tq_v = (cfg.kp * (tgt - q_s) - cfg.kd * dq_s) if sess is not None \
                    else np.zeros(cfg.j)
                for ji, tq in zip(jidx, tq_v):
                    p.setJointMotorControl2(robot, ji, p.TORQUE_CONTROL, force=float(tq))
                p.stepSimulation()
            # 记录相位与 MuJoCo 腿同口径: 步进后采样(两腿同拍相位)
            st = [p.getJointState(robot, ji) for ji in jidx]
            traj_pb[t] = np.asarray([s[0] for s in st])
            deg = np.abs(traj_pb[t] - mj_traj[min(t, len(mj_traj) - 1)]) * 180.0 / np.pi
            max_deg = max(max_deg, float(deg.max()))
            if t < win:
                per_joint = np.maximum(per_joint, deg)
                win_deg = max(win_deg, float(deg.max()))

        per_joint_deg = per_joint
        passed = bool(win_deg < PB_MAX_DEG)
        out = os.path.join(run, "exports", "sim2sim_pb_report.npz")
        np.savez(out, traj_pb=traj_pb, per_joint_deg=per_joint_deg, max_deg=max_deg,
                 win_deg=win_deg, passed=np.asarray(passed), hz=hz)
        print(f"[sim2sim] PyBullet 重放 {steps} 拍: 判定窗(前 {win} 拍) max|Δ| = {win_deg:.2f}° "
              f"/ 判据 <{PB_MAX_DEG}°;全 horizon = {max_deg:.2f}°(诊断)")
        top = np.argsort(per_joint_deg)[::-1][:5]
        print("[sim2sim] 窗内关节偏差 top5: " + ", ".join(
            f"{cfg.joints[i]}={per_joint_deg[i]:.2f}°" for i in top))
        print(f"[sim2sim] {'PASS' if passed else 'FAIL'}(跨引擎重力响应对拍) → {out}")
        return {"pass": passed, "max_deg": win_deg, "full_horizon_deg": max_deg,
                "per_joint_deg": per_joint_deg.tolist()}
    finally:
        p.disconnect()


def _quat_to_mat_pb(q: np.ndarray) -> np.ndarray:
    """wxyz 四元数 → 3×3 旋转矩阵(与 MuJoCo framequat 同约定)。"""
    w, x, y, z = q
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


if __name__ == "__main__":
    main()
