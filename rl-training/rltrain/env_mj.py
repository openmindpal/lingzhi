# -*- coding: utf-8 -*-
"""MuJoCo 训练环境(参考后端)——G1 29 DOF(可选 +灵巧手=41),自由底座训练变体。

契约对齐:
  - 观测 = rltrain.obs_assembler(白名单,逐位 = 契约布局)
  - 动作 = 绝对关节目标(neutral + a·scale,软限位钳制)
  - 伺服 = KP/KD 取自本体包 profile.dynamics,qfrc_applied 直写(MJCF 无执行器)
  - 传感 = framequat/gyro/accelerometer(契约 imu(10))+ 高斯噪声随机化
  - 奖励/终止 = rltrain.rewards 共享层(与 isaaclab 后端同函数同权重——跨后端可对拍);
    本文件只负责测量面提取(_measure: 仿真状态 → 测量字典)
资产: 任务含 assets 段时走参数化生成器(mjcf_gen: 自由底座/质量标定/灵巧手/物体/按压面),
      否则 = 自由底座变体(冒烟拓扑)。
说明: 本后端为单机参考实现(CPU/单卡可跑);GPU 大规模并行接 isaaclab 后端。
"""
from __future__ import annotations

import numpy as np
import mujoco

from rltrain.g1_cfg import G1Cfg
from rltrain.mjcf_gen import generate_for_task, build_train_variant, load_spec
from rltrain.obs_assembler import ObsAssembler
from rltrain.contract import SOFT_LIMIT_MARGIN_RAD
from rltrain import rewards
from rltrain.rewards import FALL_Z


def _find_sensor(model, stype) -> tuple[int, int]:
    for i in range(model.nsensor):
        if model.sensor_type[i] == stype:
            return int(model.sensor_adr[i]), int(model.sensor_dim[i])
    raise RuntimeError(f"MJCF 缺传感器类型 {stype}(契约 imu 块来源)")


class G1VecEnv:
    """n_envs 并行(独立 MjModel 副本——质量/摩擦随机化需模型级修改)。"""

    def __init__(self, cfg: G1Cfg, task: dict, n_envs: int, seed: int = 0,
                 free_base: bool = True):
        self.cfg, self.task, self.n = cfg, task, int(n_envs)
        self.j = cfg.j
        self.rs = np.random.RandomState(seed)
        # 高动态/闪避/恢复机制开关(任务级)
        self.airborne_ok = bool(task.get("airborne_ok"))       # 腾空不判摔(翻转/跳跃)
        self.start_fallen = bool(task.get("start_fallen"))     # 倒地初始位(恢复训练)
        self.dodge = task.get("dodge")                          # 威胁逼近(闪避训练)
        self.intercept = task.get("intercept")                  # 拦截: 运动学下落球+时间同步到达
        self.strike = task.get("strike")                        # 击打: 立柱球受击飞出(冲量判定)
        self.free_base = free_base
        # ── 资产路径: 任务 assets 段 → 参数化生成器;否则 = 自由底座变体 ──
        self.eff = None
        if task.get("assets"):
            from rltrain.mjcf_gen import generate_for_task
            self.xml, self.eff = generate_for_task(task)
            self.joint_names = self._joint_names_with_hands()
            self.j = len(self.joint_names)
        else:
            self.xml = build_train_variant()
            self.joint_names = list(cfg.joints)
        self.k = 2  # 接触力路数: 0=左腕(含左手) 1=右腕(含右手)——路序包文档化
        self.assembler = ObsAssembler(self.j, task["obs_mode"], self.k)
        self.margin = float(task.get("soft_limit_margin", SOFT_LIMIT_MARGIN_RAD))
        self.action_scale_vec = self._action_scale_vec()
        self.neutral = self._neutral_vec()
        self.soft_lo, self.soft_hi = self._soft_limit_vecs()
        self.control_hz = float(task["control_hz"])
        self.episode_s = float(task["episode_s"])

        self.models, self.datas = [], []
        for _ in range(self.n):
            m = mujoco.MjModel.from_xml_path(self.xml)
            self._apply_model_randomization(m)
            self.models.append(m)
            self.datas.append(mujoco.MjData(m))
        m0 = self.models[0]
        self._map_joints(m0)
        for m in self.models:
            self._check_joint_order(m)
        self._map_sensors(m0)
        self._map_ee(m0)
        self._map_route_geoms(m0)
        self._map_object(m0)
        # ── 出生高度/每关节 PD/官方力矩限幅——模型属性,两基座通用 ──
        self._gains_vec(m0)          # 生成 kp_arr/kd_arr(向量)——body_spec.gains,缺省=包标量
        self._torque_limits(m0)      # 官方 forcerange(关节 actuatorfrcrange)→ 伺服限幅
        self._compute_spawn(m0)      # 出生骨盆高 = 模型推导(脚底贴地+1cm 余量)
        self.dt_ctrl = 1.0 / self.control_hz
        self.n_sub = max(1, int(round(self.dt_ctrl / m0.opt.timestep)))
        self.max_ep_steps = int(self.episode_s * self.control_hz)
        self._ctx = rewards.build_ctx(task, self.neutral, self.control_hz)
        self._ctx["max_ep_steps"] = self.max_ep_steps
        rnd = task.get("randomize", {})
        self.imu_sigma = float(rnd.get("imu_noise", 0.0))
        self.enc_sigma = float(rnd.get("encoder_noise", 0.0))
        self.contact_sigma = float(rnd.get("contact_noise", 0.0))
        # 电机延迟随机化(§10: 0~20ms)——每回合采样,动作经环形缓冲延迟下发
        dly = rnd.get("motor_delay_ms", [0, 0])
        self.delay_range = (float(dly[0]), float(dly[1])) if isinstance(dly, (list, tuple)) else (0.0, float(dly))
        self.ep_delay = [0] * self.n
        self.act_buf = [[] for _ in range(self.n)]
        self.curriculum_frac = 1.0
        self.rng = [np.random.RandomState(seed * 1000 + i) for i in range(self.n)]
        # 回合状态
        self.ep_goal = [None] * self.n
        self.ep_push = [None] * self.n
        self.ep_morph = [None] * self.n
        self.ep_style = [0.0] * self.n
        self.ep_f = [0.5] * self.n
        self.ep_obj_z0 = [None] * self.n
        self.t_step = [0] * self.n
        self.prev_action = np.zeros((self.n, self.j))
        self.prev_err = [0.0] * self.n
        self.ep_peak = [1.0] * self.n
        self.ep_travel = [0.0] * self.n
        self.ep_rotsum = [0.0] * self.n
        self.ep_stable = [0] * self.n
        self.ep_obj_xy0 = [(0.0, 0.0)] * self.n
        self.ep_pen_z0 = [1.0] * self.n
        self.ep_gait = [(0.0, 0.0, 0.0)] * self.n
        self._gait_prev = [None] * self.n
        self._gait_v = [(0.0, 0.0, 0.0)] * self.n
        self.iball_cross_t = [None] * self.n
        self.ihand_cross_t = [None] * self.n
        self.ep_iball_t = [0.0] * self.n
        self.prev_tip = np.zeros((self.n, 3))
        self.prev_yaw = [None] * self.n
        self.reset()

    # ── 关节集扩展(灵巧手——spec 权威;文档序 = 基座 22 位插左手,右腕后插右手) ──
    def _joint_names_with_hands(self) -> list[str]:
        names = list(cfg.joints) if (cfg := self.cfg) else []
        if not self.eff or not self.eff.get("hands"):
            return names
        hand = self.eff["hand"]
        l = [f"l_hand_{jn}_joint" for jn in hand["joints"]]
        r = [f"r_hand_{jn}_joint" for jn in hand["joints"]]
        i = names.index("left_wrist_yaw_joint") + 1
        names = names[:i] + l + names[i:]
        names += r   # 右手链在文档序末尾(r_hand 为右臂末体)
        return names

    def _is_finger(self, name: str) -> bool:
        return "_hand_" in name

    def _hand_limits(self) -> dict[str, tuple[float, float]]:
        if not self.eff or not self.eff.get("hands"):
            return {}
        hand = self.eff["hand"]
        out = {}
        for side in ("l", "r"):
            for jn in hand["joints"]:
                out[f"{side}_hand_{jn}_joint"] = tuple(hand["limits_rad"][jn])
        return out

    def _action_scale_vec(self) -> np.ndarray:
        base = float(self.task.get("action_scale", 0.3))
        v = np.full(self.j, base)
        if self.eff and self.eff.get("hands"):
            fs = float(self.eff["hand"]["action_scale_rad"])
            for i, n in enumerate(self.joint_names):
                if self._is_finger(n):
                    v[i] = fs
        return v

    def _neutral_vec(self) -> np.ndarray:
        """按 joint_names 逐名取中立位(手关节缺省 0=张开)——顺序敏感,不得按段拼接。"""
        v = np.zeros(self.j)
        for i, n in enumerate(self.joint_names):
            v[i] = self.cfg.neutral.get(n, 0.0)
        return v

    def _soft_limit_vecs(self) -> tuple[np.ndarray, np.ndarray]:
        lim = dict(self.cfg.limits)
        lim.update(self._hand_limits())
        lo = [lim[n][0] for n in self.joint_names]
        hi = [lim[n][1] for n in self.joint_names]
        return np.asarray(lo) + self.margin, np.asarray(hi) - self.margin

    # ── 模型级随机化(mass/质心 ±20%,摩擦 ±30%——引擎训练硬约束 §10) ──
    def _apply_model_randomization(self, m) -> None:
        rnd = self.task.get("randomize", {})
        mj = float(rnd.get("mass_jitter", 0.0))
        fj = float(rnd.get("friction_jitter", 0.0))
        if mj > 0:
            factors = 1.0 + self.rs.uniform(-mj, mj, size=m.nbody)
            m.body_mass[:] *= factors
            m.body_inertia[:] *= factors[:, None]
        if fj > 0:
            m.geom_friction[:, 0] *= 1.0 + self.rs.uniform(-fj, fj, size=m.ngeom)

    def _map_joints(self, m) -> None:
        self.qadr, self.dadr = [], []
        for name in self.joint_names:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"MJCF 缺关节 {name}")
            self.qadr.append(int(m.jnt_qposadr[jid]))
            self.dadr.append(int(m.jnt_dofadr[jid]))
        self.qadr = np.asarray(self.qadr)
        self.dadr = np.asarray(self.dadr)
        if self.free_base:
            root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")
            if root < 0 or m.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
                raise RuntimeError("自由底座变体缺 freejoint root")
            self.root_qadr = int(m.jnt_qposadr[root])
            self.root_dadr = int(m.jnt_dofadr[root])
            self.pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def _check_joint_order(self, m) -> None:
        names = []
        for i in range(m.njnt):
            if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))
        if names != list(self.joint_names):
            raise RuntimeError("MJCF 关节序与包 joint_mapping(含手扩展)不一致")

    def _map_sensors(self, m) -> None:
        self.s_quat, self.d_quat = _find_sensor(m, mujoco.mjtSensor.mjSENS_FRAMEQUAT)
        self.s_gyro, self.d_gyro = _find_sensor(m, mujoco.mjtSensor.mjSENS_GYRO)
        self.s_acc, self.d_acc = _find_sensor(m, mujoco.mjtSensor.mjSENS_ACCELEROMETER)

    def _map_ee(self, m) -> None:
        self.site = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
                     for s, name in self.cfg.tool_sites.items()}

    def _map_route_geoms(self, m) -> None:
        """接触力路: 0=左腕(含左手链) 1=右腕(含右手链)——手开时聚合 palm+指段。

        接触路由体 = {side}_hand(两个基座都有: smoke 原生,official 由适配器注入掌框)。
        """
        self.route_geoms = {}
        for route, side in ((0, "l"), (1, "r")):
            geoms: set[int] = set()
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand")
            if bid >= 0:
                geoms |= set(int(g) for g in range(m.body_geomadr[bid],
                                                   m.body_geomadr[bid] + m.body_geomnum[bid]))
            if self.eff and self.eff.get("hands"):
                for bid2 in range(m.nbody):
                    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid2)
                    if nm and nm.startswith(f"{side}_s_"):   # 手指段体(生成器命名)
                        geoms |= set(int(g) for g in range(m.body_geomadr[bid2],
                                                           m.body_geomadr[bid2] + m.body_geomnum[bid2]))
            self.route_geoms[route] = geoms

    def _map_object(self, m) -> None:
        gname = self.task.get("grasp", {}).get("object")
        self.obj_body = None
        # 倒水: 杯(焊手)中心/口沿 + 水粒子自由体
        if self.task.get("pour"):
            self.rhand_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_hand")
            po = self.task.get("pour", {})
            self.cup_half = [float(v) for v in po.get("cup_half", [0.035, 0.035, 0.05])]
            self.cup_off = [float(v) for v in po.get("hand_off", [0.0, -0.04, 0.02])]
            self.water = []
            self.water_qvel0 = []
            for i in range(int(self.task.get("pour", {}).get("particles", 24))):
                b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_water_p{i}")
                if b >= 0:
                    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"obj_water_p{i}_root")
                    self.water.append(int(m.jnt_qposadr[jid]))
                    self.water_qvel0.append(int(m.jnt_dofadr[jid]))
        # 双手协同: 瓶体(左扶稳/右操作)
        if self.task.get("bimanual"):
            self.obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_bottle")
            self.obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_bottle_geom")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj_bottle_root")
            self.obj_qadr = int(m.jnt_qposadr[jid])
        # 拦截: mocap 球(匀速下降)+ 拦截平面
        if self.intercept:
            self.iball_mocap = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_intercept_ball")
            self.iball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_intercept_ball_geom")
            self.iball_mocap_idx = int(m.body_mocapid[self.iball_mocap])
            self.iball_z0 = float(m.body_pos[self.iball_mocap][2])
            self.plane_z = float(self.intercept.get("plane_z", 1.15))
            self.iball_speed = float(self.intercept.get("speed_ms", 0.3))
            self.iball_r = float(self.intercept.get("r_m", 0.05))
        # 击打: 立柱球(自由体)+ 冲量判定
        if self.strike:
            sb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_strike_ball")
            if sb < 0:
                raise RuntimeError("strike 任务缺 obj_strike_ball(assets.objects 未含 strike_ball)")
            self.sball_bid = sb
            self.sball_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_strike_ball_geom")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj_strike_ball_root")
            self.sball_qadr = int(m.jnt_qposadr[jid])
            self.sball_qvel0 = int(m.jnt_dofadr[jid])
            self.sball_mass = float(m.body_mass[sb])
            self.sball_hit = False
        # 手内精细: 笔(指尖旋转)
        if self.task.get("inhand"):
            self.pen_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_pen")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj_pen_root")
            self.pen_qadr = int(m.jnt_qposadr[jid])
            self.pen_qvel0 = int(m.jnt_dofadr[jid])
        # 穿针: 针尖 site + 孔板位姿/缝宽
        if self.task.get("needle"):
            self.needle_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
            pb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_needle_plate1")
            self.plate = {"x": float(m.body_pos[pb][0]), "y": float(m.body_pos[pb][1]),
                          "z": float(m.body_pos[pb][2]),
                          "gap": float(self.task.get("needle", {}).get("hole_gap_m", 0.003)),
                          "hz": float(m.geom_size[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_needle_plate1_geom")][2])}
        if self.dodge:
            self.threat_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "obj_threat_ball")
            if self.threat_body < 0:
                raise RuntimeError("dodge 任务缺 obj_threat_ball(assets.objects 未含 threat_ball)")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "obj_threat_ball_root")
            self.threat_qadr = int(m.jnt_qposadr[jid])
            self.obj_geom_t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "obj_threat_ball_geom")
            self.ground_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "ground")
            self.pelvis_geoms = set(int(g) for g in range(
                m.body_geomadr[self.pelvis_bid], m.body_geomadr[self.pelvis_bid] + m.body_geomnum[self.pelvis_bid]))
        if gname:
            self.obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{gname}")
            if self.obj_body < 0:
                raise RuntimeError(f"任务 grasp.object={gname} 但 MJCF 缺 obj_{gname}(assets.objects 未开?)")
            self.obj_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{gname}_geom")
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"obj_{gname}_root")
            self.obj_qadr = int(m.jnt_qposadr[jid])

    # ── 回合采样(课程) ──────────────────────────────────────
    def _sample_episode(self, e: int) -> None:
        t, r = self.task, self.rng[e]
        d = self.datas[e] if getattr(self, "datas", None) else None
        self.ep_goal[e] = None
        self.ep_push[e] = None
        self.ep_morph[e] = None
        self.ep_style[e] = 0.0
        if "goal" in t["obs_mode"] and not self.dodge:   # dodge: goal 块由威胁向量注入(演化提案)
            self.ep_goal[e] = rewards.sample_goal(t["goal_box"], t.get("goal_f_range", [0.0, 1.0]), r)
        push = t.get("push")
        if push:
            self.ep_push[e] = {
                "step": int(r.uniform(0, self.max_ep_steps * 0.7)),
                "dur": int(push.get("dur_s", 0.5) * self.control_hz),
                "n": float(push.get("n", 100.0)) * self.curriculum_frac * float(r.uniform(0.5, 1.0)),
                "dir": self._unit_horiz(r),
            }
        self.ep_delay[e] = int(round(self.rs.uniform(*self.delay_range) / 1000.0 * self.control_hz)) \
            if self.delay_range[1] > 0 else 0
        self.act_buf[e] = []
        morph = t.get("morph")
        if morph:
            idx = np.asarray(morph["joint_idx"], dtype=int)
            amp = np.asarray(morph["amp_rad"], dtype=np.float64)
            tgt = self.neutral.copy()
            tgt[idx] = self.neutral[idx] + amp * float(r.uniform(*morph.get("progress", [0.0, 1.0])))
            self.ep_morph[e] = tgt
        if "style" in t:
            self.ep_style[e] = float(r.uniform(0.0, 1.0))
        if "contact" in t["obs_mode"]:
            self.ep_f[e] = float(r.uniform(*t.get("goal_f_range", [0.3, 0.8])))
        if t.get("gait"):
            g = t["gait"].get("target", {"vx": 0.75, "vy": 0.0, "wz": 0.0})
            self.ep_gait[e] = (float(g["vx"]), float(g["vy"]), float(g["wz"]))
        self.ep_peak[e] = 1.0
        self._gait_prev[e] = None
        if self.task.get("bimanual"):
            self.ep_obj_z0[e] = float(d.qpos[self.obj_qadr + 2])
            self.ep_obj_xy0[e] = (float(d.qpos[self.obj_qadr]), float(d.qpos[self.obj_qadr + 1]))
            self.ep_stable[e] = 0
        if self.task.get("inhand"):
            self.ep_rotsum[e] = 0.0
            self.ep_pen_z0[e] = float(d.qpos[self.pen_qadr + 2])
        if self.obj_body is not None:
            d = self.datas[e]
            self.ep_obj_z0[e] = float(d.qpos[self.obj_qadr + 2])

    @staticmethod
    def _unit_horiz(r) -> np.ndarray:
        th = r.uniform(0, 2 * np.pi)
        return np.array([np.cos(th), np.sin(th), 0.0])

    def _gains_vec(self, m) -> None:
        """每关节 PD(body_spec.gains,官方口径)——缺省回退本体包标量 kp/kd。

        kd: null 的关节按官方 MJCF 位置伺服 dampratio=1 等价推导:
        kd = 2·√(kp·dof_M0)(dof_M0 = 编译模型 qpos0 关节等效惯量+armature)。
        """
        g = (load_spec() or {}).get("gains") or {}
        kp_s, kd_s = g.get("kp") or {}, g.get("kd") or {}
        if not kp_s:
            self.kp_arr = np.full(self.j, float(self.cfg.kp))
            self.kd_arr = np.full(self.j, float(self.cfg.kd))
            return
        kp = np.zeros(self.j)
        kd = np.zeros(self.j)
        for i, n in enumerate(self.joint_names):
            kpv = kp_s.get(n)
            kdv = kd_s.get(n)
            if kpv is None:
                # 手指等未声明关节: 腕伺服同款 kp=500 + dampratio=1 等价推导
                kpv = 500.0
            kp[i] = float(kpv)
            if kdv is None:
                dof = int(self.dadr[i])
                m0 = float(m.dof_M0[dof])
                kd[i] = 2.0 * np.sqrt(kp[i] * m0)
            else:
                kd[i] = float(kdv)
        self.kp_arr, self.kd_arr = kp, kd

    def _torque_limits(self, m) -> None:
        """官方力矩限(适配器挂载的关节 actuatorfrcrange)→ 伺服输出钳制。

        (0,0) = 未限(冒烟基座)——不钳制。官方模型 hip 88/139,knee 139,
        ankle/waist 50,肩肘 25,腕 5/25——真机伺服饱和的仿真对应。
        """
        lo, hi = [], []
        limited = False
        for i, n in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
            a, b = float(m.jnt_actfrcrange[jid][0]), float(m.jnt_actfrcrange[jid][1])
            if a != 0.0 or b != 0.0:
                limited = True
            lo.append(a if limited else -np.inf)
            hi.append(b if limited else np.inf)
        self.trq_lo = np.asarray(lo) if limited else None
        self.trq_hi = np.asarray(hi) if limited else None

    def _compute_spawn(self, m) -> None:
        """出生骨盆高度 = 模型推导——z = qpos0 骨盆高 − qpos0 最低碰撞点高 + 1cm 余量。

        最低碰撞点 = mj_geomDistance(地面平面 × 各碰撞 geom)——mesh 精确距离。
        出生高度必须是模型属性而非全局常量: 各基座站立高度不同(官方 G1 0.793m,
        冒烟变体 ~0.37m),常量对任一基座都可能造成出生悬空或越限。
        站立高 < FALL_Z 的资产无法支撑自由底座站立训练——构建期硬拒。
        """
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        mujoco.mj_forward(m, d)
        plane = next((g for g in range(m.ngeom)
                      if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE), None)
        if plane is not None:
            min_d = 1e9
            for g in range(m.ngeom):
                if g == plane or m.geom_contype[g] == 0:
                    continue   # 非碰撞 geom(视觉壳)不参与
                min_d = min(min_d, float(mujoco.mj_geomDistance(m, d, g, plane, 2.0, None)))
            min_z = min_d
        else:
            min_z = 0.0
        base_z = float(d.qpos[self.root_qadr + 2])
        self.spawn_z = base_z - min_z + 0.01
        if self.spawn_z < FALL_Z:
            raise RuntimeError(f"模型站立骨盆高 {self.spawn_z:.3f}m < FALL_Z {FALL_Z}m"
                               "——该资产无法支撑自由底座站立训练(基座换 official)")

    def reset(self, envs=None) -> np.ndarray:
        envs = list(range(self.n)) if envs is None else list(envs)
        obs = [None] * self.n
        for e in envs:
            m, d = self.models[e], self.datas[e]
            mujoco.mj_resetData(m, d)
            if self.free_base:
                d.qpos[self.root_qadr:self.root_qadr + 2] = 0.0
                d.qpos[self.root_qadr + 2] = self.spawn_z
                d.qpos[self.root_qadr + 3:self.root_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                if self.start_fallen:   # 倒地初始位(恢复训练): 侧躺
                    d.qpos[self.root_qadr + 2] = 0.35
                    d.qpos[self.root_qadr + 3:self.root_qadr + 7] = [0.7071, 0.7071, 0.0, 0.0]
            noise = self.rng[e].uniform(-0.03, 0.03, size=self.j)
            d.qpos[self.qadr] = self.neutral + noise
            d.qvel[:] = 0.0
            self._sample_episode(e)
            self.t_step[e] = 0
            self.prev_action[e][:] = 0.0
            self.ep_travel[e] = 0.0
            self.prev_tip[e][:] = 0.0
            self.prev_yaw[e] = None
            mujoco.mj_forward(m, d)
            if self.task.get("inhand") and getattr(self, "pen_bid", None) is not None:
                rh = d.xpos[self.rhand_bid] if hasattr(self, "rhand_bid") else None
                if rh is None:
                    rb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "r_hand")
                    rh = d.xpos[rb]
                d.qpos[self.pen_qadr:self.pen_qadr + 3] = [float(rh[0]) + 0.06, float(rh[1]), float(rh[2]) - 0.02]
                d.qpos[self.pen_qadr + 3:self.pen_qadr + 7] = [1.0, 0.0, 0.0, 0.0]
                d.qvel[self.pen_qvel0:self.pen_qvel0 + 6] = 0.0
                mujoco.mj_forward(m, d)
            self.iball_cross_t = None
            self.ihand_cross_t = None
            self.ep_iball_t[e] = 0.0
            if self.intercept:
                d.mocap_pos[self.iball_mocap_idx] = [float(m.body_pos[self.iball_mocap][0]),
                                                     float(m.body_pos[self.iball_mocap][1]),
                                                     self.iball_z0]
                self.iball_cross_t = None
                self.ep_iball_t[e] = 0.0
            if self.strike:
                d.qpos[self.sball_qadr:self.sball_qadr + 3] = [float(m.body_pos[self.sball_bid][0]),
                                                               float(m.body_pos[self.sball_bid][1]),
                                                               float(m.body_pos[self.sball_bid][2])]
                d.qvel[self.sball_qvel0:self.sball_qvel0 + 3] = 0.0
                self.sball_hit = False
            if self.task.get("pour") and getattr(self, "water", None):
                # 粒子按"沉降后实际杯位"重排——XML 静态摆放会被重力沉降错位
                rh = d.xpos[self.rhand_bid]
                cc = [float(rh[i]) + self.cup_off[i] for i in range(3)]
                rr = self.rng[e]
                for i, q in enumerate(self.water):
                    d.qpos[q] = cc[0] + rr.uniform(-0.02, 0.02)
                    d.qpos[q + 1] = cc[1] + rr.uniform(-0.02, 0.02)
                    d.qpos[q + 2] = cc[2] - 0.03 + 0.012 * (i // 6) + 0.004
                    d.qvel[self.water_qvel0[i]:self.water_qvel0[i] + 3] = 0.0
                mujoco.mj_forward(m, d)
            obs[e] = self._observe(e)
        return np.stack([obs[e] for e in envs])

    # ── 主循环 ─────────────────────────────────────────────
    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape != (self.n, self.j):
            raise ValueError(f"动作形状 {actions.shape} ≠ ({self.n},{self.j})")
        obs, rews, dones, infos = [], [], [], []
        for e in range(self.n):
            o, r, done, info = self._step_one(e, actions[e])
            obs.append(o); rews.append(r); dones.append(done); infos.append(info)
        return np.stack(obs), np.asarray(rews, dtype=np.float32), np.asarray(dones), infos

    def set_curriculum(self, frac: float) -> None:
        """扰动课程(0~1)——按训练进度线性放大推力。"""
        self.curriculum_frac = float(np.clip(frac, 0.0, 1.0))

    def _step_one(self, e: int, a: np.ndarray):
        m, d = self.models[e], self.datas[e]
        # 电机延迟: 下发目标 = delay 步前的动作(§10 随机化——真机执行器延迟的仿真对应)
        buf = self.act_buf[e]
        buf.append(a.copy())
        delay = self.ep_delay[e]
        a_eff = buf[-(delay + 1)] if delay > 0 and len(buf) > delay else a
        target = np.clip(self.neutral + a_eff * self.action_scale_vec, self.soft_lo, self.soft_hi)
        if self.task.get("inhand"):
            # 手内旋转累计(绕竖直轴的姿态角增量——转笔/转棒代理)
            qw, qx, qy, qz = (float(v) for v in d.qpos[self.pen_qadr + 3:self.pen_qadr + 7])
            yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
            if self.prev_yaw[e] is not None:
                dy = yaw - self.prev_yaw[e]
                if dy > np.pi:
                    dy -= 2 * np.pi
                if dy < -np.pi:
                    dy += 2 * np.pi
                self.ep_rotsum[e] += dy
            self.prev_yaw[e] = yaw
        if self.task.get("wipe"):
            tip = d.site_xpos[self.site["r"]]
            disp = float(np.linalg.norm(np.asarray(tip[:2]) - self.prev_tip[e][:2]))
            self.ep_travel[e] += disp if abs(float(tip[2]) - self.prev_tip[e][2]) < 0.02 else 0.0
            self.prev_tip[e] = np.asarray(tip).copy()
        self._servo(e, target, steps=self.n_sub)
        if self.intercept:
            z = float(d.mocap_pos[self.iball_mocap_idx][2])
            newz = z - self.iball_speed * (self.dt_ctrl * self.n_sub)
            d.mocap_pos[self.iball_mocap_idx][2] = max(newz, self.plane_z - 0.05)
            tip = d.site_xpos[self.site["r"]]
            if self.iball_cross_t is None and newz <= self.plane_z:
                self.iball_cross_t = self.ep_iball_t[e]
            if self.ihand_cross_t is None and abs(float(tip[2]) - self.plane_z) < 0.06                and float(np.linalg.norm(np.asarray(tip[:2]) - np.asarray(d.mocap_pos[self.iball_mocap_idx][:2]))) < self.iball_r + 0.03:
                self.ihand_cross_t = self.ep_iball_t[e]
            self.ep_iball_t[e] += self.dt_ctrl * self.n_sub
        self.t_step[e] += 1
        if self.task.get("bimanual"):
            z0 = self.ep_obj_z0[e]
            x0, y0 = self.ep_obj_xy0[e]
            tilt_ok = abs(float(d.qpos[self.obj_qadr + 3])) > 0.95
            xy_ok = float(np.linalg.norm(d.qpos[self.obj_qadr:self.obj_qadr + 2] - np.array([x0, y0]))) < 0.04
            self.ep_stable[e] += int(tilt_ok and xy_ok)
        obs = self._observe(e)
        info = self._info(e)
        m = self._measure(e, a, info)
        rew, terms = rewards.compute_reward(self._ctx, m)
        info["reward_terms"] = terms
        if m.get("threat_hit"):
            info["threat_hit"] = True
        done, _flags = rewards.check_termination(self._ctx, m)
        if done:
            o2 = self.reset([e])[0]
            info["final_obs"] = o2
        return obs, float(rew), bool(done), info

    def _servo(self, e: int, target: np.ndarray, steps: int) -> None:
        m, d = self.models[e], self.datas[e]
        push = self.ep_push[e]
        for _ in range(steps):
            q = d.qpos[self.qadr]
            dq = d.qvel[self.dadr]
            u = self.kp_arr * (target - q) - self.kd_arr * dq
            if self.trq_hi is not None:
                u = np.clip(u, self.trq_lo, self.trq_hi)   # 官方 forcerange——真机伺服饱和对应
            d.qfrc_applied[self.dadr] = u
            if self.free_base:
                d.xfrc_applied[self.pelvis_bid] = 0.0
                if self.dodge and self.threat_body is not None:
                    va = self.dodge.get("speed_ms", 0.8)
                    dx = d.qpos[self.root_qadr:self.root_qadr + 2] - d.qpos[self.threat_qadr:self.threat_qadr + 2]
                    n = np.linalg.norm(dx) + 1e-6
                    d.qvel[self.threat_qadr:self.threat_qadr + 2] = dx / n * va
                if push and push["step"] <= self.t_step[e] < push["step"] + push["dur"]:
                    d.xfrc_applied[self.pelvis_bid] = np.concatenate(
                        [push["dir"] * push["n"], [0.0, 0.0, 0.0]])
            mujoco.mj_step(m, d)

    # ── 观测(契约装配) ─────────────────────────────────────
    def _observe(self, e: int) -> np.ndarray:
        m, d = self.models[e], self.datas[e]
        r = self.rng[e]
        q = d.qpos[self.qadr] + r.normal(0, self.enc_sigma, self.j)
        dq = d.qvel[self.dadr] + r.normal(0, self.enc_sigma, self.j)
        tau = d.qfrc_applied[self.dadr]
        quat = d.sensordata[self.s_quat:self.s_quat + 4] + r.normal(0, self.imu_sigma, 4)
        gyro = d.sensordata[self.s_gyro:self.s_gyro + 3] + r.normal(0, self.imu_sigma, 3)
        acc = d.sensordata[self.s_acc:self.s_acc + 3] + r.normal(0, self.imu_sigma, 3)
        imu = np.concatenate([quat, gyro, acc])
        contact = self._route_wrench(e) if "contact" in self.task["obs_mode"] else None
        goal = self._goal_obs(e) if "goal" in self.task["obs_mode"] else None
        return self.assembler.assemble(q, dq, tau, imu, contact, goal)

    def _route_wrench(self, e: int) -> np.ndarray:
        m, d = self.models[e], self.datas[e]
        out = np.zeros(self.k * 6)
        frc = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for route, geoms in self.route_geoms.items():
                if c.geom1 in geoms or c.geom2 in geoms:
                    mujoco.mj_contactForce(m, d, i, frc)
                    rot = np.asarray(c.frame[:9]).reshape(3, 3)
                    fw = rot.T @ frc[:3]
                    tw = rot.T @ frc[3:]
                    base = route * 6
                    out[base:base + 3] += fw
                    out[base + 3:base + 6] += tw
        return out + self.rng[e].normal(0, self.contact_sigma, out.shape)

    def _goal_obs(self, e: int) -> np.ndarray:
        g = self.ep_goal[e]
        if g is not None:
            return np.asarray(g, dtype=np.float64)
        if self.dodge:   # 契约演化提案: goal 块承载威胁向量 [闪避方向x, y, 距离, 紧急度]
            d = self.datas[e]
            dx = d.qpos[self.root_qadr:self.root_qadr + 2] - d.qpos[self.threat_qadr:self.threat_qadr + 2]
            n = float(np.linalg.norm(dx)) + 1e-6
            urg = float(np.clip(1.0 - n / 2.0, 0.0, 1.0))
            return np.array([dx[0] / n, dx[1] / n, n, urg])
        return np.zeros(4)

    # ── 信息/奖励 ──────────────────────────────────────────
    def _info(self, e: int) -> dict:
        m, d = self.models[e], self.datas[e]
        info = {"t": self.t_step[e]}
        if "goal" in self.task["obs_mode"] and self.ep_goal[e] is not None:
            world = self._goal_world(e)
            ee = d.site_xpos[self.site["r"]]
            info["ee_err"] = float(np.linalg.norm(ee - world))
        if self.free_base:
            info["root_z"] = float(d.qpos[self.root_qadr + 2])
        if self.obj_body is not None:
            z = float(d.qpos[self.obj_qadr + 2])
            info["obj_lift"] = z - self.ep_obj_z0[e]
            info["obj_grasp_n"] = self._obj_contact_n(e)
        if getattr(self, "water", None) and self.task.get("pour"):
            info["water_escaped"] = self._water_escaped(e)
        if getattr(self, "needle_site", None) is not None:
            info["insert_depth"] = self._insert_depth(e)
        if self.task.get("wipe"):
            info["travel"] = float(self.ep_travel[e])
        if self.intercept:
            ball = d.mocap_pos[self.iball_mocap_idx]
            tip = d.site_xpos[self.site["r"]]
            info["iball_dist"] = float(np.linalg.norm(np.asarray(tip) - np.asarray(ball)))
            info["ball_crossed"] = self.iball_cross_t is not None
            info["hand_crossed"] = self.ihand_cross_t is not None
        if self.strike:
            v = d.qvel[self.sball_qvel0:self.sball_qvel0 + 3]
            info["ball_speed"] = float(np.linalg.norm(v))
            info["ball_z"] = float(d.qpos[self.sball_qadr + 2])
        if self.task.get("bimanual"):
            f_ref = float(self.ep_goal[e][3]) * float(self.task.get("bimanual", {}).get("f_force_n", 20.0))                 if self.ep_goal[e] is not None else 0.0
            info["f_left_err"] = float(np.linalg.norm(self._route_wrench(e)[:3]) - f_ref)
            info["bottle_stable_frac"] = self.ep_stable[e] / max(1, self.t_step[e])
        if self.task.get("inhand"):
            info["pen_rot"] = float(self.ep_rotsum[e])
            info["pen_held"] = float(self._pen_held(e))
        return info

    def _goal_world(self, e: int) -> np.ndarray:
        """goal 为基座系——世界目标 = 基座位姿 ⊕ goal(训练与派发同语义)。"""
        d = self.datas[e]
        p = d.qpos[self.root_qadr:self.root_qadr + 3]
        quat = d.qpos[self.root_qadr + 3:self.root_qadr + 7]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, np.asarray(quat))
        return p + R.reshape(3, 3) @ np.asarray(self.ep_goal[e][:3])

    def _root_track(self, e: int) -> None:
        """根位置/偏航角有限差分追踪——帧约定无关,差分出速度与角速度。"""
        d = self.datas[e]
        x = float(d.qpos[self.root_qadr])
        y = float(d.qpos[self.root_qadr + 1])
        qw = float(d.qpos[self.root_qadr + 3])
        qx = float(d.qpos[self.root_qadr + 4])
        qy = float(d.qpos[self.root_qadr + 5])
        qz = float(d.qpos[self.root_qadr + 6])
        yaw = float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        prev = self._gait_prev[e]
        dt = self.dt_ctrl * max(1, self.n_sub)
        if prev is not None:
            px, py, pyaw = prev
            self._gait_v[e] = ((x - px) / dt, (y - py) / dt, self._wrap(yaw - pyaw) / dt)
        self._gait_prev[e] = (x, y, yaw)

    @staticmethod
    def _wrap(a: float) -> float:
        while a > np.pi:
            a -= 2 * np.pi
        while a < -np.pi:
            a += 2 * np.pi
        return a

    def _gait_vel(self, e: int):
        return self._gait_v[e]

    def _ball_contact_n(self, e: int) -> int:
        """击打球与手部路线几何的接触计数。"""
        d = self.datas[e]
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == self.sball_geom or c.geom2 == self.sball_geom) and                (c.geom1 in self.route_geoms[0] or c.geom1 in self.route_geoms[1] or
                c.geom2 in self.route_geoms[0] or c.geom2 in self.route_geoms[1]):
                n += 1
        return n

    def _obj_contact_n(self, e: int) -> int:
        d = self.datas[e]
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 == self.obj_geom or c.geom2 == self.obj_geom) and \
               (c.geom1 in self.route_geoms[0] or c.geom1 in self.route_geoms[1] or
                c.geom2 in self.route_geoms[0] or c.geom2 in self.route_geoms[1]):
                n += 1
        return n

    def _water_escaped(self, e: int) -> int:
        d = self.datas[e]
        cx, cy, cz = (float(d.xpos[self.rhand_bid][i]) + self.cup_off[i] for i in range(3))
        top = cz + self.cup_half[2]
        hx, hy = self.cup_half[0], self.cup_half[1]
        n = 0
        for q in self.water:
            x, y, z = float(d.qpos[q]), float(d.qpos[q + 1]), float(d.qpos[q + 2])
            if z > top + 0.02 or abs(x - cx) > hx or abs(y - cy) > hy:
                n += 1
        return n

    def _pen_held(self, e: int) -> bool:
        """笔被持住 = 指尖/手有接触 且 未坠落。"""
        d = self.datas[e]
        if float(d.qpos[self.pen_qadr + 2]) < float(self.task.get("inhand", {}).get("drop_z", 0.9)):
            return False
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 in self.route_geoms[0] or c.geom1 in self.route_geoms[1] or                c.geom2 in self.route_geoms[0] or c.geom2 in self.route_geoms[1]:
                return True
        return False

    def _insert_depth(self, e: int) -> float:
        """穿针深度: 针尖越过孔板前沿且在缝内(x/z 对齐)——0 为未插入。"""
        d = self.datas[e]
        tip = d.site_xpos[self.needle_site]
        pl = self.plate
        if abs(float(tip[0]) - pl["x"]) > pl["gap"] / 2:
            return 0.0
        if abs(float(tip[2]) - pl["z"]) > pl["hz"]:
            return 0.0
        depth = float(tip[1]) - (pl["y"] - 0.02)
        return max(0.0, depth)

    # ── 测量面(rewards.compute_reward 的输入——仿真状态 → 后端无关测量字典) ──
    def _measure(self, e: int, a: np.ndarray, info: dict) -> dict:
        m, d = self.models[e], self.datas[e]
        rate = float(np.mean(np.abs(a - self.prev_action[e])))
        self.prev_action[e] = a.copy()
        tau_norm = float(np.mean(np.abs(d.qfrc_applied[self.dadr]))) / float(np.max(self.kp_arr))
        mm: dict = {"action_rate": rate, "tau_norm": tau_norm, "t": self.t_step[e],
                    "q": np.asarray(d.qpos[self.qadr], dtype=np.float64)}
        if self.free_base:
            mm["upright"] = float(d.qpos[self.root_qadr + 3])
            mm["root_z"] = float(d.qpos[self.root_qadr + 2])
            mm["quat_w"] = mm["upright"]
        else:
            mm["upright"] = 1.0
        if self._ctx["gait"]:
            self._root_track(e)
            vxa, vya, wza = self._gait_vel(e)
            vx_t, vy_t, wz_t = self.ep_gait[e]
            mm["vel_err"] = float(np.linalg.norm([vxa - vx_t, vya - vy_t]) + abs(wza - wz_t))
        if self._ctx["airborne_ok"]:
            self.ep_peak[e] = max(self.ep_peak[e], float(d.qpos[self.root_qadr + 2]))
            mm["peak_z"] = self.ep_peak[e]
            mm["upright_end"] = abs(float(d.qpos[self.root_qadr + 3]))
        if "goal" in self.task["obs_mode"] and self.ep_goal[e] is not None:
            err = info["ee_err"]
            mm["ee_err"] = err
            mm["prev_err"] = self.prev_err[e]
            self.prev_err[e] = err   # 本步误差成为下步 prev(进度项语义)
        if self._ctx["grasp"]:
            mm["obj_lift"] = info.get("obj_lift", 0.0)
            mm["grasp_n"] = info.get("obj_grasp_n", 0)
        if self._ctx["contact_obs"] and self._ctx["family"] == "manipulation":
            mm["fmag"] = float(np.linalg.norm(self._route_wrench(e)[:3]))
            mm["ep_f"] = self.ep_f[e]
        if self._ctx["pour"]:
            mm["water_escaped"] = info.get("water_escaped", 0)
        if self._ctx["intercept"]:
            ball = d.mocap_pos[self.iball_mocap_idx]
            tip = d.site_xpos[self.site["r"]]
            mm["iball_dist"] = float(np.linalg.norm(np.asarray(tip) - np.asarray(ball)))
            if self.iball_cross_t is not None and self.ihand_cross_t is not None:
                mm["timing_err"] = abs(self.ihand_cross_t - self.iball_cross_t)
        if self._ctx["bimanual"]:
            mm["f_left_err"] = info.get("f_left_err", 0.0)
            mm["stable_frac"] = info.get("bottle_stable_frac", 0.0)
        if self._ctx["strike"]:
            v = d.qvel[self.sball_qvel0:self.sball_qvel0 + 3]
            mm["ball_speed"] = float(np.linalg.norm(v))
            mm["ball_contacts"] = self._ball_contact_n(e)
        if self._ctx["inhand"]:
            mm["pen_rot"] = info.get("pen_rot", 0.0)
            mm["pen_held"] = float(info.get("pen_held", 0.0))
            mm["rot_target"] = float(self.ep_goal[e][0]) if "goal" in self.task["obs_mode"] else 6.0
            mm["pen_z"] = float(d.qpos[self.pen_qadr + 2])
        if self._ctx["needle"]:
            mm["insert_depth"] = info.get("insert_depth", 0.0)
        if self._ctx["wipe"]:
            mm["travel"] = info.get("travel", 0.0)
        if self._ctx["family"] == "deform":
            mm["morph_target"] = self.ep_morph[e]
        if self._ctx["family"] == "expression":
            mm["style"] = self.ep_style[e]
        if self._ctx["dodge"]:
            mm["threat_hit"] = self._threat_hit(e)
        return mm

    def _threat_hit(self, e: int) -> bool:
        """命中 = 威胁球触到机器人任意部位(地面除外——球落地不算)。"""
        d = self.datas[e]
        for i in range(d.ncon):
            c = d.contact[i]
            if c.geom1 == self.obj_geom_t and c.geom2 != self.ground_gid:
                return True
            if c.geom2 == self.obj_geom_t and c.geom1 != self.ground_gid:
                return True
        return False

    # ── 快脑专家数据(解析追靶——供 BC 蒸馏) ───────────────────
    def expert_goal(self, e: int) -> np.ndarray | None:
        g = self.ep_goal[e]
        if g is None:
            return None
        d = self.datas[e]
        ee = d.site_xpos[self.site["r"]] - d.qpos[self.root_qadr:self.root_qadr + 3]
        delta = np.asarray(g[:3]) - ee
        bound = float(self.task.get("expert_rate", 0.02))
        step = np.clip(delta, -bound, bound)
        nxt = np.asarray(g[:3]) + step
        box = np.asarray(self.task["goal_box"], dtype=np.float64)
        nxt = np.array([np.clip(nxt[i], box[i][0], box[i][1]) for i in range(3)])
        return np.concatenate([nxt, [g[3]]])
