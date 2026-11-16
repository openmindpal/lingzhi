# -*- coding: utf-8 -*-
"""Isaac Lab GPU 训练后端——契约不变的 GPU 大规模并行(direct 风格)。

设计口径:
  - direct 风格: SimulationContext + InteractiveScene + Articulation 直驱——
    不经 ManagerBasedEnv 管理制;动作/观测/奖励全在本适配器,与 MuJoCo 后端逐项同语义
  - 资产 = 本体包导出的 URDF(envs/g1/assets/g1_pb.urdf,urdf_gen 产物)——关节名/序/
    限位与包同源,不依赖 isaaclab_assets 命名(Sim2Sim 资产同源哲学)
  - 观测 = 契约布局[q|dq|tau|imu(10)|contact(6k)|goal(4)] torch 批式装配(白名单同序)
  - 奖励/终止 = rltrain.rewards 共享层(MuJoCo 同函数同权重,批数组求值)
  - imu(10) = quat(wxyz)+角速度(体系)+比力(体系)——与 MuJoCo framequat/gyro/
    accelerometer 逐项同源;contact 6k = 左右腕 ContactSensor 世界系力
  - 随机化 = 观测高斯噪声 + 电机延迟(环形缓冲) + 推力扰动(课程) + 初始位形噪声/
    倒地初始位 + 质量/摩擦抖动(接钩: C5 physx 批写,v1 未启用——见诚实边界)
  - 部分重置安全: 回合状态(目标/延迟/推力/波形位形/延迟缓冲/gait 差分基线)按
    env 子集行级重置——auto-reset 不污染其余环境的回合状态

机制覆盖矩阵(v1,GPU 生产集):
  ✓ balance(含 gait 速度跟踪) / reach / reach_contact / deform / expression /
    recover(start_fallen) / athletic(airborne)
  ✗ 外物/柔性/粒子机制(dodge/intercept/strike/pour/needle/bimanual/inhand/grasp/
    wipe)——需 mjcf_gen 资产的 USD 导出适配点,构造期诚实拒收(不降级模拟)

首次 GPU 校准点(版本漂移集中登记——逐项核对 API 名后删除对应标记):
  C1 AppLauncher 参数字典键 / UrdfFileCfg 字段
  C2 InteractiveScene 动态资产挂载(scene_cfg.<name> = cfg)
  C3 ImplicitActuatorCfg 字段名(stiffness/damping/effort_limit)
  C4 ContactSensorCfg prim 通配与 data.net_forces_w 形状
  C5 root_physx_view 批写质量/材质的 API 名与调用时机(mass_jitter/friction_jitter 接钩)
  C6 Articulation.write_external_wrench_to_sim(推力扰动)
  C7 root 线加速度通道(root_link_lin_acc_b 存在性——缺省走有限差分)
  C8 reset 写态→forward→缓冲刷新的调用序
诚实边界:
  - GPU/Isaac 环境不可用 = 构造期 SystemExit 诚实报错(不降级模拟)
  - contact 6D 力矩通道 = 零杠杆保守近似(ContactSensor 无接触点输出;MuJoCo 后端
    为逐接触真值)——Sim2Sim 对拍不含 contact 块,真值随引擎侧 wrench API 校准
  - 末端点 = r_hand 链接原点(MuJoCo 后端 = r_tool site)——常数偏差,目标条件化
    训练自洽;Sim2Sim 对拍不含 ee 绝对值
  - 质量/摩擦抖动已接钩(_apply_domain_randomization:reset 后 physx 批写,
    fail-loud)——API 名随 C5 校准核对
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from rltrain.obs_assembler import ObsAssembler
from rltrain.contract import SOFT_LIMIT_MARGIN_RAD, GOAL_DIM
from rltrain import rewards

_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "envs", "g1", "assets", "g1_pb.urdf")
_SIM_DT = 1.0 / 500.0
_G_UP = np.array([0.0, 0.0, 9.81], dtype=np.float32)   # 比力 = a − g(g 世界系 z 向下)


def _require_isaaclab() -> None:
    try:
        import isaaclab  # noqa: F401
        import isaacsim  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "[isaaclab 后端] 缺运行时依赖(GPU 服务器上安装):\n"
            "  pip install isaaclab[all]  (含 isaacsim)\n"
            f"  原始错误: {e}")


def _urdf_link_names(urdf: str) -> set[str]:
    root = ET.parse(urdf).getroot()
    return {l.get("name") for l in root.findall("link")}


class IsaacLabVecEnv:
    """与 G1VecEnv 同接口(reset/step/manifest 字段)——train.py 直接复用。"""

    def __init__(self, cfg, task: dict, n_envs: int, seed: int = 0,
                 device: str = "cuda:0", headless: bool = True, urdf_path: str | None = None):
        _require_isaaclab()
        if "cuda" not in device:
            raise SystemExit("[isaaclab 后端] 需要 GPU 设备(--device cuda:0)——CPU 请用 mujoco 后端")
        # 机制覆盖门(v1 = 无外物机制全集;外物/柔性/粒子 = USD 导出适配点,诚实拒收)
        unsupported = [k for k in ("assets", "pour", "needle", "wipe", "bimanual",
                                   "inhand", "grasp", "dodge", "intercept", "strike")
                       if task.get(k)]
        if unsupported:
            raise SystemExit(
                f"[isaaclab 后端] 任务含本后端 v1 未接机制 {unsupported}"
                "(外物/柔性/粒子需 mjcf_gen 资产的 USD 导出适配点)——"
                "GPU 覆盖集 = balance/gait/reach/reach_contact/deform/expression/"
                "recover/athletic(见文件头机制覆盖矩阵)")

        import torch
        from isaaclab.app import AppLauncher
        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sim import SimulationContext

        self._torch = torch
        self.task, self.n = task, int(n_envs)
        self.device = device
        self.k = 2   # 接触力路数: 0=左腕 1=右腕(与 MuJoCo 后端路序一致)
        self.j = cfg.j
        self._kp = float(cfg.kp)
        self.assembler = ObsAssembler(self.j, task["obs_mode"], self.k)
        self.margin = float(task.get("soft_limit_margin", SOFT_LIMIT_MARGIN_RAD))
        self.control_hz = float(task["control_hz"])
        self.episode_s = float(task["episode_s"])
        self.max_ep_steps = int(self.episode_s * self.control_hz)
        self.airborne_ok = bool(task.get("airborne_ok"))
        self.start_fallen = bool(task.get("start_fallen"))
        self.rs = np.random.RandomState(seed)
        lo, hi = cfg.limit_arrays()
        self.soft_lo = np.asarray(lo) + self.margin
        self.soft_hi = np.asarray(hi) - self.margin
        self.neutral = np.asarray(cfg.neutral_array(), dtype=np.float32)
        self.action_scale_vec = np.full(self.j, float(task.get("action_scale", 0.3)), dtype=np.float32)
        self.joint_names = list(cfg.joints)
        self._ctx = rewards.build_ctx(task, self.neutral, self.control_hz)
        self._ctx["max_ep_steps"] = self.max_ep_steps
        rnd = task.get("randomize", {})
        self.imu_sigma = float(rnd.get("imu_noise", 0.0))
        self.enc_sigma = float(rnd.get("encoder_noise", 0.0))
        self.contact_sigma = float(rnd.get("contact_noise", 0.0))
        self._mass_jitter = float(rnd.get("mass_jitter", 0.0))
        self._friction_jitter = float(rnd.get("friction_jitter", 0.0))
        self._base_masses = None    # C5: 首次随机化时捕获基准质量(物理引擎视图)
        self._base_mats = None      # C5: 基准材质属性(静/动摩擦/恢复系数)
        dly = rnd.get("motor_delay_ms", [0, 0])
        self.delay_range = (float(dly[0]), float(dly[1])) if isinstance(dly, (list, tuple)) else (0.0, float(dly))
        self._max_delay = int(np.ceil(self.delay_range[1] / 1000.0 * self.control_hz))
        self.curriculum_frac = 1.0
        torch.manual_seed(seed)

        # ── App 启动(必须先于一切 isaaclab.sim 深导入) ──
        app_launcher = AppLauncher({"headless": headless, "device": device})      # C1
        self._app = app_launcher.app

        # ── 资产: 本体包同源 URDF(urdf_gen 产物;关节名 = 包序) ──
        urdf = os.path.abspath(urdf_path or _URDF)
        if not os.path.exists(urdf):
            raise SystemExit(
                f"[isaaclab 后端] 缺 URDF 资产 {urdf}\n"
                "  生成: python -c \"from rltrain.urdf_gen import generate_urdf; generate_urdf()\"\n"
                "  (从训练变体 MJCF 导出——关节序/限位/质量惯量与包同源)")
        links = _urdf_link_names(urdf)
        want_contact = "contact" in task["obs_mode"]
        if want_contact:
            missing = [s for s in ("l_hand", "r_hand") if s not in links]
            if missing:
                raise SystemExit(f"[isaaclab 后端] contact 观测模式但 URDF 缺腕体 {missing}"
                                 "(URDF 与包不同源——重新生成)")
        robot_cfg = ArticulationCfg(
            prim_path="/World/Robot.*",
            spawn=sim_utils.UrdfFileCfg(asset_path=urdf, fix_base=False),          # C1
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={n: float(v) for n, v in zip(self.joint_names, self.neutral)}),
            actuators={"body": ImplicitActuatorCfg(                                 # C3
                joint_names_expr=[".*"], stiffness=self._kp, damping=float(cfg.kd),
                effort_limit=300.0, velocity_limit=float(cfg.max_velocity))},       # effort 上限 = URDF 同源(urdf_gen effort=300)
        )

        # ── 场景(资产/地面/传感器全部进 cfg 后一次构造——C2) ──
        decimation = max(1, int(round((1.0 / self.control_hz) / _SIM_DT)))
        self._decimation = decimation
        scene_cfg = InteractiveSceneCfg(num_envs=self.n, env_spacing=2.5)
        scene_cfg.robot = robot_cfg
        scene_cfg.ground = AssetBaseCfg(func=sim_utils.GroundPlaneCfg())
        if want_contact:
            from isaaclab.sensors import ContactSensorCfg
            scene_cfg.contact_l = ContactSensorCfg(prim_path="/World/Robot.*/l_hand")   # C4
            scene_cfg.contact_r = ContactSensorCfg(prim_path="/World/Robot.*/r_hand")
        self._scene = InteractiveScene(scene_cfg)
        self._robot: Articulation = self._scene["robot"]
        self._sim = SimulationContext.instance()
        assert self._sim is not None
        self._contact_l = self._scene["contact_l"] if want_contact else None
        self._contact_r = self._scene["contact_r"] if want_contact else None

        # ── 关节序映射(包序 → 资产序;名字找不到 = 诚实报错) ──
        asset_names = list(self._robot.joint_names)
        idx = []
        for name in self.joint_names:
            if name not in asset_names:
                raise RuntimeError(f"URDF/资产缺关节 {name}(资产与包不同源——重新生成 URDF)")
            idx.append(asset_names.index(name))
        self._idx = torch.tensor(idx, dtype=torch.long, device=device)
        self._eye = torch.arange(self.n, device=device)
        self._soft_lo_t = torch.tensor(self.soft_lo, dtype=torch.float32, device=device)
        self._soft_hi_t = torch.tensor(self.soft_hi, dtype=torch.float32, device=device)
        self._neutral_t = torch.tensor(self.neutral, dtype=torch.float32, device=device)
        self._scale_t = torch.tensor(self.action_scale_vec, dtype=torch.float32, device=device)
        self._ee_bid = self._robot.find_bodies("r_hand")[0][0]

        # ── 回合状态(GPU 批;reset 行级更新) ──
        z = torch.zeros(self.n, device=device)
        self._ep_goal = torch.zeros(self.n, GOAL_DIM, device=device)
        self._ep_f = torch.full((self.n,), 0.5, device=device)   # contact 模式力参考(与 MuJoCo ep_f 同源采样)
        self._ep_style = z.clone()
        self._ep_morph = self._neutral_t.unsqueeze(0).repeat(self.n, 1)
        self._ep_delay = torch.zeros(self.n, dtype=torch.long, device=device)
        self._push_step = torch.full((self.n,), 1 << 30, dtype=torch.long, device=device)
        self._push_dur = torch.zeros(self.n, dtype=torch.long, device=device)
        self._push_force = torch.zeros(self.n, 3, device=device)
        self._t = torch.zeros(self.n, dtype=torch.long, device=device)
        self._prev_action = torch.zeros(self.n, self.j, device=device)
        self._prev_err = z.clone()
        self._ep_peak = torch.ones(self.n, device=device)
        self._act_buf = torch.zeros(self.n, self._max_delay + 1, self.j, device=device)
        self._buf_w = torch.zeros(self.n, dtype=torch.long, device=device)
        self._gait_prev = torch.zeros(self.n, 3, device=device)
        self._gait_has_prev = torch.zeros(self.n, dtype=torch.bool, device=device)
        self._gait_v = torch.zeros(self.n, 3, device=device)
        self._prev_lin_w = torch.zeros(self.n, 3, device=device)
        # morph(形变族): 目标位形 = neutral + idx 段 amp·progress(逐回合采样)
        self._morph_cfg = None
        if self._ctx["morph"]:
            mo = self._ctx["morph"]
            self._morph_cfg = (torch.tensor(np.asarray(mo["joint_idx"], dtype=np.int64), device=device),
                               torch.tensor(np.asarray(mo["amp_rad"], dtype=np.float32), device=device),
                               mo.get("progress", [0.0, 1.0]))
        self.reset()

    # ── 回合采样(课程;按 env 子集行级——auto-reset 不污染其余环境) ──
    def _sample_episode(self, ids: list[int]) -> None:
        t, torch, r = self.task, self._torch, self.rs
        dev, n = self.device, len(ids)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=dev)
        if "goal" in t["obs_mode"]:
            bx = np.asarray(t["goal_box"], dtype=np.float64)
            f_range = t.get("goal_f_range", [0.0, 1.0])
            goal = np.stack([rewards.sample_goal(bx, f_range, r) for _ in range(n)])
            self._ep_goal[ids_t] = torch.as_tensor(goal, dtype=torch.float32, device=dev)
        if "contact" in t["obs_mode"]:
            f_lo, f_hi = t.get("goal_f_range", [0.3, 0.8])
            self._ep_f[ids_t] = torch.as_tensor(r.uniform(f_lo, f_hi, size=n).astype(np.float32),
                                                device=dev)
        if self._ctx["family"] == "expression":
            self._ep_style[ids_t] = torch.as_tensor(r.uniform(0.0, 1.0, size=n).astype(np.float32),
                                                    device=dev)
        if self._morph_cfg is not None:
            idx_t, amp_t, prog = self._morph_cfg
            p = r.uniform(*prog, size=n).astype(np.float32)
            tgt = self._neutral_t.unsqueeze(0).repeat(n, 1)
            tgt[:, idx_t] += amp_t.unsqueeze(0) * torch.as_tensor(p, device=dev).unsqueeze(1)
            self._ep_morph[ids_t] = tgt
        if self._max_delay > 0:
            self._ep_delay[ids_t] = torch.as_tensor(
                np.ceil(r.uniform(self.delay_range[0], self.delay_range[1],
                                  size=n) / 1000.0 * self.control_hz).astype(np.int64), device=dev)
        else:
            self._ep_delay[ids_t] = 0
        self._act_buf[ids_t] = 0.0
        self._buf_w[ids_t] = 0
        push = t.get("push")
        if push:
            base = float(push.get("n", 100.0)) * self.curriculum_frac
            self._push_step[ids_t] = torch.as_tensor(
                r.randint(0, int(self.max_ep_steps * 0.7), size=n), device=dev)
            self._push_dur[ids_t] = int(push.get("dur_s", 0.5) * self.control_hz)
            mag = torch.as_tensor(base * r.uniform(0.5, 1.0, size=n).astype(np.float32), device=dev)
            th = torch.as_tensor(r.uniform(0, 2 * np.pi, size=n).astype(np.float32), device=dev)
            self._push_force[ids_t] = torch.stack([mag * torch.cos(th), mag * torch.sin(th),
                                                   torch.zeros_like(mag)], dim=1)
        else:
            self._push_step[ids_t] = 1 << 30
            self._push_dur[ids_t] = 0
            self._push_force[ids_t] = 0.0
        self._t[ids_t] = 0
        self._prev_action[ids_t] = 0.0
        self._prev_err[ids_t] = 0.0
        self._ep_peak[ids_t] = 1.0
        self._gait_prev[ids_t] = 0.0
        self._gait_has_prev[ids_t] = False
        self._gait_v[ids_t] = 0.0
        self._prev_lin_w[ids_t] = 0.0

    def _reset_idx(self, ids_t) -> None:
        """写初始态(站姿/倒地) + 初始噪声 → forward → 缓冲刷新。"""
        torch = self._torch
        n = len(ids_t)
        noise = (torch.rand(n, self.j, device=self.device) * 2 - 1) * 0.03
        qpos = self._neutral_t.unsqueeze(0).expand(n, -1).clone() + noise
        root = torch.zeros(n, 7, device=self.device)
        root[:, 2] = 1.0
        root[:, 0] = 1.0   # quat wxyz 单位
        if self.start_fallen:   # 倒地初始位(恢复训练): 侧躺——与 MuJoCo 后端同值
            root[:, 2] = 0.35
            root[:, 0] = 0.7071
            root[:, 1] = 0.7071
        vel = torch.zeros(n, self.j, device=self.device)
        self._robot.write_root_pose_to_sim(root, env_ids=ids_t)                   # C8
        self._robot.write_joint_state_to_sim(qpos, vel,
                                             joint_ids=self._idx.unsqueeze(0).expand(n, -1),
                                             env_ids=ids_t)
        self._robot.set_joint_position_target(qpos,
                                              joint_ids=self._idx.unsqueeze(0).expand(n, -1))
        self._robot.write_data_to_sim()
        self._robot.reset(env_ids=ids_t)
        self._sim.forward()
        self._apply_domain_randomization(ids_t)   # C5 域随机化批写

    def _apply_domain_randomization(self, ids_t) -> None:
        """C5(生产训练前置——GPU 首跑校准点):质量/摩擦抖动 physx 批写。

        语义与 MuJoCo 后端逐项同源(质量±mj/摩擦±fj,均匀分布)。失败必须
        fail-loud——静默跳过 = 无域随机化训练,策略过拟合标称动力学。
        版本漂移点 = root_physx_view 批写 API 名与材质属性形状(GPU 首跑核对)。
        """
        mj, fj = self._mass_jitter, self._friction_jitter
        if mj <= 0 and fj <= 0:
            return
        torch = self._torch
        view = getattr(self._robot, "root_physx_view", None)
        if view is None:
            raise RuntimeError("C5: root_physx_view 不可用——isaacsim 版本 API 校准")
        n = len(ids_t)
        if self._base_masses is None:
            self._base_masses = view.get_body_masses().clone()          # (N, n_b)
        if mj > 0:
            base = self._base_masses[ids_t]
            scale = 1.0 + (torch.rand(n, base.shape[1], device=self.device) * 2 - 1) * mj
            view.set_body_masses(base * scale, env_ids=ids_t)
        if fj > 0:
            if self._base_mats is None:
                self._base_mats = view.get_material_properties().clone()  # (N, n_s, 3)
            mats = self._base_mats[ids_t].clone()
            scale = 1.0 + (torch.rand(n, mats.shape[1], device=self.device) * 2 - 1) * fj
            mats[..., 0] *= scale                                        # 静摩擦
            mats[..., 1] *= scale                                        # 动摩擦
            view.set_material_properties(mats, env_ids=ids_t)

    def reset(self, envs=None) -> np.ndarray:
        ids = list(range(self.n)) if envs is None else list(envs)
        self._sample_episode(ids)
        self._reset_idx(self._torch.as_tensor(ids, dtype=torch.long, device=self.device))
        return self._observe().cpu().numpy()

    # ── 主循环 ─────────────────────────────────────────────
    def set_curriculum(self, frac: float) -> None:
        """扰动课程(0~1)——下个回合采样起生效(与 MuJoCo 后端同语义)。"""
        self.curriculum_frac = float(np.clip(frac, 0.0, 1.0))

    def step(self, actions: np.ndarray):
        torch = self._torch
        a = torch.as_tensor(np.asarray(actions, dtype=np.float32), device=self.device)
        if a.shape != (self.n, self.j):
            raise ValueError(f"动作形状 {tuple(a.shape)} ≠ ({self.n},{self.j})")
        # 电机延迟环形缓冲(§10: 0~20ms)——warmup 期回退当前拍(与 MuJoCo 后端同语义)
        if self._max_delay > 0:
            self._act_buf[self._eye, self._buf_w] = a
            a_eff = self._act_buf[self._eye, (self._buf_w - self._ep_delay) % (self._max_delay + 1)]
            a_eff = torch.where((self._t <= self._ep_delay).unsqueeze(1), a, a_eff)
            self._buf_w = (self._buf_w + 1) % (self._max_delay + 1)
        else:
            a_eff = a
        target = torch.clamp(self._neutral_t + a_eff * self._scale_t,
                             self._soft_lo_t, self._soft_hi_t)
        # 推力扰动(课程窗口内对骨盆施加水平外力——C6)
        if bool(self.task.get("push")):
            active = (self._t >= self._push_step) & (self._t < self._push_step + self._push_dur)
            wrench_f = torch.zeros(self.n, 1, 3, device=self.device)
            wrench_f[:, 0, :] = torch.where(active.unsqueeze(1), self._push_force,
                                            torch.zeros_like(self._push_force))
            self._robot.write_external_wrench_to_sim(wrench_f, torch.zeros_like(wrench_f))   # C6
        self._robot.set_joint_position_target(target,
                                              joint_ids=self._idx.unsqueeze(0).expand(self.n, -1))
        self._robot.write_data_to_sim()
        for _ in range(self._decimation):
            self._sim.step()
        self._scene.update(dt=_SIM_DT * self._decimation)
        self._t += 1

        rew, done, infos = self._reward_and_terminate(a)
        obs = self._observe().cpu().numpy()
        return obs, rew.astype(np.float32), done, infos

    # ── 观测(契约批式装配;白名单同序) ──────────────────────
    def _observe(self):
        torch = self._torch
        d = self._robot.data
        q = d.joint_pos[:, self._idx]
        dq = d.joint_vel[:, self._idx]
        tau = (d.applied_torque[:, self._idx] if hasattr(d, "applied_torque")
               else torch.zeros_like(q))
        root = d.root_state_w                     # [pos3|quat4|lin_vel3|ang_vel3]
        quat = root[:, 3:7]                       # wxyz——与 MuJoCo framequat 同约定
        R = _quat_to_mat(quat)
        gyro_b = torch.bmm(R.transpose(1, 2), root[:, 10:13].unsqueeze(2)).squeeze(2)
        if hasattr(d, "root_link_lin_acc_b"):     # C7 原生比力(体系)
            acc_b = d.root_link_lin_acc_b
        else:                                     # C7 有限差分回退
            acc_w = (root[:, 7:10] - self._prev_lin_w) / (_SIM_DT * self._decimation)
            acc_b = torch.bmm(R.transpose(1, 2),
                              (acc_w + torch.as_tensor(_G_UP, device=self.device)).unsqueeze(2)
                              ).squeeze(2)
        blocks = [q, dq, tau, quat, gyro_b, acc_b]
        if self.enc_sigma > 0:                    # 编码器噪声(q/dq 独立采样——与 MuJoCo 后端同)
            blocks[0] = blocks[0] + torch.randn_like(q) * self.enc_sigma
            blocks[1] = blocks[1] + torch.randn_like(dq) * self.enc_sigma
        if self.imu_sigma > 0:
            for i in (3, 4, 5):
                blocks[i] = blocks[i] + torch.randn_like(blocks[i]) * self.imu_sigma
        if "contact" in self.task["obs_mode"]:
            fl = self._contact_l.data.net_forces_w[:, 0, :]    # 世界系力(C4)
            fr = self._contact_r.data.net_forces_w[:, 0, :]
            zero = torch.zeros_like(fl)
            contact = torch.cat([fl, zero, fr, zero], dim=1)   # 6D×2 路;力矩通道 = 零杠杆保守近似(文件头诚实边界)
            if self.contact_sigma > 0:
                contact = contact + torch.randn_like(contact) * self.contact_sigma
            blocks.append(contact)
        if "goal" in self.task["obs_mode"]:
            blocks.append(self._ep_goal)
        obs = torch.cat(blocks, dim=1)
        if obs.shape[1] != self.assembler.dim:
            raise RuntimeError(f"装配宽度 {obs.shape[1]} ≠ 契约宽度 {self.assembler.dim}")
        if not torch.isfinite(obs).all():
            raise RuntimeError("观测含非有限值(传感故障须诚实暴露)")
        return obs

    # ── 奖励/终止(共享层;批数组) ───────────────────────────
    def _reward_and_terminate(self, a):
        torch = self._torch
        d = self._robot.data
        root = d.root_state_w
        q = d.joint_pos[:, self._idx]
        tau = (d.applied_torque[:, self._idx] if hasattr(d, "applied_torque")
               else torch.zeros_like(q))
        m = {
            "action_rate": (a - self._prev_action).abs().mean(dim=1).cpu().numpy(),
            "tau_norm": tau.abs().mean(dim=1).cpu().numpy() / max(self._kp, 1.0),
            "t": self._t.cpu().numpy(),
            "q": q.cpu().numpy(),
            "upright": root[:, 3].cpu().numpy(),
            "root_z": root[:, 2].cpu().numpy(),
            "quat_w": root[:, 3].cpu().numpy(),
        }
        self._prev_action = a.clone()
        if self._ctx["gait"]:
            self._root_track_batch(root[:, 0:3], root[:, 3:7])
            tgt = self._ctx["gait"].get("target", {"vx": 0.75, "vy": 0.0, "wz": 0.0})
            vxy = self._gait_v[:, 0:2] - torch.tensor([tgt["vx"], tgt["vy"]], device=self.device)
            ve = vxy.norm(dim=1) + (self._gait_v[:, 2] - tgt["wz"]).abs()
            m["vel_err"] = ve.cpu().numpy()
        if self.airborne_ok:
            self._ep_peak = torch.maximum(self._ep_peak, root[:, 2])
            m["peak_z"] = self._ep_peak.cpu().numpy()
            m["upright_end"] = root[:, 3].abs().cpu().numpy()
        if "goal" in self.task["obs_mode"]:
            gw = self._goal_world(root[:, 0:3], root[:, 3:7]).cpu().numpy()
            err = np.linalg.norm(self._ee_pos().cpu().numpy() - gw, axis=1)
            m["ee_err"] = err
            m["prev_err"] = self._prev_err.cpu().numpy()
            self._prev_err = torch.as_tensor(err, dtype=torch.float32, device=self.device)
        if self._ctx["contact_obs"] and self._ctx["family"] == "manipulation":
            m["fmag"] = np.linalg.norm(self._contact_l.data.net_forces_w[:, 0, :].cpu().numpy(),
                                       axis=1)
            m["ep_f"] = self._ep_f.cpu().numpy()
        if self._ctx["family"] == "deform":
            m["morph_target"] = self._ep_morph.cpu().numpy()
        if self._ctx["family"] == "expression":
            m["style"] = self._ep_style.cpu().numpy()
        rew, _terms = rewards.compute_reward(self._ctx, m)
        done, _flags = rewards.check_termination(self._ctx, m)
        done_np = np.asarray(done, dtype=bool)
        infos = [{} for _ in range(self.n)]
        if done_np.any():
            done_ids = [int(e) for e in np.nonzero(done_np)[0]]
            obs_full = self.reset(done_ids)   # 行级重置;final_obs = 重置后首帧(与 MuJoCo 后端同语义)
            for e in done_ids:
                infos[e] = {"final_obs": obs_full[e]}
        return rew, done_np, infos

    def _ee_pos(self):
        """末端点 = r_hand 链接原点(文件头诚实边界:与 MuJoCo site 有常数偏差)。"""
        return self._robot.data.body_pos_w[:, self._ee_bid, :]

    def _goal_world(self, pos, quat):
        """goal 基座系 → 世界系(与 MuJoCo _goal_world 同式: p + R@goal)。"""
        R = _quat_to_mat(quat)
        return pos + torch.bmm(R, self._ep_goal[:, 0:3].unsqueeze(2)).squeeze(2)

    def _root_track_batch(self, pos_w, quat):
        """根位置/偏航有限差分速度(与 MuJoCo _root_track 同式——帧约定无关)。"""
        yaw = _quat_to_yaw(quat)
        dt = _SIM_DT * self._decimation
        cur = torch.cat([pos_w[:, 0:2], yaw.unsqueeze(1)], dim=1)
        has = self._gait_has_prev
        if bool(has.any()):
            dlt = cur - self._gait_prev
            dlt[:, 2] = (dlt[:, 2] + np.pi) % (2 * np.pi) - np.pi
            self._gait_v = torch.where(has.unsqueeze(1), dlt / dt, self._gait_v)
        self._gait_prev = cur.clone()
        self._gait_has_prev[:] = True


def _quat_to_mat(quat):
    """wxyz 四元数 → 旋转矩阵 (N,3,3)——纯 torch,不依赖 isaaclab 版本 API。"""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=1).reshape(-1, 3, 3)


def _quat_to_yaw(quat):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
