"""MuJoCo 物理装载/步进/传感读取——桥接器的物理背板。

模型约定（assets/models/*.xml）：
  - 关节 = joint_mapping.ordered 同名 hinge/slide（映射序目标直驱）
  - 无需 actuator 定义——桥接器在关节层仿真 PD 伺服 + 重力/科氏前馈
    （tau = kp·(tgt−q) − kd·dq + qfrc_bias——MuJoCo 每步算好的重力+科氏项，
    真机低层控制器的标准配置——仿真保真度与真机一致），
    经 data.qfrc_applied 注入：与引擎契约同构（引擎下发目标+增益，本体层伺服）
  - 可选传感（存在即读，缺席诚实降级）：site "imu" + 传感 gyro/accelerometer/
    framequat（IMU 块）；force/torque 传感（ft 块——路序 = 包文档化）

诚实纪律：回流只含现实同款传感量——上帝视角真值（qpos 直通例外=编码器等价物）
永不扩流；IMU 缺传感时输出单位四元数+零（静置物理事实，非伪造动态）。
"""
from __future__ import annotations

import math

import mujoco
import numpy as np


class MjModel:
    """单模型物理会话：目标流 → mj_step → 传感快照。"""

    def __init__(self, xml_path: str, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.joint_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                            for i in range(self.model.njnt)]
        self.joint_qpos_addr = [int(self.model.jnt_qposadr[i]) for i in range(self.model.njnt)]
        self.joint_dof_addr = [int(self.model.jnt_dofadr[i]) for i in range(self.model.njnt)]
        self._imu_site = self._find_site("imu")
        self._sensor_ids = {name: self._find_sensor(name)
                            for name in ("gyro", "accelerometer", "framequat")}
        self._ft_sensors = [(self._find_sensor(f"ft_{i}_force"),
                             self._find_sensor(f"ft_{i}_torque"))
                            for i in range(8)]
        self._last_targets = np.zeros(self.model.nu if self.model.nu else len(self.joint_names))
        import threading
        self._render_lock = threading.Lock()
        self._external: dict[int, float] = {}   # dof → 外扰力矩（伺服域叠加，suite 注入通道）

    def set_external(self, dof_addr: int, torque: float) -> None:
        """持续外扰力矩（tri_01 平衡断言/套件注入的数据通道——每子步与 PD 叠加）。"""
        if torque == 0.0:
            self._external.pop(dof_addr, None)
        else:
            self._external[dof_addr] = torque

    # ── 寻址 ──
    def _find_site(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return sid if sid >= 0 else -1

    def _find_sensor(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return sid if sid >= 0 else -1

    def channels(self, mapping: list[str]) -> list[int]:
        """mapping 序 → (qpos_adr, dof_adr)；未命中 = 诚实报错（模型与包必须一致）。"""
        out = []
        for name in mapping:
            if name not in self.joint_names:
                raise KeyError(f"模型缺映射关节 {name}（模型关节={self.joint_names}）")
            j = self.joint_names.index(name)
            out.append((self.joint_qpos_addr[j], self.joint_dof_addr[j]))
        return out

    def step(self, targets: list[float], channels: list[tuple[int, int]],
             kp: float, kd: float, n_substeps: int) -> None:
        """引擎一拍 = N 子步：逐子步 PD 伺服（关节层）→ mj_step。"""
        for _ in range(n_substeps):
            q = self.data.qpos
            dq = self.data.qvel
            for i, (qa, da) in enumerate(channels):
                tau = (kp * (targets[i] - q[qa]) - kd * dq[da]
                       + self.data.qfrc_bias[da] + self._external.get(da, 0.0))
                self.data.qfrc_applied[da] = tau
            mujoco.mj_step(self.model, self.data)

    def disturbance(self, dof_addr: int, torque: float) -> None:
        """外部扰动注入（tri_01 平衡断言的数据通道）。"""
        self.data.xfrc_applied[:] = 0
        self.data.qfrc_applied[dof_addr] += torque

    # ── 传感快照 ──
    def read_state(self, channels: list[tuple[int, int]], slots: int = 35) -> dict:
        """读引擎快照槽位：q/dq/tau 按映射序填槽（槽外零）、IMU、力觉路。"""
        n = len(channels)
        q = [0.0] * slots
        dq = [0.0] * slots
        tau = [0.0] * slots
        for i, (qa, da) in enumerate(channels):
            q[i] = float(self.data.qpos[qa])
            dq[i] = float(self.data.qvel[da])
            tau[i] = float(self.data.qfrc_applied[da])
        imu = self._read_imu()
        ft = self._read_ft()
        return {"n": n, "q": q, "dq": dq, "tau": tau, "imu": imu,
                "n_ft": ft["n_ft"], "ft": ft["ft"]}

    def _read_imu(self) -> list[float]:
        """IMU 块 10 元：framequat/gyro/accelerometer 传感在场即读；缺席 = 静置事实。"""
        quat = [1.0, 0.0, 0.0, 0.0]
        gyro = [0.0] * 3
        accel = [0.0, 0.0, 0.0]
        if self._sensor_ids["framequat"] >= 0:
            adr = int(self.model.sensor_adr[self._sensor_ids["framequat"]])
            quat = [float(v) for v in self.data.sensordata[adr:adr + 4]]
        if self._imu_site >= 0 and self._sensor_ids["gyro"] >= 0:
            adr = int(self.model.sensor_adr[self._sensor_ids["gyro"]])
            gyro = [float(v) for v in self.data.sensordata[adr:adr + 3]]
        if self._imu_site >= 0 and self._sensor_ids["accelerometer"] >= 0:
            adr = int(self.model.sensor_adr[self._sensor_ids["accelerometer"]])
            accel = [float(v) for v in self.data.sensordata[adr:adr + 3]]
        return list(quat) + list(gyro) + list(accel)

    def _read_ft(self) -> dict:
        """N 路六维力觉：ft_{i}_force/ft_{i}_torque 传感对在场即一路；缺席 n_ft=0（诚实缺省）。"""
        ft = [[0.0] * 6 for _ in range(8)]
        n_ft = 0
        for i in range(8):
            f_id, t_id = self._ft_sensors[i]
            if f_id < 0 or t_id < 0:
                break
            fa = int(self.model.sensor_adr[f_id])
            ta = int(self.model.sensor_adr[t_id])
            ft[i] = [float(v) for v in self.data.sensordata[fa:fa + 3]] + \
                    [float(v) for v in self.data.sensordata[ta:ta + 3]]
            n_ft += 1
        return {"n_ft": n_ft, "ft": ft}

    # ── 相机渲染 ──
    def render_frame_jpeg(self, cam_name: str, width: int, height: int) -> bytes | None:
        """仿真相机渲染 → JPEG 字节（http_image 通路；渲染器不可用 = None 诚实缺帧）。"""
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            return None
        try:
            from PIL import Image
        except ImportError:
            return None
        with self._render_lock:
            renderer = getattr(self, "_renderer", None)
            if renderer is None or renderer[0] != (width, height):
                renderer = mujoco.Renderer(self.model, height=height, width=width)
                self._renderer = ((width, height), renderer)
            renderer.update_scene(self.data, camera=cam_id)
            rgb = renderer.render()
        import io
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG")
        return buf.getvalue()

    def release_renderer(self) -> None:
        """显式释放 GL 渲染器——避免解释器关闭期上下文析构竞争（exit 段错误根因）。"""
        import threading
        with self._render_lock:
            renderer = getattr(self, "_renderer", None)
            if renderer is not None:
                try:
                    renderer[1].close()
                except Exception:
                    pass
                self._renderer = None

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self, keyframe: str | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        if keyframe:
            k = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if k >= 0:
                mujoco.mj_resetDataKeyframe(self.model, self.data, k)
        mujoco.mj_forward(self.model, self.data)


def substep_ratio(sim_hz: float, policy_hz: float, model_timestep: float) -> int:
    """引擎一拍的子步数 = 仿真拍/策略拍（技术文档 §2.3 频率契约）。"""
    ratio = int(round(sim_hz / policy_hz))
    return max(1, ratio)


def eps_ne(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) > tol


def unit_quat_check(imu: list[float]) -> bool:
    w, x, y, z = imu[:4]
    return abs(math.sqrt(w * w + x * x + y * y + z * z) - 1.0) < 1e-3
