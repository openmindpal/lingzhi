"""mindpal 标准 DDS 线类型（桥接器 ↔ 引擎对拍依据）。

权威 = packages/ddswrapper/mindpal.idl + src/wrapper.c + 引擎 dds_cdr.rs：
  - 指令（引擎→本体）：mindpal::JointTargets，CDR-LE 封装头 4B + 体 424B
    （n:u32 + q[35]:f32 + kp[35]:f32 + kd[35]:f32——映射序目标弧度 + 增益）
  - 状态（本体→引擎）：mindpal::JointStates，CDR-LE 头 4B + 体 660B
    （n:u32 + q/dq/tau 各 35:f32 + imu[10]:f32 + n_ft:u32 + ft[8][6]:f32）
  - QoS：指令 writer 默认（reliable）；状态 reader best_effort（wrapper.c:81-82）
  - 全字段 4 字节自然对齐 ⇒ CDR-LE 体 == C 结构体映像（wrapper memcpy 零解析）

cyclonedds-python 类型名经 typename 注解钉死为 IDL 全名——与包装库
dds_topic_descriptor 线上类型匹配（SEDP 发现按类型一致性）。
"""
import struct
from dataclasses import dataclass

from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import array, float32, uint32

# 注意：本模块禁用 `from __future__ import annotations`——IDL 机制要求字段注解
# 为真实类型对象（字符串注解无法解析 array[...]）。

MP_JOINT_SLOTS = 35          # 电机槽位契约（跨包装库单一槽位——mindpal.idl）
MP_FT_MAX = 8                # 力觉路数容量（观测空间 v5）
IMU_W = 10                   # quat_wxyz(4) + gyro_xyz(3) + accel_xyz(3)

JOINTTARGETS_BODY_BYTES = 4 + MP_JOINT_SLOTS * 4 * 3          # 424
JOINTSTATES_BODY_BYTES = 4 + MP_JOINT_SLOTS * 4 * 3 + IMU_W * 4 + 4 + MP_FT_MAX * 6 * 4  # 660
CDR_LE_HEADER = b"\x00\x01\x00\x00"                            # OMG CDR 1.x 小端封装头


@dataclass
class JointTargets(IdlStruct, typename="mindpal::JointTargets"):
    """引擎 → 本体：映射序绝对关节目标（rad）+ PD 增益（来自包 profile.dynamics）。"""

    n: uint32
    q: array[float32, MP_JOINT_SLOTS]
    kp: array[float32, MP_JOINT_SLOTS]
    kd: array[float32, MP_JOINT_SLOTS]


@dataclass
class JointStates(IdlStruct, typename="mindpal::JointStates"):
    """本体 → 引擎：全量状态回读（观测装配 v5 数据源；槽位序 = 包 state_order）。"""

    n: uint32
    q: array[float32, MP_JOINT_SLOTS]
    dq: array[float32, MP_JOINT_SLOTS]
    tau: array[float32, MP_JOINT_SLOTS]
    imu: array[float32, IMU_W]
    n_ft: uint32
    ft: array[array[float32, 6], MP_FT_MAX]


def zero_states_slots() -> JointStates:
    """全零状态样本（断流前静默基线——仅供测试，正常路径禁用：诚实缺帧优先）。"""
    return JointStates(
        n=0,
        q=[0.0] * MP_JOINT_SLOTS,
        dq=[0.0] * MP_JOINT_SLOTS,
        tau=[0.0] * MP_JOINT_SLOTS,
        imu=[1.0, 0.0, 0.0, 0.0] + [0.0] * 6,
        n_ft=0,
        ft=[[0.0] * 6 for _ in range(MP_FT_MAX)],
    )


def pack_jointtargets_body(n: int, q: list[float], kp: float, kd: float) -> bytes:
    """按引擎 dds_cdr::encode_jointtargets 同构打包体 424B（自测/回放用）。"""
    q = list(q[:n]) + [0.0] * (MP_JOINT_SLOTS - min(n, MP_JOINT_SLOTS))
    return struct.pack("<I", n) + struct.pack(f"<{MP_JOINT_SLOTS}f", *q) + \
        struct.pack(f"<{MP_JOINT_SLOTS}f", *([kp] * MP_JOINT_SLOTS)) + \
        struct.pack(f"<{MP_JOINT_SLOTS}f", *([kd] * MP_JOINT_SLOTS))


assert struct.calcsize("<I") + MP_JOINT_SLOTS * 4 * 3 == JOINTTARGETS_BODY_BYTES
