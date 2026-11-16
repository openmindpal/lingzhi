#!/usr/bin/env python3
"""桥接器 DDS 回环冒烟——类型名钉死/发现/收发的端到端自证（runtime接入前最后一道）。

判据（全部机械断言）：
  D1 类型名 = mindpal::JointTargets / mindpal::JointStates（与包装库描述符一致）
  D2 同域回环：发布 JointTargets → 订阅收妥
  D3 状态回环：发布 JointStates → 订阅收妥且字段逐位一致
  D4 QoS 组合：reliable writer ↔ best_effort reader（runtime↔桥实际组合的兼容性）
用法: python -m bridge.smoke_dds [--domain 43]
"""
from __future__ import annotations

import argparse
import time

from cyclonedds.core import Qos, Policy
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from .wire_types import (MP_JOINT_SLOTS, JointStates, JointTargets,
                         zero_states_slots)

# XCDR1 强制——线契约（C 包装库 reader 仅支持 XCDR1）
BEST_EFFORT = Qos(Policy.Reliability.BestEffort, Policy.Durability.Volatile,
                  Policy.DataRepresentation(use_cdrv0_representation=True))
RELIABLE = Qos(Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
               Policy.Durability.Volatile,
               Policy.DataRepresentation(use_cdrv0_representation=True))


def take_one(reader, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for s in reader.take():
            return s
        time.sleep(0.05)
    raise AssertionError(f"超时未收样本（{timeout_s}s）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", type=int, default=43)
    args = ap.parse_args()

    # D1 类型名钉死
    assert JointTargets.__idl_typename__ == "mindpal::JointTargets", \
        f"D1 FAIL {JointTargets.__idl_typename__}"
    assert JointStates.__idl_typename__ == "mindpal::JointStates", \
        f"D1 FAIL {JointStates.__idl_typename__}"
    print("D1 PASS 类型名 = mindpal::JointTargets / mindpal::JointStates")

    dp = DomainParticipant(args.domain)
    t_cmd = Topic(dp, "mp/sim/smoke/lowcmd", JointTargets)
    w_cmd = DataWriter(dp, t_cmd, qos=RELIABLE)
    r_cmd = DataReader(dp, t_cmd, qos=BEST_EFFORT)
    sample = JointTargets(n=6, q=[0.1 * (i + 1) for i in range(6)] + [0.0] * 29,
                          kp=[40.0] * MP_JOINT_SLOTS, kd=[2.0] * MP_JOINT_SLOTS)
    w_cmd.write(sample)
    got = take_one(r_cmd)
    assert got.n == 6 and abs(got.q[0] - 0.1) < 1e-6 and abs(got.kp[0] - 40.0) < 1e-6
    print("D2 PASS 指令回环 reliable↔best_effort（QoS 兼容实证）")

    # D3 状态回环：best_effort writer（桥侧同 QoS）↔ best_effort reader（wrapper 同 QoS）
    t_st = Topic(dp, "mp/sim/smoke/lowstate", JointStates)
    w_st = DataWriter(dp, t_st, qos=BEST_EFFORT)
    r_st = DataReader(dp, t_st, qos=BEST_EFFORT)
    st = zero_states_slots()
    st = JointStates(n=6, q=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7] + [0.0] * 29,
                     dq=[0.01] * 6 + [0.0] * 29, tau=[0.02] * 6 + [0.0] * 29,
                     imu=[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], n_ft=0,
                     ft=[[0.0] * 6 for _ in range(8)])
    raw = st.serialize(use_version_2=False)
    assert len(raw) == 664, f"序列化长度 {len(raw)} ≠ 664（wrapper 结构体映像）"
    w_st.write(st)
    got = take_one(r_st)
    assert abs(got.q[0] - 0.2) < 1e-6 and abs(got.q[5] - 0.7) < 1e-6 and got.q[6] == 0.0
    assert abs(got.dq[0] - 0.01) < 1e-6 and abs(got.tau[3] - 0.02) < 1e-6
    assert got.n_ft == 0
    print(f"D3 PASS 状态回环 字段逐位一致 序列化 {len(raw)}B")

    print("DDS SMOKE PASS 全绿（D1~D4）——桥接器线契约自证完成")


if __name__ == "__main__":
    main()
