"""桥接器主服务——虚拟机器人 DDS 端点（仿真验证平台唯一自建件主循环）。

用法:
  python -m bridge.server --model ../assets/models/arm_6dof.xml \
      --package ../packages/sim-arm-6dof/body-package.toml [--speed 4.0] [--seed 0]

契约（technical spec §2 / packages/ddswrapper 线上实测）:
  订阅  cmd_topic    mindpal::JointTargets   CDR-LE 428B（reliable——runtime writer 默认）
  发布  state_topic  mindpal::JointStates    CDR-LE 664B（best_effort——wrapper reader 同 QoS）
  节拍  policy 拍 = 包 [protocol].frequency_hz；子步比 = sim_hz/policy_hz
  槽位  状态槽位序 = 包 profile.state_order（缺省 = joint_mapping.ordered）
  相机  http_image GET /frame.jpg（camera_stream，--http-port）

honest纪律: 目标流缺失 = 保持伺服上一目标（物理事实）；模型/包关节不一致 = 启动即退出。
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from cyclonedds.core import Qos, Policy
from cyclonedds.domain import DomainParticipant
from cyclonedds.pub import DataWriter
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from cyclonedds.util import duration

from .camera_stream import FrameServer
from .faults import FaultState, apply_to_snapshot, command_gate
from .mj_model import MjModel, substep_ratio
from .time_control import TickClock
from .wire_types import JointStates, JointTargets

SIM_HZ = 500.0            # 仿真物理拍（MuJoCo timestep=0.002）
# 线契约 = CDR-LE v0（XCDR1 强制——C 包装库 reader 仅支持 XCDR1，缺省 XCDR2 会错位）。
BEST_EFFORT = Qos(Policy.Reliability.BestEffort, Policy.Durability.Volatile,
                  Policy.DataRepresentation(use_cdrv0_representation=True))
RELIABLE = Qos(Policy.Reliability.Reliable(max_blocking_time=duration(seconds=1)),
               Policy.Durability.Volatile,
               Policy.DataRepresentation(use_cdrv0_representation=True))


def load_pkg(path: str) -> dict:
    return tomllib.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="mindpal 仿真桥接器（虚拟机器人 DDS 端点）")
    ap.add_argument("--model", required=True, help="MJCF 模型路径")
    ap.add_argument("--package", required=True, help="仿真本体包 body-package.toml")
    ap.add_argument("--speed", type=float, default=1.0, help="时间加速倍率（>1 = 快于实时）")
    ap.add_argument("--seed", type=int, default=0, help="确定性种子")
    ap.add_argument("--http-port", type=int, default=8284, help="相机帧 HTTP 端口")
    ap.add_argument("--fault", default="none",
                    choices=["none", "stream_loss", "frozen_frame", "estop", "force_spike"])
    ap.add_argument("--max-ticks", type=int, default=0, help="调试上限（0 = 无限）")
    args = ap.parse_args()

    pkg = load_pkg(args.package)
    proto = pkg["protocol"]
    mapping = proto["joint_mapping"]["ordered"]
    dds_cfg = proto["dds"]
    domain_id = int(dds_cfg["domain_id"])
    cmd_topic_name = dds_cfg["actuator_topics"]["joint_cmd"]["topic_name"]
    st_topic_name = dds_cfg["sensor_topics"]["joint_state"]["topic_name"]
    freq = float(proto["frequency_hz"])
    kp = float(pkg["profile"]["dynamics"]["kp"])
    kd = float(pkg["profile"]["dynamics"]["kd"])
    state_order = pkg["profile"].get("state_order") or mapping
    cam = (pkg.get("acquisition", {}).get("cameras") or [{}])[0]

    # 槽位契约：状态槽位 = state_order 序——桥接器按包声明填槽
    model = MjModel(args.model, seed=args.seed)
    if state_order != model.joint_names[:len(state_order)]:
        raise SystemExit(
            f"模型关节 {model.joint_names} 与包 state_order {state_order} 不一致——先改模型或包")
    channels = model.channels(mapping)
    model.reset()
    n_sub = substep_ratio(SIM_HZ, freq, model.timestep)
    period_ms = 1000.0 / freq

    participant = DomainParticipant(domain_id)
    t_cmd = Topic(participant, cmd_topic_name, JointTargets)
    t_st = Topic(participant, st_topic_name, JointStates)
    reader = DataReader(participant, t_cmd, qos=BEST_EFFORT)   # reliable writer 兼容
    writer = DataWriter(participant, t_st, qos=BEST_EFFORT)    # wrapper reader = best_effort

    fault = FaultState()
    if args.fault != "none":
        fault.trigger(args.fault, snapshot=None, ms=800, n=200.0)
    frame_server = None
    if cam.get("kind") == "http_image":
        intr = cam.get("intrinsics", {})
        frame_server = FrameServer(model, cam.get("cam_name", "wrist_cam"),
                                   int(intr.get("width", 640)), int(intr.get("height", 480)),
                                   port=args.http_port)
        frame_server.start()

    print(f"[bridge] domain={domain_id} cmd={cmd_topic_name} state={st_topic_name} "
          f"policy={freq}Hz substeps={n_sub} speed={args.speed} fault={fault.kind}")

    clock = TickClock(period_s=1.0 / freq, speed=args.speed)
    targets = [0.0] * len(mapping)
    last_snap: dict | None = None
    ticks = 0
    try:
        while True:
            # 1) 取最新指令（estop 注入 = 桥侧忽略新目标）
            if not command_gate(fault):
                for sample in reader.take():
                    n = min(int(sample.n), len(mapping))
                    targets = [float(v) for v in sample.q[:n]]

            model.step(targets, channels, kp, kd, n_sub)

            # 3) 传感回流（故障注入按声明处理：停供/冻结/力超限）
            snap = model.read_state(channels)
            out = apply_to_snapshot(fault, snap, period_ms)
            if out is not None:
                writer.write(JointStates(
                    n=out["n"], q=out["q"], dq=out["dq"], tau=out["tau"],
                    imu=out["imu"], n_ft=out["n_ft"], ft=out["ft"]))
                last_snap = out

            clock.wait_tick()
            ticks += 1
            if args.max_ticks and ticks >= args.max_ticks:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if frame_server:
            frame_server.stop()
        print(f"[bridge] 收场 ticks={ticks} sim_time={clock.sim_time:.1f}s "
              f"fault_log={fault.log[-3:]}")


if __name__ == "__main__":
    main()
