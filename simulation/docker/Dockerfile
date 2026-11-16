# ── 灵智仿真验证包镜像（技术文档 §5.2 独立复现的一键载体）─────────────
#
# 关键事实：底座 device-agent-rs:safety 已内置 CycloneDDS 0.10.5 全套
# （libddsc.so + idlc 0.10.5 + gcc）——builder 无需源码构建（git clone 被网络拒的
# 根因也一并消解），直接构建 ddswrapper + 跨语言对拍探针。
#
#   builder  = device-agent-rs:safety（CycloneDDS/gcc/idlc 就绪）
#              → 构建 ddswrapper（libmindpal_dds_std.so——引擎 dlopen 同款库）
#              → 编译 std_probe（桥接器↔包装库跨语言对拍探针）
#   runtime  = device-agent-rs:safety（真 Rust 引擎 /usr/local/bin/mindpal-device-agent）
#              + python3 venv（mujoco/cyclonedds/onnxruntime 套件运行面）
#              + simulation/ 全树（bridge/packages/suites/tasks/tools/runner/demo）
#
# 用法（仓库根执行）:
#   docker build -f simulation/docker/Dockerfile -t mindpal-sim-verifier .
#   docker compose -f simulation/docker/compose.yaml run --rm verifier   # 7 套件一键复现
#   docker compose -f simulation/docker/compose.yaml --profile live up -d bridge
#   docker compose -f simulation/docker/compose.yaml run --rm consumer   # C 探针吃 Python 桥
#
# 引擎在环（下一程）：本镜像含引擎二进制 + wrapper + libddsc——
# 装载 simulation/packages/* 即可经 DDS 对拍桥接器（compose engine 服务待 tri 引擎版套件）

# ── 阶段 1：builder（底座自带 CycloneDDS——直接编译 wrapper + 探针）──
FROM device-agent-rs:safety AS builder
USER root
WORKDIR /w
COPY packages/ddswrapper ./ddswrapper
COPY simulation/docker/std_probe.c ./std_probe.c
RUN cd ddswrapper && sh build.sh 2>&1 | tail -1 \
    && cc -O2 -I. ../std_probe.c -o /w/std_probe -L. -lmindpal_dds_std -lddsc -lm -Wl,-rpath,'$ORIGIN'

# ── 阶段 1.5：引擎 tri 测试编译（load-dynamic=免 ORT 下载；crates.io 需可达）──
FROM rust:1.88-bookworm AS testbuild
RUN apt-get update && apt-get install -y --no-install-recommends     pkg-config libssl-dev libasound2-dev libdbus-1-dev     && rm -rf /var/lib/apt/lists/*
ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C link-arg=-Wl,--unresolved-symbols=ignore-all"
WORKDIR /build
COPY packages/protocol-rs ./packages/protocol-rs
COPY apps/device-agent-rs ./apps/device-agent-rs
# include_str! 契约文件（body_package_controller 引用 repo 根 fixtures——构建上下文必须携带）
COPY fixtures/full-chain-motion-phrase.json ./fixtures/full-chain-motion-phrase.json
# fault_injection_e2e 装载真实宇树 G1 模板——无此目录则故障注入 e2e 全拒
COPY apps/body-packages /build/apps/body-packages
WORKDIR /build/apps/device-agent-rs
# cargo 缓存挂载——源码变更后重编译只增量（registry+target 跨构建持久）
RUN --mount=type=cache,target=/usr/local/cargo/registry     --mount=type=cache,target=/usr/local/cargo/git     --mount=type=cache,target=/build/apps/device-agent-rs/target     cargo test --lib --no-run --no-default-features     --features local_tts,safety,biometric,dense_vector     && cp $(ls -t target/debug/deps/mindpal_device_agent-* | grep -v '\.d$' | grep -v '\.so$' | head -1) /out-tri-libtest

# ── 阶段 2：runtime（引擎底座 + python 套件运行面）──
FROM device-agent-rs:safety
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv libosmesa6 libosmesa6-dev libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
# 仿真感知在环：MUJOCO_GL=osmesa——无 GPU 容器内的 headless 渲染后端
# （相机帧通路 FrameServer 的渲染依赖，EGL 在无 GPU 环境不可用）
ENV MUJOCO_GL=osmesa
ENV PATH=/opt/venv/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/mpdds:/usr/local/lib
COPY --from=builder /w/ddswrapper/libmindpal_dds_std.so /opt/mpdds/
COPY --from=builder /w/ddswrapper/joint_sim /opt/mpdds/
COPY --from=builder /w/std_probe /opt/mpdds/
RUN pip install --no-cache-dir \
    mujoco cyclonedds onnxruntime onnx numpy pyyaml jsonschema pillow pybullet
COPY --from=testbuild /out-tri-libtest /build/tri_libtest
COPY apps/device-agent-rs/fixtures/tri-family /build/apps/device-agent-rs/fixtures/tri-family
COPY simulation /build/simulation
COPY simulation/docker/entry-joint-sim.sh /usr/local/bin/joint_sim
RUN chmod +x /usr/local/bin/joint_sim
ENV PYTHONPATH=/build/simulation
ENV MINDPAL_SIM_PKG=/build/apps/device-agent-rs/fixtures/tri-family/body-package.toml
ENV MINDPAL_SIM_MODEL=/build/simulation/assets/models/biped_29dof.xml
WORKDIR /build/simulation
CMD ["python3", "-m", "runner.run_all"]

# 底座镜像 ENTRYPOINT=mindpal-device-agent——验证包以 python 套件为主命令，重置
ENTRYPOINT []
