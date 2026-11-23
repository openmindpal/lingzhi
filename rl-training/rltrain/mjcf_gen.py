# -*- coding: utf-8 -*-
"""参数化 MJCF 生成器——训练变体资产(自由底座/质量标定/灵巧手/物体/按压面)。

权威分层(诚实声明):
  - 拓扑/关节/限位 = 冒烟 MJCF(biped_29dof.xml,与 unitree-g1 包同源)
  - 几何/质量/手/物体 = envs/g1/body_spec.yaml([样例]——宇树规格回填)
  - 完整本体重精化 = 官方资产替换点(替换生成基座,本生成器的叠加逻辑不变)
产物缓存 = envs/g1/assets/gen/<hash>.xml(内容寻址,删即重生成)。
"""
from __future__ import annotations

import hashlib
import os
import re

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(_HERE, "..", "envs", "g1", "assets")
SPEC = os.path.join(_HERE, "..", "envs", "g1", "body_spec.yaml")
SMOKE = os.path.join(_HERE, "..", "..", "simulation", "assets", "models", "biped_29dof.xml")
OFFICIAL = os.path.join(ASSETS, "official", "g1.xml")
GEN_DIR = os.path.join(ASSETS, "gen")

MASS_RE = re.compile(r'mass="([0-9.eE+-]+)"')


def load_spec(spec_path: str = SPEC) -> dict:
    with open(spec_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _body_span(xml: str, name: str) -> tuple[int, int]:
    """返回 body 元素(含子树)的 [start, end) 字节区间。"""
    m = re.search(rf'<body name="{name}"[ >]', xml)
    if not m:
        raise RuntimeError(f"MJCF 缺 body {name}")
    depth, i = 0, m.start()
    for tok in re.finditer(r"<body\b|</body>", xml[m.start():]):
        if tok.group(0) == "<body":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.start(), m.start() + tok.end()
    raise RuntimeError(f"body {name} 未闭合")


def _mass_scale(xml: str, target_kg: float) -> str:
    """总质量标度(简化标定——[样例]总质量,宇树规格回填)。"""
    masses = [float(v) for v in MASS_RE.findall(xml)]
    total = sum(masses)
    f = target_kg / total
    return MASS_RE.sub(lambda m: f'mass="{float(m.group(1)) * f:.4f}"', xml)


def _hand_chain(side: str, hand: dict) -> str:
    """单手手指链(插入 hand body 内部末尾)——[样例]几何,由 spec.fingers 驱动(五指)。

    段体名 = {side}_s_{joint}(接触路由按前缀聚合);根段横向铺开在掌部。
    """
    r = hand["finger_r_m"]
    lim = hand["limits_rad"]
    fingers = hand["fingers"]
    pm, sm = hand["mass_kg"] * 0.4, hand["mass_kg"] * 0.6
    out = [f'      <geom name="{side}_palm" type="box" size="{hand["palm_lwh_m"][0]/2} '
           f'{hand["palm_lwh_m"][1]/2} {hand["palm_lwh_m"][2]/2}" '
           f'mass="{pm:.3f}" rgba="0.3 0.3 0.4 1"/>']

    def seg(finger, s_i, jn, axis, pos, length, depth, tip):
        lo, hi = lim[jn]
        pad = "  " * (depth + 6)
        s = (f'{pad}<body name="{side}_s_{jn}" pos="{pos}">\n'
             f'{pad}  <joint name="{side}_hand_{jn}_joint" type="hinge" axis="{axis}" '
             f'range="{lo} {hi}"/>\n'
             f'{pad}  <geom type="capsule" fromto="0 0 0 0 0 {length}" size="{r}" '
             f'mass="{sm/6:.4f}"/>\n')
        if tip:
            s += f'{pad}  <site name="{side}_tip_{jn}" pos="0 0 {length + 0.004}" size="0.003"/>\n'
        return s

    for fi, f in enumerate(fingers):
        segs = f["segs"]
        root_x = 0.035 * (fi - (len(fingers) - 1) / 2)
        parts = []
        prev_len = 0.0
        for si, seg_d in enumerate(segs):
            pos = f"{root_x:.3f} -0.01 0.03" if si == 0 else f"0 0 {prev_len:.4f}"
            parts.append(seg(f["name"], fi, seg_d["j"], seg_d["axis"], pos,
                             seg_d["len_m"], 1 + si, tip=(si == len(segs) - 1)))
            prev_len = seg_d["len_m"]
        for i in reversed(range(len(segs))):
            parts.append(f'{"  " * (i + 7)}</body>\n')
        out.append("".join(parts))
    return "".join(out)


def _insert_hands(xml: str, hand: dict) -> str:
    for side in ("l", "r"):
        s, e = _body_span(xml, f"{side}_hand")
        block = xml[s:e]
        inner = block[: block.rfind("</body>")]
        xml = xml[:s] + inner + "\n" + _hand_chain(side, hand) + "    </body>" + xml[e:]
    return xml


def _flex_cloth(o: dict) -> str:
    """柔性布 = 单层 flex 网格(均匀 spacing ≥ 单元尺寸;radius 属性收细)。"""
    n = o["n_grid"][0]
    sp = o["size_m"][0] / n
    pos = " ".join(str(v) for v in o["pos_m"])
    return (f'  <flexcomp name="obj_{o["name"]}" type="grid" count="{n + 1} {n + 1} 1" '
            f'spacing="{sp:.5f} {sp:.5f} {sp:.5f}" pos="{pos}" radius="0.004"/>\n')


def _open_box(o: dict) -> str:
    h = o["half_lwh_m"]
    t = o["wall_t"]
    px, py, pz = o["pos_m"]
    walls = [
        (px, py + h[1] + t / 2, pz, h[0] + t, t, h[2]),
        (px, py - h[1] - t / 2, pz, h[0] + t, t, h[2]),
        (px + h[0] + t / 2, py, pz, t, h[1], h[2]),
        (px - h[0] - t / 2, py, pz, t, h[1], h[2]),
    ]
    add = []
    for i, (x, y, z, sx, sy, sz) in enumerate(walls):
        add.append(f'  <body name="obj_{o["name"]}_w{i}" pos="{x:.4f} {y:.4f} {z:.4f}">\n'
                   f'    <geom name="obj_{o["name"]}_w{i}" type="box" size="{sx/2:.5f} {sy/2:.5f} {sz/2:.5f}" '
                   f'mass="{o["mass_kg"]/4:.4f}" rgba="0.8 0.8 0.85 1" contype="1" conaffinity="1"/>\n'
                   f'  </body>\n')
    return "".join(add)


def _particles(o: dict) -> str:
    n, r = int(o["n"]), o["r_m"]
    fx, fy, fz = o["fill_pos_m"]
    rs = np.random.RandomState(11)
    add = []
    h = 0.02
    for i in range(n):
        x = fx + rs.uniform(-h, h)
        y = fy + rs.uniform(-0.02, 0.02)
        z = fz + 0.012 * (i // 6) + 0.004
        add.append(f'  <body name="obj_{o["name"]}_p{i}" pos="{x:.4f} {y:.4f} {z:.4f}">\n'
                   f'    <freejoint name="obj_{o["name"]}_p{i}_root"/>\n'
                   f'    <inertial pos="0 0 0" mass="0.002" diaginertia="1e-7 1e-7 1e-7"/>\n'
                   f'    <geom name="obj_{o["name"]}_p{i}" type="sphere" size="{r}" mass="0.002" '
                   f'rgba="0.25 0.45 0.9 1" contype="1" conaffinity="1"/>\n'
                   f'  </body>\n')
    return "".join(add)


def _needle_plate(o: dict) -> str:
    gap = o["hole_gap_m"]
    px, py, pz = o["pos_m"]
    t = 0.002
    parts = []
    for sgn in (1, -1):
        x = px + sgn * (gap / 2 + t / 2)
        parts.append(f'  <body name="obj_{o["name"]}_plate{sgn}" pos="{x:.4f} {py:.4f} {pz:.4f}">\n'
                     f'    <geom name="obj_{o["name"]}_plate{sgn}" type="box" size="{t/2:.5f} 0.02 0.04" '
                     f'mass="2" rgba="0.6 0.6 0.65 1" contype="1" conaffinity="1"/>\n'
                     f'  </body>\n')
    return "".join(parts)


def _insert_needle(xml: str, o: dict) -> str:
    """针装右腕(随手动)——尖端 site 供插入深度判定。"""
    s, e = _body_span(xml, "r_hand")
    block = xml[s:e]
    inner = block[: block.rfind("</body>")]
    needle = ('      <geom name="obj_needle" type="capsule" fromto="0 0 0 0 '
              f'{o["needle_len_m"]} 0" size="{o["needle_r_m"]}" mass="0.01" '
              'rgba="0.9 0.9 0.95 1" contype="1" conaffinity="1"/>\n'
              f'      <site name="needle_tip" pos="0 {o["needle_len_m"] + 0.002} 0" size="0.002"/>\n')
    return xml[:s] + inner + "\n" + needle + "    </body>" + xml[e:]


def _insert_objects(xml: str, objects: list, skip: set | None = None) -> str:
    add = []
    for o in objects:
        if o.get("name") in (skip or set()):
            continue
        if o.get("kind") == "flex_cloth":
            add.append(_flex_cloth(o)); continue
        if o.get("kind") == "open_box":
            add.append(_open_box(o)); continue
        if o.get("kind") == "particles":
            add.append(_particles(o)); continue
        if o.get("kind") == "needle_plate":
            add.append(_needle_plate(o)); continue
        if o.get("kind") == "mocap_ball":
            pos = " ".join(str(v) for v in o["pos_m"])
            add.append(f'  <body name="obj_{o["name"]}" mocap="true" pos="{pos}">\n'
                       f'    <geom name="obj_{o["name"]}_geom" type="sphere" size="{o["r_m"]}" '
                       f'mass="{o["mass_kg"]}" rgba="0.95 0.3 0.2 1" contype="1" conaffinity="1"/>\n'
                       f'  </body>\n')
            continue
        size = f'"{o["r_m"]}"' if o["kind"] == "sphere" else \
               (f'"{o["r_m"]} {o["h_m"]/2}"' if o["kind"] == "cylinder" else
                f'"{" ".join(str(v) for v in o.get("half_lwh_m", [0.04, 0.04, 0.04]))}"')
        joint = (f'    <freejoint name="obj_{o["name"]}_root"/>\n'
                 f'    <inertial pos="0 0 0" mass="{o["mass_kg"]}" '
                 f'diaginertia="1e-4 1e-4 1e-4"/>\n') if not o.get("static") else ""
        add.append(
            f'  <body name="obj_{o["name"]}" pos="{o["pos_m"][0]} {o["pos_m"][1]} {o["pos_m"][2]}">\n'
            f'{joint}'
            f'    <geom name="obj_{o["name"]}_geom" type="{o["kind"]}" size={size} '
            f'mass="{o["mass_kg"]}" rgba="0.9 0.6 0.1 1" contype="1" conaffinity="1"/>\n'
            f'  </body>\n')
    return xml.replace("</worldbody>", "".join(add) + "</worldbody>")


def _insert_press_wall(xml: str, wall: dict) -> str:
    g = (f'  <geom name="press_wall" type="box" pos="{wall["pos_m"][0]} {wall["pos_m"][1]} '
         f'{wall["pos_m"][2]}" size="{wall["half_lwh_m"][0]} {wall["half_lwh_m"][1]} '
         f'{wall["half_lwh_m"][2]}" contype="1" conaffinity="1" rgba="0.5 0.5 0.55 1"/>\n')
    return xml.replace("</worldbody>", g + "</worldbody>")


def _insert_terrain(xml: str, terrain: dict) -> str:
    """不平地形(±3cm 台块——平衡族抗扰规格 §7.1)。固定种子 = 确定性资产。"""
    rs = np.random.RandomState(7)
    add = []
    n = int(terrain.get("n_blocks", 10))
    lo, hi = terrain.get("h_range_m", [0.005, 0.03])
    area = terrain.get("area_m", [[-1.0, 1.0], [-1.0, 1.0]])
    hs = terrain.get("half_size_m", [0.08, 0.08])
    for i in range(n):
        x = rs.uniform(*area[0]); y = rs.uniform(*area[1])
        if abs(x) < 0.25 and abs(y) < 0.25:
            continue   # 避开出生点
        h = rs.uniform(lo, hi) / 2
        add.append(f'  <geom name="terrain_{i}" type="box" pos="{x:.3f} {y:.3f} {h:.4f}" '
                   f'size="{hs[0]} {hs[1]} {h:.4f}" contype="1" conaffinity="1" '
                   f'rgba="0.4 0.45 0.4 1"/>\n')
    return xml.replace("</worldbody>", "".join(add) + "</worldbody>")


def _official_base() -> str:
    """官方资产基座(MuJoCo Menagerie unitree_g1 g1_29dof_rev_1_0)→ 平台训练变换。

    官方质量/惯量/几何/限位/IMU 位原样保留(权威 = 官方模型,不做质量标度)。
    变换仅限平台接口面:
      1. 剥 <actuator>/<keyframe>——平台伺服 = qfrc_applied 直写(PD 见 body_spec.gains)
      2. 执行器 forcerange → 关节 actuatorfrcrange(官方力矩限,伺服限幅读这里)
      3. freejoint 改名 root(env_mj 契约名)
      4. 注入契约传感器 framequat/gyro/accelerometer @ imu_in_pelvis(官方 IMU 位)
      5. 注入末端 site l_tool/r_tool(腕偏航链内,[平台定义]近似——按末端规格回填)
      6. meshdir 重定向到官方资产目录(生成产物位于 gen/,官方资产位于 official/)
    """
    xml = open(OFFICIAL, encoding="utf-8").read()
    # 官方力矩限提取: <position ... name="X" joint="Y" forcerange="-88 88"/>
    for m in re.finditer(r'<position[^>]*joint="([a-z_]+)"[^>]*forcerange="(-?[0-9.e]+) (-?[0-9.e]+)"', xml):
        jn, lo, hi = m.group(1), m.group(2), m.group(3)
        pat = re.compile(rf'(<joint name="{jn}"[^>]*?)(/>)')
        if not pat.search(xml):
            raise RuntimeError(f"官方模型缺关节 {jn}(forcerange 无法挂载)")
        xml = pat.sub(rf'\1 actuatorfrcrange="{lo} {hi}"\2', xml, count=1)
    xml = re.sub(r'  <actuator>.*?</actuator>\n', '', xml, flags=re.S)
    xml = re.sub(r'  <keyframe>.*?</keyframe>\n', '', xml, flags=re.S)
    xml = xml.replace('<freejoint name="floating_base_joint"/>', '<freejoint name="root"/>')
    if '<freejoint name="root"/>' not in xml:
        raise RuntimeError("官方模型缺 freejoint(自由底座变体无法生成)")
    sensors = ('  <sensor>\n'
               '    <framequat objtype="site" objname="imu_in_pelvis"/>\n'
               '    <gyro site="imu_in_pelvis"/>\n'
               '    <accelerometer site="imu_in_pelvis"/>\n'
               '  </sensor>\n')
    xml = xml.replace("</mujoco>", sensors + "</mujoco>")
    for side, prefix in (("l", "left"), ("r", "right")):
        pat = re.compile(rf'(<body name="{prefix}_wrist_yaw_link"[^>]*>)')
        if not pat.search(xml):
            raise RuntimeError(f"官方模型缺 {prefix}_wrist_yaw_link(末端 site 无法注入)")
        # 末端 site + 掌框(l_hand/r_hand——焊接刚性体,手包生成器挂点;[样例]几何,
        # 按 Dex3-1 官方手规格回填——官方 with_hands 变体为替换点)
        xml = pat.sub(rf'\1\n          <site name="{side}_tool" pos="0.08 0 0" size="0.004"/>\n'
                      rf'          <body name="{side}_hand" pos="0.08 0 0">\n'
                      rf'          </body>', xml, count=1)
    xml = xml.replace('<compiler angle="radian" meshdir="assets"/>',
                      '<compiler angle="radian" meshdir="../official/meshes"/>')
    # 官方 g1.xml = 纯机器人模型(scene.xml 才有地板)——注入地面(官方 scene 同款平面)
    if 'name="floor"' not in xml:
        xml = xml.replace('<worldbody>',
                          '<worldbody>\n'
                          '    <geom name="floor" type="plane" size="0 0 0.05" '
                          'rgba="0.4 0.45 0.4 1"/>\n', 1)
    header = ("<!-- 官方规格训练变体(自动生成,勿手改): 基座 = MuJoCo Menagerie unitree_g1\n"
              "     g1_29dof_rev_1_0(宇树官方模型,DeepMind 策展) + 平台变换(_official_base)\n"
              "     质量/几何/限位/IMU = 官方原值;PD/力矩限幅 = body_spec.gains + 官方 forcerange -->\n")
    return header + xml


def generate(hands: bool = False, objects: bool = False, press_wall: bool = False,
             terrain: bool = False, cup_on_hand: bool = False, spec_path: str = SPEC,
             src: str | None = None) -> tuple[str, dict]:
    """生成训练变体 XML → 返回 (路径, 生效配置)。

    基座由 body_spec.base 决定: official = 官方资产(规格回填后,真机口径) /
    smoke = 冒烟拓扑([样例],固定底座冒烟测试口径)。
    叠加项: 自由底座(恒开) + 总质量标度(仅 smoke) + 手 + 物体 + 按压面 + 不平地形。
    """
    spec = load_spec(spec_path)
    base = str(spec.get("base", "smoke"))
    if base == "official":
        if not os.path.exists(OFFICIAL):
            raise SystemExit(f"body_spec.base=official 但官方资产缺失: {OFFICIAL}")
        xml = _official_base()
    else:
        xml = open(src or SMOKE, encoding="utf-8").read()
        xml = _mass_scale(xml, float(spec["total_mass_kg"]))
        # 全局 default → 具名 class(bot)+骨盆 childclass:机器人树继承阻尼/contype,
        # worldbody 级新资产(flex/粒子/墙)不受 joint damping 影响(flex 顶点禁 damping)
        xml = xml.replace(
            '<default>\n    <joint damping="0.8" armature="0.01"/>\n'
            '    <geom rgba="0.75 0.75 0.8 1" contype="1" conaffinity="0"/>\n  </default>',
            '<default>\n    <default class="bot">\n      <joint damping="0.8" armature="0.01"/>\n'
            '      <geom rgba="0.75 0.75 0.8 1" contype="1" conaffinity="0"/>\n    </default>\n  </default>')
        xml = xml.replace('<body name="pelvis" pos="0 0 1.0">',
                          '<body name="pelvis" childclass="bot" pos="0 0 1.0">')
        pelvis = re.search(r'<body name="pelvis"[^>]*>', xml)
        if "<freejoint" not in xml:
            xml = xml[:pelvis.end()] + '\n    <freejoint name="root"/>' + xml[pelvis.end():]
    eff = {"base": base, "hands": False, "objects": [], "press_wall": False, "j": 29}
    if hands and spec.get("hands", {}).get("enabled", False) is not None:
        hand = dict(spec["hands"])
        hand["enabled"] = True
        hand["joints"] = [seg["j"] for f in hand["fingers"] for seg in f["segs"]]
        xml = _insert_hands(xml, hand)
        n = len(hand["joints"])   # 每手关节数(由 fingers 段链驱动——五指[样例] 8)
        eff["hands"] = True
        eff["hand"] = hand
        eff["j"] = 29 + 2 * n
    if objects:
        eff["objects"] = list(spec.get("objects", []))
    cup_skip = {"cup"} if (cup_on_hand and any(o.get("name") == "cup" for o in eff["objects"])) else set()
    if cup_skip:
        cup = next(o for o in eff["objects"] if o["name"] == "cup")
        xml = _insert_cup_hand(xml, cup)
        eff["cup_on_hand"] = True
        for o in eff["objects"]:
            if o["name"] == "cup":
                o["_skip"] = True   # 已焊手,不再落 worldbody
            if o["name"] == "water":
                o["fill_pos_m"] = [0.0, 0.08, 1.075]   # [样例] 实测 r_hand 持杯位——重置后粒子在杯内
    if eff["objects"]:
        xml = _insert_objects(xml, eff["objects"], skip=cup_skip)
    if press_wall and spec.get("press_wall", {}).get("enabled", False) is not None:
        wall = dict(spec["press_wall"])
        wall["enabled"] = True
        eff["press_wall"] = True
        xml = _insert_press_wall(xml, wall)
    if any(o.get("kind") == "needle_plate" for o in eff["objects"]):
        npo = next(o for o in eff["objects"] if o.get("kind") == "needle_plate")
        xml = _insert_needle(xml, npo)
    if terrain and spec.get("terrain"):
        xml = _insert_terrain(xml, spec["terrain"])
        eff["terrain"] = True
    os.makedirs(GEN_DIR, exist_ok=True)
    key = hashlib.sha256(xml.encode()).hexdigest()[:8]
    out = os.path.join(GEN_DIR, f"g1_train_{key}.xml")
    if not os.path.exists(out):
        header = ("<!-- 训练变体(自动生成,勿手改): 基座=body_spec.base "
                  f"({base}) + 叠加\n"
                  f"     项: free-base / mass={spec.get('total_mass_kg', '官方原值')}kg / "
                  f"hands={eff['hands']} / objects={[o['name'] for o in eff['objects']]} / "
                  f"wall={eff['press_wall']} -->\n")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(header + xml)
    eff["xml"] = out
    return out, eff


def _insert_cup_hand(xml: str, cup: dict) -> str:
    """杯焊右腕(持杯姿态)——水粒子自由体在杯内,随臂加速/倾斜产生真实洒出动力学。"""
    h = cup["half_lwh_m"]
    t = cup["wall_t"]
    cx, cy, cz = 0.0, -0.04, 0.02   # 相对右腕掌心偏移[样例]
    parts = []
    walls = [(cx, cy + h[1] + t/2, cz, h[0]+t, t, h[2]),
             (cx, cy - h[1] - t/2, cz, h[0]+t, t, h[2]),
             (cx + h[0] + t/2, cy, cz, t, h[1], h[2]),
             (cx - h[0] - t/2, cy, cz, t, h[1], h[2])]
    for i, (x, y, z, sx, sy, sz) in enumerate(walls):
        parts.append(f'      <geom name="obj_cup_w{i}" type="box" size="{sx/2:.5f} {sy/2:.5f} {sz/2:.5f}" '
                     f'mass="{cup["mass_kg"]/4:.4f}" rgba="0.8 0.8 0.85 1" contype="1" conaffinity="1"/>\n')
    s0, e0 = _body_span(xml, "r_hand")
    block = xml[s0:e0]
    inner = block[: block.rfind("</body>")]
    return xml[:s0] + inner + "\n" + "".join(parts) + "    </body>" + xml[e0:]


def generate_for_task(task: dict) -> tuple[str, dict]:
    a = task.get("assets", {}) or {}
    return generate(hands=bool(a.get("hands")), objects=bool(a.get("objects")),
                    press_wall=bool(a.get("press_wall")), terrain=bool(a.get("terrain")),
                    cup_on_hand=bool(a.get("cup_on_hand")))


def build_train_variant(spec_path: str = SPEC) -> str:
    """env_mj 无 assets 任务时的训练变体入口——基座由 body_spec.base 决定。"""
    out, _ = generate(spec_path=spec_path)
    return out


def build_fixed_base_variant(spec_path: str = SPEC) -> str:
    """Sim2Sim MuJoCo 侧(骨盆焊接)——训练变体去 freejoint,内容寻址缓存。

    对拍口径: 固定基座重力响应窗 = 无接触纯惯量段,两引擎 URDF 可转换子集。
    """
    src = build_train_variant(spec_path)
    xml = open(src, encoding="utf-8").read()
    xml = re.sub(r'\s*<freejoint name="root"/>', '', xml, count=1)
    if '<freejoint' in xml:
        raise RuntimeError("固定底座变体仍含 freejoint")
    # 去地板: 对拍判定窗 = 无接触悬空重力响应段(官方模型站高脚触地,不去即引入
    # 接触求解器差异——属引擎实现不属模型 parity)
    xml = re.sub(r'\s*<geom name="floor"[^>]*/>', '', xml, count=1)
    # 积分器对齐: 对拍 = 模型 parity(质量/惯量/几何),积分器语义不属对拍范围
    # ——两引擎统一 Euler,保证判定窗内耦合动力学可比
    xml = xml.replace('<option integrator="implicitfast"/>',
                      '<option integrator="Euler"/>', 1)
    header = ("<!-- Sim2Sim 固定底座变体(自动生成,勿手改): 源 = 训练变体去 freejoint -->\n")
    os.makedirs(GEN_DIR, exist_ok=True)
    key = hashlib.sha256(xml.encode()).hexdigest()[:8]
    out = os.path.join(GEN_DIR, f"g1_fixed_{key}.xml")
    if not os.path.exists(out):
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(header + xml)
    return out


if __name__ == "__main__":
    p, eff = generate(hands=True, objects=True, press_wall=True)
    print(f"xml={p}\nj={eff['j']} hands={eff['hands']} objects={len(eff['objects'])}")
