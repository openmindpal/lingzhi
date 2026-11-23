# -*- coding: utf-8 -*-
"""G1 本体配置——与本体包逐项同源(单一来源 = unitree-g1/body-package.toml)。

关节序/限位/KP·KD/中立位全部解析自本体包,训练环境不得另立数值;
MJCF 训练变体 = 冒烟模型(biped_29dof.xml,关节名/序/限位与包同源) + 自由底座
(平衡/行走训练需要可倒地;冒烟模型骨盆固定,见该文件头诚实边界)。
"""
from __future__ import annotations

import os
import re
import sys

# 直跑(self-test)时的包路径引导
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rltrain._compat import tomllib  # noqa: E402  — 唯一兼容点(rltrain/_compat)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# 本体包契约样例:首选 body-sdk 对接套件(交付面),回退 body-packages(历史位置)
_PKG_CANDIDATES = [
    os.path.join(_REPO, "body-sdk", "examples", "unitree-g1", "body-package.toml"),
    os.path.join(_REPO, "apps", "body-packages", "unitree-g1", "body-package.toml"),
]
PKG_TOML = next((p for p in _PKG_CANDIDATES if os.path.exists(p)), _PKG_CANDIDATES[0])
MJCF_SMOKE = os.path.join(_REPO, "simulation", "assets", "models", "biped_29dof.xml")
MJCF_FREE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "envs", "g1", "assets", "g1_train_free.xml")


class G1Cfg:
    def __init__(self, pkg_toml: str = PKG_TOML):
        with open(pkg_toml, "rb") as f:
            pkg = tomllib.load(f)
        self.pkg_path = pkg_toml
        self.joints: list[str] = pkg["protocol"]["joint_mapping"]["ordered"]
        self.j = len(self.joints)
        self.state_order: list[str] = pkg["profile"]["state_order"]
        dyn = pkg["profile"]["dynamics"]
        self.kp: float = float(dyn["kp"])
        self.kd: float = float(dyn["kd"])
        self.max_velocity: float = float(dyn["max_velocity_rad_s"])
        self.control_hz: float = float(pkg["protocol"]["frequency_hz"])
        self.limits: dict[str, tuple[float, float]] = {
            name: (float(v["limits"]["min"]), float(v["limits"]["max"]))
            for name, v in pkg["profile"]["joints"].items()
        }
        self.neutral: dict[str, float] = {k: float(v) for k, v in pkg["calibration"]["neutral"].items()}
        self.tool_sites = {"l": "l_tool", "r": "r_tool"}   # 末端 site(MJCF 内同名)
        self._validate()

    # ── 校验(装载校验镜像) ─────────────────────────────────
    def _validate(self) -> None:
        from rltrain.contract import neutral_in_soft_limits
        if self.j != 29:
            raise ValueError(f"G1 关节数应为 29,包声明 {self.j}")
        missing = [n for n in self.joints if n not in self.limits]
        if missing:
            raise ValueError(f"关节缺限位声明: {missing}")
        bad = neutral_in_soft_limits(self.neutral, self.limits)
        if bad:
            raise ValueError(f"中立位越出软限位(装载校验必拒): {bad}")

    def limit_arrays(self):
        """按 joint_mapping 序输出 (lo, hi) 数组(rad)。"""
        lo = [self.limits[n][0] for n in self.joints]
        hi = [self.limits[n][1] for n in self.joints]
        return lo, hi

    def neutral_array(self) -> list[float]:
        return [self.neutral.get(n, 0.0) for n in self.joints]


def build_free_base_mjcf(out_path: str = MJCF_FREE, src: str = MJCF_SMOKE) -> str:
    """训练变体:冒烟 MJCF + 骨盆自由底座(freejoint)。

    冒烟模型骨盆固定(见其文件头诚实边界)——平衡/行走训练需要可支撑/可倒地。
    本变换只插一行 <freejoint/>,其余(关节名/序/限位/传感)原样继承。
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        return out_path
    raw = open(src, encoding="utf-8").read()
    pat = re.compile(r'(<body name="pelvis"[^>]*>)')
    if not pat.search(raw):
        raise RuntimeError("MJCF 缺 pelvis body——自由底座变体无法生成")
    if "<freejoint" in raw:
        raise RuntimeError("源 MJCF 已含自由底座")
    variant = pat.sub(r'\1\n    <freejoint name="root"/>', raw, count=1)
    header = ("<!-- 训练变体(自动生成,勿手改): 源 = simulation/assets/models/biped_29dof.xml\n"
              "     变换 = pelvis 加 freejoint(平衡/行走训练需自由底座);重生成: 删除本文件再跑训练 -->\n")
    variant = variant.replace("<mujoco model=\"biped_29dof\">",
                              "<mujoco model=\"biped_29dof_free\">", 1)
    variant = header + variant
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(variant)
    return out_path


if __name__ == "__main__":
    cfg = G1Cfg()
    print(f"joints={cfg.j} kp={cfg.kp} kd={cfg.kd} hz={cfg.control_hz}")
    print("neutral OK(全部在软限位内)")
    print("free-base variant:", build_free_base_mjcf())
