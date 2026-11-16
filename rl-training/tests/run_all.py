#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""平台自检总入口——一条命令验证全部功能逻辑(契约/装配/生成器/环境/GPU契约/学习/导出打包/负例)。

用法:
  python tests/run_all.py          # 全套(约 2~4 分钟)
  python tests/run_all.py --fast   # 跳过学习冒烟(约 30 秒)
退出码 = 唯一裁决(0 全绿)。
"""
import argparse
import importlib
import os
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # rl-training

MODULES_FULL = ["t_contract", "t_assembler", "t_cfg_gen", "t_envs", "t_gpu_contract",
                "t_learn_smoke", "t_quality", "t_pretrain", "t_export_package", "t_negatives",
                "t_contact_scorer"]
MODULES_FAST = ["t_contract", "t_assembler", "t_cfg_gen", "t_envs", "t_gpu_contract",
                "t_quality", "t_pretrain", "t_negatives"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    tmp = os.path.join(_HERE, ".tmp")
    os.makedirs(tmp, exist_ok=True)
    fails = 0
    for m in (MODULES_FAST if args.fast else MODULES_FULL):
        t0 = time.time()
        try:
            mod = importlib.import_module(f"tests.{m}")
            mod.run(tmp)
            print(f"[PASS] {m} ({time.time() - t0:.1f}s)")
        except Exception as e:
            fails += 1
            print(f"[FAIL] {m}: {e}")
            traceback.print_exc()
    print("TESTS_PASS(平台自检全绿)" if fails == 0 else f"TESTS_FAIL({fails})")
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
