# -*- coding: utf-8 -*-
"""G1 配置入口——实现唯一来源 = rltrain.g1_cfg(避免双份配置漂移)。

用法:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from rltrain.g1_cfg import G1Cfg, build_free_base_mjcf
"""
from rltrain.g1_cfg import G1Cfg, build_free_base_mjcf, MJCF_FREE, MJCF_SMOKE  # noqa: F401
