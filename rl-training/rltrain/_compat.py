# -*- coding: utf-8 -*-
"""Python 版本兼容层——rl-training 内唯一的 tomllib 兼容点。

背景: Python 3.10 无标准库 tomllib(3.11+ 才有),tomli 为等价 backport。
调用方统一 `from rltrain._compat import tomllib`;
平台最低版本升至 3.11+ 时删除本模块,调用方恢复 import tomllib。
(注意: body-sdk 为独立交付套件,自带内联 fallback,不做跨套件依赖——设计边界)
"""
try:
    import tomllib          # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib

__all__ = ["tomllib"]
