# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Shared loading helpers for the zero-GPU checks.

The repository directory is ``ComfyUI-FlashAttention4`` (hyphens), so it cannot
be imported by name. Load it as a synthetic package with ``__path__`` pointing
at the repo root, the same way MOSS-Storyboard's tests do.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_file(module_name: str, filename: str):
    path = os.path.join(ROOT, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_package(package_name: str = "flash_attention4_under_test"):
    if package_name in sys.modules:
        return sys.modules[package_name]
    path = os.path.join(ROOT, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        package_name, path, submodule_search_locations=[ROOT]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module


def install_stub_comfy():
    """Install just enough of ``comfy.ldm.modules.attention`` for nodes.py."""
    if "comfy.ldm.modules.attention" in sys.modules:
        return sys.modules["comfy.ldm.modules.attention"]

    comfy = types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    modules = types.ModuleType("comfy.ldm.modules")
    attention = types.ModuleType("comfy.ldm.modules.attention")

    registry = {}

    def register_attention_function(name, func):
        if name not in registry:
            registry[name] = func

    def get_attention_function(name, default=...):
        if name in registry:
            return registry[name]
        if default is ...:
            raise KeyError(name)
        return default

    attention.REGISTERED_ATTENTION_FUNCTIONS = registry
    attention.register_attention_function = register_attention_function
    attention.get_attention_function = get_attention_function

    comfy.ldm = ldm
    ldm.modules = modules
    modules.attention = attention

    sys.modules["comfy"] = comfy
    sys.modules["comfy.ldm"] = ldm
    sys.modules["comfy.ldm.modules"] = modules
    sys.modules["comfy.ldm.modules.attention"] = attention
    return attention
