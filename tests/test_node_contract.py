# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Zero-GPU checks for node registration, contract and MODEL patch semantics.

A stub ``comfy.ldm.modules.attention`` stands in for the host, so this runs in a
plain Python environment. No model, no weights, no GPU.

Run: python tests/test_node_contract.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import install_stub_comfy, load_package

FAILED = []


def check(name, ok, detail=""):
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


class FakeModel:
    def __init__(self):
        self.clones = 0
        self.attention = []

    def clone(self):
        self.clones += 1
        return FakeModel()

    def set_model_optimized_attention(self, fn):
        self.attention.append(fn)


def main():
    attention = install_stub_comfy()
    pkg = load_package()
    nodes = sys.modules[pkg.__name__ + ".nodes"]
    fa4 = sys.modules[pkg.__name__ + ".attention_fa4"]

    print("== 1. Package exports ==")
    check("__version__ = 0.2.1", pkg.__version__ == "0.2.1", pkg.__version__)
    check("NODE_CLASS_MAPPINGS 存在", bool(nodes.NODE_CLASS_MAPPINGS))
    check("NODE_DISPLAY_NAME_MAPPINGS 存在", bool(nodes.NODE_DISPLAY_NAME_MAPPINGS))
    check(
        "注册了两个节点",
        set(nodes.NODE_CLASS_MAPPINGS) == {"FlashAttention4ModelPatch", "FlashAttention4Stats"},
        str(sorted(nodes.NODE_CLASS_MAPPINGS)),
    )

    print("== 2. flash4 registration ==")
    check("flash4 已注册", "flash4" in attention.REGISTERED_ATTENTION_FUNCTIONS)
    check(
        "flash4 -> attention_flash_4",
        attention.REGISTERED_ATTENTION_FUNCTIONS.get("flash4") is fa4.attention_flash_4,
    )

    print("== 3. Model patch node contract ==")
    cls = nodes.NODE_CLASS_MAPPINGS["FlashAttention4ModelPatch"]
    inputs = cls.INPUT_TYPES()
    check("INPUT_TYPES 有 required", "required" in inputs)
    check("required 只有 model", list(inputs["required"]) == ["model"], str(list(inputs["required"])))
    check("model 类型是 MODEL", inputs["required"]["model"][0] == "MODEL")
    check("model 有 tooltip", len(inputs["required"]["model"][1].get("tooltip", "")) >= 12)
    check("RETURN_TYPES = MODEL", cls.RETURN_TYPES == ("MODEL",), str(cls.RETURN_TYPES))
    check("RETURN_NAMES = model", cls.RETURN_NAMES == ("model",), str(cls.RETURN_NAMES))
    check("CATEGORY = model_patches/attention", cls.CATEGORY == "model_patches/attention", cls.CATEGORY)
    check("FUNCTION 存在", callable(getattr(cls, cls.FUNCTION)))

    print("== 4. MODEL patch semantics ==")
    original = FakeModel()
    result = cls().patch(original)
    check("返回一个 MODEL", isinstance(result, tuple) and len(result) == 1)
    patched = result[0]
    check("返回的是 clone，不是原对象", patched is not original)
    check("原模型未被修改", original.attention == [])
    check("clone 被调用一次", original.clones == 1, str(original.clones))
    check("clone 上安装了 flash4", patched.attention == [fa4.attention_flash_4], str(patched.attention))

    print("== 5. Stats node contract ==")
    stats_cls = nodes.NODE_CLASS_MAPPINGS["FlashAttention4Stats"]
    check("Stats RETURN_TYPES 是 7 个 INT", stats_cls.RETURN_TYPES == ("INT",) * 7, str(stats_cls.RETURN_TYPES))
    check("Stats 有 reset", "reset" in stats_cls.INPUT_TYPES()["required"])
    fa4.reset_stats()
    s = stats_cls().read(True)
    check("reset=True 归零后 fa4_calls=0", s[0] == 0, str(s))
    # simulate a dtype fallback via CPU fp32
    import torch

    fn = fa4.attention_flash_4
    fn(torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64), torch.randn(1, 2, 5, 64), 2, skip_reshape=True)
    s = stats_cls().read(False)
    check("dtype fallback 后 sdpa_dtype>0", s[2] >= 1, str(s))

    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
