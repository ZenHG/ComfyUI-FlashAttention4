# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Zero-GPU checks for the FA4 attention adapter.

No ComfyUI import, no CUDA, no FA4 kernel. Every case runs on CPU tensors and
exercises the SDPA fallback / reshape paths that must stay correct even when
FA4 is unavailable.

Run: python tests/test_adapter.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _load import load_file

FAILED = []


def check(name, ok, detail=""):
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


def main():
    fa4 = load_file("flash_attention4_adapter_under_test", "attention_fa4.py")
    fn = fa4.attention_flash_4

    print("== 1. skip_reshape=True (MiniMax-H3 path) ==")
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    v = torch.randn(2, 3, 5, 8)
    mask = torch.zeros(2, 1, 5, 5)
    out = fn(q, k, v, 3, mask=mask, skip_reshape=True)
    check("输出 [B, L, H*D]", tuple(out.shape) == (2, 5, 24), str(tuple(out.shape)))

    out = fn(q, k, v, 3, mask=mask, skip_reshape=True, skip_output_reshape=True)
    check("skip_output_reshape 保持 [B, H, L, D]", tuple(out.shape) == (2, 3, 5, 8), str(tuple(out.shape)))

    print("== 2. skip_reshape=False (packed QKV path) ==")
    packed = torch.randn(2, 5, 24)
    out = fn(packed, packed, packed, 3, mask=mask)
    check("输出 [B, L, H*D]", tuple(out.shape) == (2, 5, 24), str(tuple(out.shape)))

    print("== 3. GQA (fewer K/V heads) ==")
    q = torch.randn(1, 4, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    gqa_mask = torch.zeros(1, 1, 5, 5)
    out = fn(q, k, v, 4, mask=gqa_mask, skip_reshape=True)
    check("输出 [B, L, Hq*D]", tuple(out.shape) == (1, 5, 32), str(tuple(out.shape)))

    print("== 4. CPU fp32 without mask still returns the right shape ==")
    q = torch.randn(1, 2, 7, 64)
    out = fn(q, q, q, 2, skip_reshape=True)
    check("输出 [B, L, H*D]", tuple(out.shape) == (1, 7, 128), str(tuple(out.shape)))

    print("== 5. Module surface ==")
    check("KERNEL_REPO 是 SM120 仓库", fa4.KERNEL_REPO == "SecondNatureComputing/flash-attn-4-sm120", fa4.KERNEL_REPO)
    check("__all__ 导出 attention_flash_4", "attention_flash_4" in fa4.__all__)
    check("container_function 存在", callable(getattr(fn, "container_function", None)))
    check("STATS 计数器存在", isinstance(fa4.STATS, dict) and "fa4" in fa4.STATS)

    print("== 6. v0.2 dispatch: STATS + 4D mask + head_dim ==")
    fa4.reset_stats()
    # CPU fp32 -> sdpa_dtype
    q = torch.randn(1, 2, 7, 64)
    fn(q, q, q, 2, skip_reshape=True)
    s = fa4.stats_snapshot()
    check("CPU fp32 计入 sdpa_dtype", s["sdpa_dtype"] >= 1, str(s))
    # mask 4D -> sdpa_mask
    fa4.reset_stats()
    q = torch.randn(1, 2, 5, 8)
    m = torch.zeros(1, 1, 5, 5)
    fn(q, q, q, 2, mask=m, skip_reshape=True)
    s = fa4.stats_snapshot()
    check("mask 计入 sdpa_mask", s["sdpa_mask"] >= 1, str(s))
    # head_dim=32 -> sdpa_headdim (dtype is CPU fp32, but headdim is checked first)
    fa4.reset_stats()
    q = torch.randn(1, 2, 5, 32)
    # use CUDA-less path but dim_head check comes before dtype check
    fn(q, q, q, 2, skip_reshape=True)
    s = fa4.stats_snapshot()
    check("head_dim=32 计入 sdpa_headdim", s["sdpa_headdim"] >= 1, str(s))

    print("== 7. Kernel import (no GPU call) ==")
    try:
        kernel_fn = fa4._load_flash_attn_4()
    except Exception as error:
        # A machine without the FA4 Python stack is still a valid zero-GPU
        # target; report SKIP rather than pretending the import passed.
        print("  [SKIP] FA4 import unavailable: %s" % error)
    else:
        check("flash_attn_func 可 import", callable(kernel_fn))

    print("== 8. container_function unwraps AttentionTensorContainer ==")
    qc = torch.randn(1, 2, 5, 64)
    kc = torch.randn(1, 2, 5, 64)
    vc = torch.randn(1, 2, 5, 64)
    class C:
        def __init__(self, t): self.t=t
        def take(self): return self.t
    out = fn.container_function(C(qc), C(kc), C(vc), 2, skip_reshape=True)
    check("container 路径输出 [B,L,H*D]", tuple(out.shape)==(1,5,128), str(tuple(out.shape)))

    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
