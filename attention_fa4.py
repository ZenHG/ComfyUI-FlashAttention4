# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""FlashAttention 4 adapter for ComfyUI's attention function signature.

Two ways to get the FA4 kernel, tried in order:

1. ``flash_attn.cute.flash_attn_func`` -- Dao-AILab's upstream FA4 package
   (CuTe DSL). This is what ComfyUI-flash-attention-3 imports.
2. ``kernels.get_kernel("SecondNatureComputing/flash-attn-4-sm120")`` -- the
   SM120 / SM121 consumer-Blackwell bundle used on RTX 50 series.

The kernel is loaded lazily on the first attention call, so importing this
module (and the node pack) never needs a GPU, a CUDA toolkit, or a network.

Tensor layout is ComfyUI's: ``q`` is ``[B, H, L, D]`` when ``skip_reshape`` is
true (MiniMax-H3 passes this), otherwise ``[B, L, H * D]``. FA4 wants
``[B, L, H, D]``, so the call path transposes once before the kernel and once
after it.

Masked attention, unsupported dtypes/devices, non-64/128 head dims and GQA
where the key/value head count differs from the query head count all fall back
to SDPA with the same output layout. Every fallback increments ``STATS`` so an
A/B run can prove FA4 actually ran instead of silently degrading to SDPA.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import types

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

KERNEL_REPO = "SecondNatureComputing/flash-attn-4-sm120"

# A/B guard: read these after a run. ``fa4`` must be > 0 and ``sdpa_*`` /
# ``kernel_error`` must be 0 for the run to count as a real FA4 measurement.
STATS = {
    "fa4": 0,
    "sdpa_mask": 0,
    "sdpa_dtype": 0,
    "sdpa_gqa": 0,
    "sdpa_headdim": 0,
    "sdpa_meta": 0,
    "kernel_error": 0,
    # "none" | "upstream" (flash_attn.cute) | "kernels" (SecondNatureComputing bundle)
    "fa4_source": "none",
}

_flash_attn_4_func = None
_load_error: Exception | None = None


def reset_stats():
    """Zero the fallback counters (call before a timed A/B run)."""
    for key in STATS:
        STATS[key] = "none" if key == "fa4_source" else 0


def stats_snapshot():
    return dict(STATS)


def _prepare_kernel_import():
    """Apply import-time compatibility shims for the FA4 CuTe DSL bundle.

    The SM120 bundle is Linux-first. Three things break `import` on Windows /
    cutlass 4.8 before any GPU code runs:

    * its ``cache_utils`` and ``quack`` import ``fcntl`` (Unix-only);
    * ``quack/__init__.py`` pulls in RMSNorm / softmax kernels we do not need;
    * cutlass 4.4+ moved ``ThrMma`` from ``cutlass.cute.core`` to
      ``cutlass.cute.atom``.

    None of these touch the kernel source or trigger a build. They only make
    the pure-Python package importable; the actual CuTe JIT still happens on
    the first call, on the GPU.
    """
    if sys.platform == "win32":
        # The persistent-cache path uses fcntl.flock. Keep the in-memory cache.
        os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"

        try:
            import fcntl  # noqa: F401
        except Exception:
            fcntl_stub = types.ModuleType("fcntl")
            fcntl_stub.LOCK_SH = 1
            fcntl_stub.LOCK_EX = 2
            fcntl_stub.LOCK_NB = 4
            fcntl_stub.LOCK_UN = 8

            def _flock(_fd, _operation):
                raise RuntimeError(
                    "fcntl.flock is unavailable on Windows; "
                    "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED must stay 0"
                )

            fcntl_stub.flock = _flock
            sys.modules["fcntl"] = fcntl_stub

        if "quack" not in sys.modules:
            quack_spec = importlib.util.find_spec("quack")
            if quack_spec is not None and quack_spec.submodule_search_locations:
                quack_shell = types.ModuleType("quack")
                quack_shell.__path__ = list(quack_spec.submodule_search_locations)
                quack_shell.__version__ = "0"
                sys.modules["quack"] = quack_shell

    try:
        import cutlass.cute as cute
        import cutlass.cute.core as cute_core

        if not hasattr(cute_core, "ThrMma"):
            from cutlass.cute.atom import ThrMma

            cute_core.ThrMma = ThrMma
            log.debug("Patched cutlass.cute.core.ThrMma for cutlass >= 4.4")

        # cutlass 4.8 renamed make_fragment(shape, dtype) to
        # make_rmem_tensor(layout_or_shape, dtype).
        if not hasattr(cute, "make_fragment"):
            cute.make_fragment = cute.make_rmem_tensor
            log.debug("Patched cutlass.cute.make_fragment for cutlass >= 4.8")
    except Exception as error:
        log.debug("cutlass compatibility shim skipped: %s", error)


def _load_flash_attn_4():
    """Return FA4's ``flash_attn_func``, loading the kernel on first use."""
    global _flash_attn_4_func, _load_error
    if _flash_attn_4_func is not None:
        return _flash_attn_4_func
    if _load_error is not None:
        raise RuntimeError(
            "FlashAttention 4 is unavailable; the first load already failed"
        ) from _load_error

    _prepare_kernel_import()

    forced = os.environ.get("FLASH_ATTENTION4_SOURCE", "auto").strip().lower()
    if forced in ("upstream", "kernels"):
        log.info("[FlashAttention4] FLASH_ATTENTION4_SOURCE=%s (forced)", forced)

    try:
        if forced == "kernels":
            raise ImportError("forced to kernels by FLASH_ATTENTION4_SOURCE")
        from flash_attn.cute import flash_attn_func as fa4_func

        STATS["fa4_source"] = "upstream"
    except Exception as upstream_error:
        log.info(
            "flash_attn.cute is unavailable (%s); trying the SM120 kernel bundle",
            upstream_error,
        )
        try:
            from kernels import get_kernel

            flash_attn_4 = get_kernel(KERNEL_REPO)
            fa4_func = flash_attn_4.flash_attn_func
            STATS["fa4_source"] = "kernels"
        except Exception as bundle_error:
            _load_error = bundle_error
            missing = getattr(bundle_error, "name", None)
            hint = f" Missing module: {missing}." if missing else ""
            raise RuntimeError(
                "FlashAttention 4 is unavailable. Install Dao-AILab's FA4 "
                "(`pip install flash-attn-4`) or let `kernels` download "
                f"`{KERNEL_REPO}` (needs `pip install kernels`).{hint} "
                "The SM120 CuTe DSL bundle also needs cuda-python, "
                "nvidia-cutlass-dsl[cu13], quack-kernels and apache-tvm-ffi."
            ) from bundle_error

    _flash_attn_4_func = fa4_func
    log.info("FlashAttention 4 kernel loaded")
    return _flash_attn_4_func


def _heads_from_dim(tensor: torch.Tensor, dim_head: int, name: str) -> int:
    if tensor.ndim == 4:
        return tensor.shape[-2]
    if tensor.ndim == 3:
        return tensor.shape[-1] // dim_head
    raise ValueError(f"{name} must be 3D or 4D, got {tuple(tensor.shape)}")


def _reshape_qkv_to_heads(q, k, v, b, heads, dim_head, enable_gqa=False):
    """ComfyUI's reshape, inlined so this module has no host import at load time."""
    q = q.view(b, -1, heads, dim_head)
    if enable_gqa:
        key_heads = _heads_from_dim(k, dim_head, "Key")
        value_heads = _heads_from_dim(v, dim_head, "Value")
    else:
        key_heads = heads
        value_heads = heads
    k = k.view(b, -1, key_heads, dim_head)
    v = v.view(b, -1, value_heads, dim_head)
    return q, k, v


def _sdpa_fallback(q, k, v, b, heads, dim_head, mask, skip_output_reshape, enable_gqa=False, causal=False):
    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=causal,
        enable_gqa=enable_gqa or k.shape[-3] != q.shape[-3],
    )
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


def attention_flash_4(
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    """ComfyUI attention function backed by FlashAttention 4.

    Signature matches ``comfy.ldm.modules.attention.attention_pytorch`` so the
    function can be registered as ``flash4`` or installed on a model patch with
    ``ModelPatcher.set_model_optimized_attention``.
    """
    causal = bool(kwargs.get("causal", False))
    enable_gqa = bool(kwargs.get("enable_gqa", False))

    if attn_precision is not None and str(attn_precision) not in ("fp16", "bf16", "half", "bfloat16"):
        # FA4 only runs half precision; do not silently ignore an fp32 request.
        STATS["sdpa_dtype"] += 1
        # fall through to dtype check below, which will also catch fp32 tensors

    if skip_reshape:
        if q.ndim != 4:
            raise ValueError(f"skip_reshape=True expects 4D q, got {tuple(q.shape)}")
        b, _, _, dim_head = q.shape
    else:
        if q.ndim != 3:
            raise ValueError(f"skip_reshape=False expects 3D q, got {tuple(q.shape)}")
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = _reshape_qkv_to_heads(q, k, v, b, heads, dim_head, enable_gqa)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

    if mask is not None:
        # Same broadcast shape ComfyUI's SDPA backend uses.
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4:
            STATS["sdpa_mask"] += 1
            return _sdpa_fallback(q, k, v, b, heads, dim_head, mask, skip_output_reshape, enable_gqa, causal)
        STATS["sdpa_mask"] += 1
        return _sdpa_fallback(q, k, v, b, heads, dim_head, mask, skip_output_reshape, enable_gqa, causal)

    if q.is_meta or k.is_meta or v.is_meta:
        STATS["sdpa_meta"] += 1
        return _sdpa_fallback(q, k, v, b, heads, dim_head, None, skip_output_reshape, enable_gqa, causal)

    kv_heads = k.shape[-3] if k.ndim >= 3 else heads
    if enable_gqa and kv_heads != heads:
        STATS["sdpa_gqa"] += 1
        return _sdpa_fallback(q, k, v, b, heads, dim_head, None, skip_output_reshape, enable_gqa, causal)

    if dim_head not in (64, 128):
        STATS["sdpa_headdim"] += 1
        return _sdpa_fallback(q, k, v, b, heads, dim_head, None, skip_output_reshape, enable_gqa, causal)

    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        STATS["sdpa_dtype"] += 1
        return _sdpa_fallback(q, k, v, b, heads, dim_head, None, skip_output_reshape, enable_gqa, causal)

    try:
        flash_attn_func = _load_flash_attn_4()
        out = flash_attn_func(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            causal=causal,
        )[0].transpose(1, 2)
        STATS["fa4"] += 1
    except Exception as error:
        STATS["kernel_error"] += 1
        log.warning("FlashAttention 4 failed, using SDPA: %s", error)
        return _sdpa_fallback(q, k, v, b, heads, dim_head, None, skip_output_reshape, enable_gqa, causal)

    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


# ComfyUI's wrap_attn passes AttentionTensorContainer instances here when the
# override is composed (e.g. Spark dense_backend). Unwrap with .take() before
# touching .ndim/.shape; a bare passthrough would hit
# "'AttentionTensorContainer' object has no attribute 'ndim'".
def _unwrap_container(x):
    take = getattr(x, "take", None)
    if callable(take):
        return take()
    return x


def _flash_attn_4_container(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    return attention_flash_4(
        _unwrap_container(q),
        _unwrap_container(k),
        _unwrap_container(v),
        heads,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        **kwargs,
    )


attention_flash_4.container_function = _flash_attn_4_container

__all__ = [
    "KERNEL_REPO",
    "STATS",
    "attention_flash_4",
    "reset_stats",
    "stats_snapshot",
]
