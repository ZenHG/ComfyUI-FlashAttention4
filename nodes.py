# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""ComfyUI node: patch a MODEL with FlashAttention 4.

The node is a thin shell. It does not build or ship a kernel; it only:

1. registers ``flash4`` in ComfyUI's attention registry (so other packs and
   scripts can select it by name), and
2. clones the incoming MODEL and installs the FA4 adapter as its optimized
   attention override.

Everything expensive (downloading / JIT-compiling the CuTe DSL kernel) happens
on the first attention call, not at import time.

Since v0.2 the node also exposes the adapter's fallback counters as a
``FlashAttention4Stats`` node, so an A/B run can prove FA4 actually ran
instead of silently degrading to SDPA.
"""

from __future__ import annotations

import logging

from comfy.ldm.modules import attention as comfy_attention

try:
    from . import attention_fa4
except ImportError:  # standalone source-tree tests
    import attention_fa4  # type: ignore[no-redef]

log = logging.getLogger(__name__)

comfy_attention.register_attention_function(
    "flash4", attention_fa4.attention_flash_4
)


class FlashAttention4ModelPatch:
    """Install FlashAttention 4 as the model's optimized attention backend."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Model to accelerate: connect after the UNETLoader / model loader "
                            "and before the sampler. The FA4 kernel loads lazily on the first "
                            "attention call."
                        )
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = (
        "Patch a MODEL to use FlashAttention 4 (CuTe DSL) for dense attention. "
        "The SM120/SM121 kernel bundle is loaded through `kernels`; masked "
        "attention, GQA, non-64/128 head dims and non-half dtypes fall back to "
        "SDPA and increment the stats node's counters."
    )

    def patch(self, model):
        patched = model.clone()
        patched.set_model_optimized_attention(attention_fa4.attention_flash_4)
        log.info("[FlashAttention4] installed flash4 on %s", type(model).__name__)
        return (patched,)


class FlashAttention4Stats:
    """Read/reset the FA4 adapter's fallback counters for A/B runs."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reset": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "True = zero the counters before sampling; False = keep counting."},
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "fa4_calls",
        "sdpa_mask",
        "sdpa_dtype",
        "sdpa_gqa",
        "sdpa_headdim",
        "sdpa_meta",
        "kernel_errors",
    )
    FUNCTION = "read"
    CATEGORY = "model_patches/attention"
    DESCRIPTION = (
        "Read the FA4 adapter's fallback counters. fa4_calls must be > 0 and "
        "all sdpa_*/kernel_errors must be 0 for a run to count as real FA4."
    )

    def read(self, reset):
        if reset:
            attention_fa4.reset_stats()
            log.info("[FlashAttention4] stats reset")
        s = attention_fa4.stats_snapshot()
        if reset:
            # The reset node executes BEFORE the sampler, so printing immediately
            # always shows zeros. Poll in a daemon thread and print as soon as a
            # counter moves, or at the deadline for an all-zero (FA4 unused) run.
            import threading

            def _deferred_snapshot():
                import time as _time
                deadline = _time.time() + 900
                while _time.time() < deadline:
                    s2 = attention_fa4.stats_snapshot()
                    log.info("[FlashAttention4] live stats %s", s2)
                    print("[FlashAttention4] live stats %s" % s2, flush=True)
                    _time.sleep(30)

            threading.Thread(target=_deferred_snapshot, daemon=True).start()
        log.info("[FlashAttention4] stats %s", s)
        print("[FlashAttention4] stats %s" % s, flush=True)
        return (
            int(s.get("fa4", 0)),
            int(s.get("sdpa_mask", 0)),
            int(s.get("sdpa_dtype", 0)),
            int(s.get("sdpa_gqa", 0)),
            int(s.get("sdpa_headdim", 0)),
            int(s.get("sdpa_meta", 0)),
            int(s.get("kernel_error", 0)),
        )


class FlashAttention4StatsLine:
    """One-line text form of the counters, for A/B logs and Preview-style nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fa4_calls": ("INT", {"default": 0}),
                "sdpa_mask": ("INT", {"default": 0}),
                "sdpa_dtype": ("INT", {"default": 0}),
                "sdpa_gqa": ("INT", {"default": 0}),
                "sdpa_headdim": ("INT", {"default": 0}),
                "sdpa_meta": ("INT", {"default": 0}),
                "kernel_errors": ("INT", {"default": 0}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("line",)
    FUNCTION = "format"
    OUTPUT_NODE = True
    CATEGORY = "model_patches/attention"
    DESCRIPTION = (
        "Format the FA4 counters as one line: `fa4=.. sdpa_mask=.. ... kernel_errors=..`. "
        "Wire the Stats node's 7 outputs here and the string shows up in the run's UI."
    )

    def format(
        self,
        fa4_calls,
        sdpa_mask,
        sdpa_dtype,
        sdpa_gqa,
        sdpa_headdim,
        sdpa_meta,
        kernel_errors,
    ):
        line = (
            "fa4={} sdpa_mask={} sdpa_dtype={} sdpa_gqa={} sdpa_headdim={} "
            "sdpa_meta={} kernel_errors={}".format(
                int(fa4_calls),
                int(sdpa_mask),
                int(sdpa_dtype),
                int(sdpa_gqa),
                int(sdpa_headdim),
                int(sdpa_meta),
                int(kernel_errors),
            )
        )
        ok = int(fa4_calls) > 0 and not any(
            (sdpa_mask, sdpa_dtype, sdpa_gqa, sdpa_headdim, sdpa_meta, kernel_errors)
        )
        line += "  => FA4 ACTIVE" if ok else "  => NOT A CLEAN FA4 RUN"
        log.info("[FlashAttention4] %s", line)
        print("[FlashAttention4] %s" % line, flush=True)
        return {"ui": {"text": (line,)}, "result": (line,)}


NODE_CLASS_MAPPINGS = {
    "FlashAttention4ModelPatch": FlashAttention4ModelPatch,
    "FlashAttention4Stats": FlashAttention4Stats,
    "FlashAttention4StatsLine": FlashAttention4StatsLine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashAttention4ModelPatch": "FlashAttention4 Model Patch",
    "FlashAttention4Stats": "FlashAttention4 Stats",
    "FlashAttention4StatsLine": "FlashAttention4 Stats Line",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
