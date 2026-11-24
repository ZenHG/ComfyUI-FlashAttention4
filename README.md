# ComfyUI-FlashAttention4

MODEL patch that routes dense attention through **FlashAttention 4** (CuTe DSL)
on NVIDIA GPUs from Ampere onward (SM80 / SM89 / SM90 / SM100 / SM110 /
SM120). The SM120/SM121 kernel bundle is loaded through `kernels`; no kernel
source is rebuilt here.

- Node: `FlashAttention4 Model Patch` (`FlashAttention4ModelPatch`, MODEL → MODEL)
- Stats: `FlashAttention4 Stats` (`FlashAttention4Stats`, 7 INT outputs)
- Stats line: `FlashAttention4 Stats Line` (`FlashAttention4StatsLine`, STRING, marks clean runs)
- Registry name: `flash4`
- Version: **0.2.2**

## Install

1. `git clone` this repo into `ComfyUI/custom_nodes/ComfyUI-FlashAttention4`.
2. Install the runtime deps into the Python that runs ComfyUI:
   `kernels>=0.9,<0.14` (tested with `0.12.0`), `huggingface-hub==0.36.2`,
   `cuda-python`, `nvidia-cutlass-dsl[cu13]==4.8.1`, `quack-kernels`,
   `apache-tvm-ffi`.
3. Restart ComfyUI.

Do **not** `pip install -U kernels` to 0.14+: it pulls `huggingface-hub` 2.x
and breaks ComfyUI's `diffusers` / `transformers`. Pin `0.12.0` + `hub 0.36.2`.

The kernel bundle is **not shipped**; `kernels` downloads
`SecondNatureComputing/flash-attn-4-sm120` on first use (BSD-3-Clause,
Dao-AILab). This node pack itself is MIT.

Windows note: the SM120 bundle imports `fcntl` and `quack.cache`; the adapter
stubs `fcntl` and disables the persistent CuTe cache
(`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0`). The price is a one-time JIT on
the first attention call after each ComfyUI restart.

## Usage

Insert `FlashAttention4 Model Patch` in the MODEL chain, after the loader / LoRA /
sparse-attention pack and before the sampler. It clones the model and installs
the FA4 adapter as the optimized attention override.

### Proving FA4 actually ran

FA4 is optional: on any unsupported input it falls back to SDPA. Add
`FlashAttention4 Stats` with `reset=True` before the run, then read the
outputs. A valid FA4 run has:

- `fa4_calls > 0`
- all `sdpa_mask` / `sdpa_dtype` / `sdpa_gqa` / `sdpa_headdim` / `sdpa_meta` = 0
- `kernel_errors` = 0

The stats node also prints to the host stdout every 30 s
(`[FlashAttention4] live stats {...}`), because a reset node runs *before* the
sampler and would otherwise only ever report zeros. If `kernel_errors`
increments, the log line `FlashAttention 4 failed, using SDPA` names the cause.

`FlashAttention4 Stats Line` turns the seven counters into one readable string
(`fa4=15 sdpa_mask=0 ... kernel_errors=0  => FA4 ACTIVE`) and marks any run that
degraded to SDPA with `NOT A CLEAN FA4 RUN`.

## What falls back to SDPA

- `mask` is not None (FA4 has no attn-mask path)
- dtype is not fp16/bf16, or tensor is on CPU/meta
- `head_dim` not in (64, 128)
- `enable_gqa=True` with `kv_heads != q_heads`

Each case increments a STATS counter, so silent degradation is visible.

## Supported platforms

FA4 (CuTe DSL) supports **SM80 / SM89 / SM90 / SM100 / SM110 / SM120** — the
upstream README says "optimized for Hopper and Blackwell", and this bundle's
`interface.py` asserts `arch // 10 in [8, 9, 10, 11, 12]`. The bundle name
`flash-attn-4-sm120` refers to the *targeted PRs* it carries, not a hard limit.

| Platform | Status |
|---|---|
| Windows x86 + SM120/SM121 (RTX 50) | **Tested**: RTX 5070, torch 2.13+cu130, Python 3.12 |
| Linux/Windows x86 + SM80/SM89 (Ampere/Ada) | Expected to work (SM80 kernel present); not tested here |
| Linux/Windows x86 + SM90 (Hopper) | Expected to work (SM90 kernel present); not tested here |
| Linux/Windows x86 + SM100/SM110 (Blackwell DC) | Expected to work (SM100 kernel present); not tested here |
| CPU-only / no GPU | Imports and runs; attention falls back to SDPA |
| AMD ROCm / Apple MPS / Windows-ARM | FA4 has no kernel; attention falls back to SDPA |

This node pack adds **no** arch check of its own: it calls `flash_attn_func`
and lets the bundle dispatch. If the bundle has no kernel for your card, the
call raises and the adapter falls back to SDPA (counted in `kernel_errors`).

FA4 is a **dense** kernel. On workloads where a quantized (Kitchen INT8, Sage)
or sparse (Spark) backend is faster, FA4 can be slower. Benchmark before
swapping a production chain.

## Interaction with sparse-attention packs

If a sparse pack (e.g. Spark-H3) is in the chain, it installs its own
per-block attention patch and **FA4 will not see those calls**. In our tests
Spark kept reporting `comfy_kitchen_calls` with FA4 installed, and the FA4
counter stayed at 0. Set the sparse pack's `dense_backend` explicitly if you
want FA4 to serve its dense path — `inherit` does **not** mean FA4.

## Benchmarks

Workload: 640×832, 124 frames, 6 steps, `er_sde`, LoRA×2, RTX 5070.
Times are ComfyUI `Prompt executed` seconds; cold = first run after a restart.

| Backend | Cold | Hot | Note |
|---|---:|---:|---|
| Kitchen INT8 + Spark | 121.7s | 67.1s | sparse + quantized dense |
| Kitchen INT8 only | 71.7s | 67.9s | quantized dense |
| FA4 only | 126.5s | **116.1s** | dense FA4, 1.7× slower than Kitchen |
| FA4 + Spark | 68.4s | 65.8s | FA4 unused (Spark routes dense to Kitchen) |

### Step count matters more than the backend

Same graph (Kitchen, LoRA×2, no Spark), only the sigma table changed:

| Steps | Time |
|---:|---:|
| 3 | **39.9s** |
| 6 | 67.9s |
| 8 | 94.5s |

≈ 11 s/step including a fixed ~10 s for VAE decode + save.

### Quality snapshot (same 640×832×124, seed 12345)

| Run | Body sharpness | High-freq ratio | Temporal SSIM | Seam/body |
|---|---:|---:|---:|---:|
| 3 steps | 31.5 | 0.9719 | 0.9946 | 0.893 |
| 6 steps | 39.7 | 0.9821 | 0.9784 | 0.715 |
| FA4 6 steps | 41.7 | 0.9818 | 0.9765 | 0.712 |

3 steps is ~21% softer than 6 steps but has the steadiest frames; 8 steps adds
no measurable detail over 6.

## License

MIT for this node pack. The FA4 kernel bundle is BSD-3-Clause (Dao-AILab).