# ComfyUI-FlashAttention4

MODEL patch that routes dense attention through **FlashAttention 4** (CuTe DSL)
on NVIDIA Blackwell / RTX 50 (SM120/SM121). The SM120 kernel bundle is loaded
through `kernels`; no kernel source is rebuilt here.

- Node: `FlashAttention4 Model Patch` (`FlashAttention4ModelPatch`, MODEL → MODEL)
- Stats: `FlashAttention4 Stats` (`FlashAttention4Stats`, 7 INT outputs)
- Registry name: `flash4`

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

For A/B runs, add `FlashAttention4 Stats` with `reset=True` before the run,
then read the outputs. A valid FA4 run has `fa4 > 0` and all `sdpa_*` /
`kernel_errors` = 0. If `kernel_errors` increments, the log line
`FlashAttention 4 failed, using SDPA` names the cause.

## What falls back to SDPA

- `mask` is not None (FA4 has no attn-mask path)
- dtype is not fp16/bf16, or tensor is on CPU/meta
- `head_dim` not in (64, 128)
- `enable_gqa=True` with `kv_heads != q_heads`

Each case increments a STATS counter, so silent degradation is visible.

## Supported platforms

| Platform | Status |
|---|---|
| Windows x86 + SM120/SM121 | Tested (RTX 5070, torch 2.13+cu130, Python 3.12) |
| Linux x86 + SM80/SM90/SM100/SM120 | Expected to work via the same `kernels` bundle; not CI-tested |
| CPU-only / no GPU | Imports and runs; attention falls back to SDPA |
| AMD ROCm / Apple MPS / Windows-ARM | FA4 has no kernel; attention falls back to SDPA |

## Benchmarks

Workload: 640×832, 124 frames, 10 steps, `er_sde`, LoRA×2, RTX 5070.
Times are ComfyUI `Prompt executed` seconds.

| Backend | Time | Note |
|---|---:|---|
| Kitchen INT8 + Spark | 77.4–141.8s | sparse + quantized dense |
| Kitchen INT8 only | 86.9s | quantized dense |
| FA4 only | 141.1–177.9s | dense FA4, cold JIT included |
| FA4 + Spark | 85.6s | Spark still routes dense through Kitchen |

FA4 is a dense kernel: on workloads where a quantized or sparse backend is
faster, FA4 can be slower. Benchmark before swapping a production chain.

## License

MIT for this node pack. The FA4 kernel bundle is BSD-3-Clause (Dao-AILab).