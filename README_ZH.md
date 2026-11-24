# ComfyUI-FlashAttention4（中文说明）

> 英文版是规范源：[`README.md`](README.md)。本文件是中文对照，内容以英文版为准。

![FlashAttention4 Model Patch](banner.png)

**一个把 dense attention 路由到 FlashAttention 4（CuTe DSL）的 ComfyUI 节点。**
本包不编译 kernel 源码——SM120/Blackwell 的 kernel bundle 由
[`kernels`](https://huggingface.co/SecondNatureComputing/flash-attn-4-sm120) 在首次使用时下载。

> 🔴 **装之前先看这段**：FA4 是 **dense** attention kernel。在 MiniMax-H3 上
> 640×832 / 124 帧 / 6 步的实测里，**Kitchen INT8 更快**（热态 67.9s vs 116.1s）。
> FA4 接近无损，但**不是「比量化快」**。先在自己的负载上量一量——[数据在下面](#5--基准)。

**什么时候选 FA4**：想要**接近无损的 dense 后端**时——比如模型没有量化 INT8 后端、
画质优先的渲染、或稀疏包允许你指定 dense 路径。如果你的模型已经有量化后端
（Kitchen INT8、Sage），那通常更快——先量再换。

| 项 | 值 |
|---|---|
| 版本 | **0.2.4** |
| 许可 | **MIT**（[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md)；kernel bundle 是 BSD-3-Clause） |
| 宿主 | **ComfyUI ≥ 0.38.0** |
| GPU | NVIDIA **SM80+**（Ampere / Ada / Hopper / Blackwell）。**只在 SM120（RTX 5070）实测过** |
| 依赖 | `kernels`、`cuda-python`、`nvidia-cutlass-dsl[cu13]`、`quack-kernels`、`apache-tvm-ffi` |
| Registry | [`comfyui-fa4`](https://registry.comfy.org/nodes/comfyui-fa4) |

**本包保证**：节点能加载、能 patch MODEL、要么跑 FA4 要么**出声地**回退 SDPA（带计数器）。
**本包不保证**：FA4 比你现在用的后端更快。

---

## 1 · 快速开始

### 1.1 安装

| 方式 | 怎么做 |
|---|---|
| **Manager**（推荐） | 搜 **FlashAttention4** → Install → 重启 ComfyUI |
| **手动** | `cd <ComfyUI>/custom_nodes` → `git clone https://github.com/ZenHG/ComfyUI-FlashAttention4` |
| **Registry** | <https://registry.comfy.org/nodes/comfyui-fa4> |

Manager 会自动装 `requirements.txt`。手动安装只需一条命令，
**装到跑 ComfyUI 的那个 Python**：

```bash
<Python> -m pip install -r requirements.txt
```

`requirements.txt` 里是 CuTe DSL 运行栈（`kernels`、`cuda-python`、
`nvidia-cutlass-dsl[cu13]`、`quack-kernels`、`apache-tvm-ffi`）。然后适配器：

1. 先试 `flash_attn.cute`（上游 `flash-attn-4`，4.0.0b40 起自带 SM120）；
2. 失败回退 `kernels.get_kernel("SecondNatureComputing/flash-attn-4-sm120")`。

不用改配置。要强制走某一路，设 `FLASH_ATTENTION4_SOURCE=upstream` 或 `=kernels`。

🔴 **不要 `pip install -U kernels` 到 0.14+**。它会拉 `huggingface-hub` 2.x，
搞坏 ComfyUI 的 `diffusers` / `transformers`。`requirements.txt` 已钉 `>=0.9,<0.14`。

🔴 **装对 Python**。ComfyUI 跑在哪个解释器（启动日志里的 `ComfyUI Path`），
就装到那个；装错了节点能加载，但第一次 attention 调用会失败。

### 1.2 成功长什么样

节点搜索里应该能看到**三个**节点：

```
FlashAttention4ModelPatch    （FlashAttention4 Model Patch，   MODEL → MODEL）
FlashAttention4Stats         （FlashAttention4 Stats，          7 个 INT 输出）
FlashAttention4StatsLine     （FlashAttention4 Stats Line，     STRING）
```

**第一次 attention 调用**时，日志应该出现：

```
[FlashAttention4] installed flash4 on ModelPatcherDynamic
flash_attn.cute is unavailable (No module named 'flash_attn'); trying the SM120 kernel bundle
FlashAttention 4 kernel loaded
```

看到 `flash_attn.cute is unavailable` 是**正常的**——那条路只给源码编译用。
关键是下一行必须是 `FlashAttention 4 kernel loaded`。如果看到的是
`FlashAttention 4 failed, using SDPA`，去看 [§3.3](#33-打了-using-sdpa)。

### 1.3 第一次跑

1. 把 **FlashAttention4 Model Patch** 插进 MODEL 链——在 loader / LoRA /
   稀疏注意力包之后、采样器之前。
2. （可选）在采样器之前放一个 **FlashAttention4 Stats**，`reset=True`。
3. 跑。第一次 attention 调用会 JIT 编译 kernel；Windows 上**每次重启 ComfyUI 后**
   这一次会多花约 30 秒。

### 1.4 怎么确认 FA4 真的跑了

FA4 在不支持的输入上会回退 SDPA，所以「没报错」不等于「FA4 跑了」。看计数器：

| 输出 | 通过 |
|---|---|
| `fa4` | **> 0** |
| `sdpa_mask` / `sdpa_dtype` / `sdpa_gqa` / `sdpa_headdim` / `sdpa_meta` | **全为 0** |
| `kernel_errors` | **0** |

`FlashAttention4 Stats` 每 30 秒往宿主 stdout 打一行
（`[FlashAttention4] stats {...}`）——因为 reset 节点在采样器**之前**执行，
不延迟打印的话只会看到全 0。

`FlashAttention4 Stats` 还会返回 `fa4_source`——`upstream`（PyPI 的 `flash_attn.cute`）、
`kernels`（SecondNatureComputing bundle）、或 `none`（还没加载）。
`FlashAttention4 Stats Line` 把同样的计数器格式化成一行，并追加
`FA4 ACTIVE` 或 `NOT A CLEAN FA4 RUN`。

要强制走某一路，在启动 ComfyUI 前设
`FLASH_ATTENTION4_SOURCE=upstream` 或 `=kernels`。默认 `auto`：先 upstream，再 kernels。

### 1.5 卸载

删掉 `custom_nodes/ComfyUI-FlashAttention4`，如果没别的东西用那五个 pip 包就一并卸掉，
重启 ComfyUI。本包没有别的东西被宿主依赖。

---

## 2 · 什么情况下回退到 SDPA

「SDPA」= PyTorch 自带的 `scaled_dot_product_attention`（通用保底实现）。
任何不支持的情况都落到它：画面不会崩，但也没有 FA4 的加速。

| 条件 | 原因 |
|---|---|
| `mask` 不是 None | FA4 没有 attention-mask 路径 |
| dtype 不是 fp16/bf16 | FA4 是半精度 kernel |
| 张量在 CPU / meta | 没有 CUDA kernel |
| `head_dim` 不在 (64, 128) | kernel 只支持这两个 |
| `enable_gqa=True` 且 `kv_heads != q_heads` | FA4 只在支持时走非 packed GQA 路径 |

每种情况都会加各自的计数器，所以降级**看得见**，不是静默。

---

## 3 · 报错怎么办

### 3.1 节点没出现

- 文件夹套多了一层（`custom_nodes/ComfyUI-FlashAttention4/ComfyUI-FlashAttention4/`）。往上挪。
- 没重启 ComfyUI。节点注册只在启动时发生。
- 看启动日志里的 `Cannot import ... ComfyUI-FlashAttention4`，traceback 会点名缺哪个模块。

### 3.2 `ModuleNotFoundError: No module named 'kernels'`（或 `quack`、`cutlass`、`cuda`）

**跑 ComfyUI 的那个解释器**里缺运行依赖。用 §1.1 的那个解释器再装一遍。自检：

```bash
<Python> -c "import kernels, cuda, cutlass, quack, tvm_ffi; print('ok')"
```

### 3.3 打了 `using SDPA`

`kernel_errors` 涨了。它前一行会点名原因：

| 日志 | 含义 | 怎么办 |
|---|---|---|
| `No module named 'flash_attn'` + bundle 下载失败 | `kernels` 拉不到 bundle | 查网络 / HF 访问，重试 |
| `CUDA error` / `no kernel image is available` | 这个显卡架构没有对应 kernel | 看 [§4](#4-支持的平台) |
| `head_dim` / `shape` 报错 | 不支持的张量形状 | 适配器本该拦住；请开 issue |

> 说明：这里的 “SDPA” 指 PyTorch 自带的通用注意力实现（`scaled_dot_product_attention`）。
> 它是**保底路径**——任何不支持的情况都退回它，画面不会崩，但也没有 FA4 的加速。

### 3.4 第一次调用要 30 秒（或几分钟）

Windows 上正常：CuTe DSL kernel 每次重启 ComfyUI 后 JIT 编译一次
（持久缓存被关掉，因为它要 `fcntl.flock`）。同一会话里后面的跑很快。

### 3.5 FA4 比之前还慢

某些负载上确实会——见 [§5](#5--基准)。FA4 是 dense BF16；量化或稀疏后端可能更快。
先用 `FlashAttention4 Stats` 确认 FA4 到底有没有在跑。

### 3.6 还是卡住

开 issue 时带上：ComfyUI 版本、torch/CUDA 版本、GPU 型号、
`[FlashAttention4]` 的日志行、`FlashAttention4 Stats` 的输出。
**不要贴 API key 或本机绝对路径。**

---

## 4 · 支持的平台

FA4（CuTe DSL）支持 **SM80 / SM89 / SM90 / SM100 / SM110 / SM120**——上游
`Dao-AILab/flash-attention` 写的是「optimize for Hopper and Blackwell」，而这个 bundle 的
`interface.py` 断言 `arch // 10 in [8, 9, 10, 11, 12]`。bundle 名里的 `sm120`
指的是它额外带的那批 PR，**不是硬件门槛**。

| 平台 | 状态 |
|---|---|
| Windows x86 + SM120/SM121（RTX 50） | **已测**：RTX 5070、torch 2.13+cu130、Python 3.12 |
| Linux/Windows x86 + SM80/SM89（Ampere/Ada） | 预期可用（有 SM80 kernel）；**本机未测** |
| Linux/Windows x86 + SM90（Hopper） | 预期可用（有 SM90 kernel）；**本机未测** |
| Linux/Windows x86 + SM100/SM110（Blackwell DC） | 预期可用（有 SM100 kernel）；**本机未测** |
| 纯 CPU / 无 GPU | 能 import 能跑；attention 回退 SDPA |
| AMD ROCm / Apple MPS / Windows-ARM | 没有 FA4 kernel；attention 回退 SDPA |

本节点**不做**架构检查：它调 `flash_attn_func`，由 bundle 自己分派。
没有对应 kernel ⇒ 调用抛错 ⇒ 回退 SDPA（计入 `kernel_errors`）。

---

## 5 · 基准

**测试底模**：MiniMax-H3（13.7–13.9B 级），底模文件
`Cobijada_H3_Hybrid_Uncensored_pruned_hybrid_int8_convrot_pruned.safetensors`
（INT8 convrot，21 GB）+ 两个 LoRA（`H3-Licon-VBVR-I2V-V2`、
`pika_h3_flex_ref2va_v1_avg_rank_55_bf16`），强度都是 1.0。
**底模和 LoRA 都不随包分发，需自备。**

**负载**：640×832、124 帧、6 步、`er_sde`、RTX 5070、
ComfyUI `Prompt executed` 秒数。cold = 重启后第一次。

| 后端 | Cold | Hot | 备注 |
|---|---:|---:|---|
| Kitchen INT8 + Spark | 121.7s | 67.1s | 稀疏 + 量化 dense |
| Kitchen INT8 only | 71.7s | 67.9s | 量化 dense |
| FA4 only | 126.5s | **116.1s** | dense FA4，比 Kitchen 慢约 1.7× |
| FA4 + Spark | 68.4s | 65.8s | FA4 没被用上（Spark 把 dense 路由到 Kitchen） |

### 步数比后端更值钱

同一张图、同一底模（Kitchen、LoRA×2、无 Spark），只改 sigma 表：

| 步数 | 耗时 |
|---:|---:|
| 3 | **39.9s** |
| 6 | 67.9s |
| 8 | 94.5s |

≈ 11 s/步（含固定 ~10s 的 VAE decode + save）。

### 质量快照（640×832×124，seed 12345）

| 跑 | 体区锐度 | 高频占比 | 时序 SSIM | 缝/体 |
|---|---:|---:|---:|---:|
| 3 步 | 31.5 | 0.9719 | 0.9946 | 0.893 |
| 6 步 | 39.7 | 0.9821 | 0.9784 | 0.715 |
| FA4 6 步 | 41.7 | 0.9818 | 0.9765 | 0.712 |

3 步比 6 步软约 21%，但帧间最稳；8 步相比 6 步没有可测的细节增益。

### 与稀疏包的相互作用

稀疏包（如 Spark-H3）会装它自己的 per-block attention patch。这时 FA4
**根本收不到那些调用**——实测里 Spark 一直打 `comfy_kitchen_calls`，FA4 计数器保持 0。
想让 FA4 接管 dense 路径，要显式设稀疏包的 `dense_backend`；`inherit` **不等于** FA4。

---

## 6 · 与 `comfyui-flash-attention4` 无关

registry id `comfyui-flash-attention4` 属于另一个**已不维护**的包
（`sorryhyun`，v1.0.0，2026-03）。本项目是 **`comfyui-fa4`**——独立实现，
不打包 kernel 源码，通过 `kernels` 装 SM120 bundle。

---

## 7 · 许可

本节点包 MIT。FA4 kernel bundle 是 BSD-3-Clause（Dao-AILab）。