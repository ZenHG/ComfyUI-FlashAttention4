# 发布说明（Release Notes）

> **这份文件写给用户，不是写给开发者。**
> 两个渠道的用户可见说明都取这里（唯一真相源）：Comfy Registry 的 changelog 与 GitHub Release 的正文。
>
> 格式规矩：
> - 一版一节，**新的在最上面**，标题固定 `## x.y.z — YYYY-MM-DD`。
> - 每节**必须中英双语**：中文在前，一行 `<!-- EN -->` 分隔，英文在后。
> - 只写**用户能感知的变化**。有行为变化必须用 🔴 点明「要不要动配置」。
> - 每节结尾固定两行：中文 `**升级动作**：…` / 英文 `**Action required**: …`。
>
> 长度：中文段 ≤700 字符，英文段 ≤1700 字符。

---

## 0.2.4 — 2026-11-24

**能看出 FA4 是从哪条路加载的；PyPI 装法写进 README。**

- `FlashAttention4 Stats` 多一个 `fa4_source` 输出：`upstream` / `kernels` / `none`。
- 新增 `FLASH_ATTENTION4_SOURCE=auto|upstream|kernels` 环境变量，可强制选路。
- README 增加 PyPI `flash-attn-4` 安装路（上游 4.0.0b40 已自带 SM120 kernel）。
- 🔴 行为没变；老工作流不用改。

**升级动作**：无。覆盖安装即可。

<!-- EN -->

**You can now see which route loaded FA4; the PyPI install path is documented.**

- `FlashAttention4 Stats` gains a `fa4_source` output: `upstream` / `kernels` / `none`.
- New `FLASH_ATTENTION4_SOURCE=auto|upstream|kernels` env var to force a route.
- README adds the PyPI `flash-attn-4` install route (4.0.0b40 ships SM120 kernels).
- 🔴 No behaviour change; existing workflows keep working.

**Action required**: None. Overwrite to upgrade.

## 0.2.3 — 2026-11-24

**Registry 包名定为 `comfyui-fa4`；FA4 的支持范围写全为 SM80 及以上。**

- 包名从 `comfyui-flash-attention4` 改为 `comfyui-fa4`（前者在 registry 上已被别的包占用，无法使用）。
- FA4 支持范围按上游源码写全：**SM80 / SM89 / SM90 / SM100 / SM110 / SM120**（Ampere、Ada、Hopper、Blackwell）。
  本机只实测过 SM120（RTX 5070）；其它架构标为「预期可用、未实测」。
- 节点名、节点分类、参数都没变；已装的用户升级不需要改工作流。
- ⚠ 本节点**不做**显卡架构检查：调用 `flash_attn_func`，由 kernel bundle 自己分派；
  没有对应 kernel 时报错并回退 SDPA（计入 `kernel_errors`）。

**升级动作**：已装的用户直接覆盖 `custom_nodes/comfyui-fa4`；从 Manager 安装的走 Manager 更新。

<!-- EN -->

**Registry package name is now `comfyui-fa4`; FA4 support is stated as SM80 and newer.**

- Package name changed from `comfyui-flash-attention4` to `comfyui-fa4` (the former is already taken on the registry by another pack).
- FA4 support now matches upstream source: **SM80 / SM89 / SM90 / SM100 / SM110 / SM120** (Ampere, Ada, Hopper, Blackwell).
  Only SM120 (RTX 5070) was tested here; other architectures are marked "expected to work, not tested".
- Node names, category and parameters are unchanged; existing workflows keep working.
- ⚠ This node does **not** check the GPU architecture itself: it calls `flash_attn_func` and lets the kernel bundle dispatch. With no matching kernel it raises and falls back to SDPA (counted in `kernel_errors`).

**Action required**: Overwrite `custom_nodes/comfyui-fa4` if you installed manually, or update through the Manager.

---

## 0.2.2 — 2026-11-24

**新增 `FlashAttention4 Stats Line`：把 7 个计数器变成一行可读文本。**

- 新增节点 `FlashAttention4 Stats Line`（7 个 INT 输入 → 1 个 STRING 输出）。
- 输出形如：`fa4=15 sdpa_mask=0 sdpa_dtype=0 sdpa_gqa=0 sdpa_headdim=0 sdpa_meta=0 kernel_errors=0  => FA4 ACTIVE`。
- 只要有任何一项回退，结尾会标 `NOT A CLEAN FA4 RUN`。
- 解决了「`FlashAttention4 Stats` 的 7 个 INT 输出在前端只能看到第一个」的问题。

**升级动作**：无。新装或覆盖即可；旧工作流不受影响。

<!-- EN -->

**New `FlashAttention4 Stats Line` node: the seven counters as one readable string.**

- New node `FlashAttention4 Stats Line` (7 INT inputs → 1 STRING output).
- Output looks like: `fa4=15 sdpa_mask=0 ... kernel_errors=0  => FA4 ACTIVE`.
- Any fallback marks the line `NOT A CLEAN FA4 RUN`.
- Fixes the UI limitation where only the first of the seven `FlashAttention4 Stats` outputs was visible.

**Action required**: None. Install or overwrite; existing workflows are unaffected.

---

## 0.2.1 — 2026-11-24

**首个公开版本。FA4 的 ComfyUI MODEL patch，零 pip 依赖，只包壳不重编 kernel。**

- 新增 `FlashAttention4 Model Patch`（MODEL → MODEL）：把 dense attention 路由到
  FlashAttention 4（CuTe DSL），kernel 由 `kernels` 首次调用时下载。
- 新增 `FlashAttention4 Stats`：7 个 INT 输出，读「FA4 跑了几次、回退几次」。
  每 30 秒打一行 `live stats` 到宿主 stdout。
- 🔴 宿主 Python 必须能 import `kernels`、`cuda-python`、`nvidia-cutlass-dsl[cu13]`、
  `quack-kernels`、`apache-tvm-ffi`；装错环境节点不生效。
- ⚠ FA4 是 dense kernel：量化 / 稀疏后端更快的负载上可能更慢，先量再换。
- ⚠ 链上有 Spark-H3 时 FA4 **收不到** dense 调用；需显式设 Spark 的
  `dense_backend`，`inherit` 不等于 FA4。

**升级动作**：新装。clone 到 `ComfyUI/custom_nodes/ComfyUI-FlashAttention4`，
按 README 装依赖，重启 ComfyUI。

<!-- EN -->

**First public release. A ComfyUI MODEL patch for FA4. No pip dependencies, wrapper only, no kernel rebuild.**

- New `FlashAttention4 Model Patch` (MODEL → MODEL): routes dense attention through
  FlashAttention 4 (CuTe DSL). The kernel is downloaded by `kernels` on first call.
- New `FlashAttention4 Stats`: 7 INT outputs so an A/B run can read how many attention calls
  went through FA4 and how many fell back to SDPA. It also prints a `live stats` line to
  the host stdout every 30 s, so you do not have to wait for the run to finish.
- 🔴 The host Python must be able to import `kernels`, `cuda-python`,
  `nvidia-cutlass-dsl[cu13]`, `quack-kernels`, `apache-tvm-ffi`. Installed into the
  wrong environment, the node will not load.
- ⚠ FA4 is a dense kernel: on workloads where a quantized or sparse backend is faster,
  FA4 can be slower. Benchmark before swapping a production chain.
- ⚠ If a sparse pack (e.g. Spark-H3) is in the chain, FA4 **will not receive** dense
  calls — Spark routes them to Kitchen itself. Set Spark's `dense_backend` explicitly;
  `inherit` does not mean FA4.

**Action required**: New install. Clone into `ComfyUI/custom_nodes/ComfyUI-FlashAttention4`,
install the runtime deps from the README, restart ComfyUI.