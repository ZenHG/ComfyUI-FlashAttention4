# 发布说明（Release Notes）

> **这份文件写给用户，不是写给开发者。**
> 两个渠道的用户可见说明都取这里（唯一真相源）：Comfy Registry 的 changelog 与 GitHub Release 的正文。
>
> 格式规矩：
> - 一版一节，**新的在最上面**，标题固定 `## x.y.z — YYYY-MM-DD`。
> - 每节**必须中英双语**：中文在前，一行 `<!-- EN -->` 分隔，英文在后。
> - 只写**用户能感知的变化**。有行为变化必须用 🔴 点明「要不要动配置」。
> - 每节结尾固定两行：中文 `**升级动作**：…` / 英文 `**Action required**: …`。

---

## 0.2.1 — 2026-11-24

**首个公开版本。FA4 的 ComfyUI MODEL patch，零elnstall 依赖，只包壳不重编 kernel。**

- 新增 `FlashAttention4 Model Patch`（MODEL → MODEL）：把 dense attention 路由到
  FlashAttention 4（CuTe DSL），SM120/SM121 的 kernel 由 `kernels` 在首次调用时下载。
- 新增 `FlashAttention4 Stats`：7 个 INT 输出，A/B 时能直接读「FA4 真的跑了几次、
  有多少次回退到 SDPA」。每 30 秒还会往宿主 stdout 打一行 `live stats`，不用等收尾。
- 🔴 需要宿主 Python 能 import `kernels`、`cuda-python`、`nvidia-cutlass-dsl[cu13]`、
  `quack-kernels`、`apache-tvm-ffi`；装到错误的 Python 环境里节点不会生效。
- ⚠ FA4 是 dense kernel：在量化 / 稀疏后端更快的负载上，FA4 可能更慢。
  先量再换。
- ⚠ 如果链上有 Spark-H3 这类稀疏节点，FA4 **收不到** dense 调用（Spark 自己路由
  到 Kitchen）。要让 FA4 生效，需显式设 Spark 的 `dense_backend`；`inherit` 不等于 FA4。

**升级动作**：新装。把仓库 clone 到 `ComfyUI/custom_nodes/ComfyUI-FlashAttention4`，
按 README 装运行依赖，重启 ComfyUI。

<!-- EN -->

**First public release. A ComfyUI MODEL patch for FA4. No pip dependencies, wrapper only, no kernel rebuild.**

- New `FlashAttention4 Model Patch` (MODEL → MODEL): routes dense attention through
  FlashAttention 4 (CuTe DSL). The SM120/SM121 kernel is downloaded by `kernels` on first call.
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

**升级动作**：无。新装即可；已装的用户直接覆盖，不需要改配置。
