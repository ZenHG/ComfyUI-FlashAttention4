# THIRD-PARTY-NOTICES

本包（ComfyUI-FlashAttention4）自身代码为 MIT。以下是与本包**相关但未随包分发**的第三方内容，
以及本包与其关系的准确表述。

## 一、运行时宿主：ComfyUI（GPL-3.0）

- 本包**不复制** ComfyUI 的任何源码，但在运行时 `import comfy.ldm.modules.attention`
  并以自定义节点形式运行在宿主进程内。
- ComfyUI 本体许可见 <https://github.com/comfyanonymous/ComfyUI/blob/master/LICENSE>（GPL-3.0）。
- 本包调用的宿主 API 属**公开内核**：`comfy.ldm.modules.attention.register_attention_function`、
  `ModelPatcher.clone` / `set_model_optimized_attention`。这些都是宿主节点自己也在用的入口。

## 二、FA4 kernel bundle：BSD-3-Clause（Dao-AILab）

- 本包**不打包** kernel 源码；`kernels` 在首次使用时从 Hugging Face 拉
  `SecondNatureComputing/flash-attn-4-sm120`（下游分发，含 6 个未合并的上游 SM120 PR）。
- 上游：`Dao-AILab/flash-attention`（FlashAttention-4，CuTe DSL）。
- 许可：BSD-3-Clause；完整文本见 bundle 内的 `LICENSE`。

## 三、运行栈（pip 依赖）

| 名称 | 用途 | 许可 |
|---|---|---|
| `kernels` | 下载 / 加载 kernel bundle | Apache-2.0 |
| `nvidia-cutlass-dsl` | CuTe DSL 编译器 | NVIDIA License |
| `quack-kernels` | CuTe 编译辅助 | Apache-2.0 |
| `apache-tvm-ffi` | TVM FFI | Apache-2.0 |
| `cuda-python` | CUDA driver 绑定 | CUDA Toolkit 许可 |

⚠ 本包不对上述依赖的许可做任何担保或转授。商用前请自行核对各项目当次许可文本。

## 四、思路来源（无代码复制）

- `simonri/ComfyUI-flash-attention-3` 的 `attention_flash_4` 适配器：本包的
  `attention_fa4.py` 按其签名与 layout 约定重写；**未复制其源码**。
- `zechengtang/Spark-H3` 的 `kernel_builder.py` / `nodes.py`：本包的节点骨架按其
  MODEL patch 结构重写；**未复制其源码**。
