# CHANGES

> 开发者向的历史记录。用户可见的升级说明写在 [`RELEASE-NOTES.md`](RELEASE-NOTES.md)。

## 0.2.1 — 2026-11-24
- 📊 README 的 A/B 表换成**同宿主、同 seed、同 sigma 表**的实测值（6 步）：
  Kitchen+Spark 67.1s(热) / Kitchen 67.9s(热) / FA4 116.1s(热) / FA4+Spark 65.8s。
- 📉 新增步数-时间曲线（3/6/8 步 = 39.9 / 67.9 / 94.5s，≈11s/步）。
- 🔍 新增质量快照（body sharpness / HF ratio / temporal SSIM / seam-body）：
  3 步比 6 步软约 21%，但帧间最稳；8 步不再增加细节。
- ⚠️ README 明写：Spark 在链上时 **FA4 收不到 dense 调用**（Spark 日志仍
  `comfy_kitchen_calls`，`dense_backend="inherit"` 不等于 FA4）。
- 🧹 开源卫生：移除内部语境（无内部工作流名、无与仓库无关的基线代号、无单项目结论）；
  工具提示英文；扫描无本机路径、无凭据、无 CJK。
- 🧰 `tests/test_metadata.py` 断言与重写后的 README 对齐（新增 `FlashAttention4Stats` / `Spark-H3`）。
- 📦 首次 `git init` + 提交（此前只有工作目录，没有版本历史）。

## 0.2.0 — 2026-11-24
- ✨ 新增 `FlashAttention4Stats`（7 个 INT 输出 + `reset`），A/B 时可读
  `fa4` / `sdpa_mask` / `sdpa_dtype` / `sdpa_gqa` / `sdpa_headdim` / `sdpa_meta` / `kernel_errors`。
- 🛡️ dispatch 表：mask / dtype / GQA / head_dim / meta 一次性判定，逐类计数；
  FA4 无 attn-mask 路径，mask 非空一律走 SDPA。
- 🐛 `container_function` 现在会 `.take()` 解包 `AttentionTensorContainer`；
  修掉 Spark 组合路径里的 `'AttentionTensorContainer' object has no attribute 'ndim'`。
- 🧪 `FlashAttention4Stats.read(reset=True)` 额外起 daemon 线程，30s 后打印
  `live stats`，解决「reset 节点先于采样器执行，立即打印全是 0」。
- 🐛 `nodes.py` 注册两个节点（`FlashAttention4ModelPatch` + `FlashAttention4Stats`）。
- ✅ 零 GPU 闸 `tools/check_zero_gpu.py` 覆盖 adapter / node-contract / metadata / ruff。

## 0.1.0 — 2026-11-23
- ✨ 首个版本：`FlashAttention4ModelPatch`（MODEL → MODEL）+ `attention_fa4` 适配器。
- 🔧 两条加载路径：`flash_attn.cute`（上游 FA4）与
  `kernels.get_kernel("SecondNatureComputing/flash-attn-4-sm120")`（RTX 50 的 SM120 bundle）。
- 🪟 Windows 兼容层：`fcntl` stub、`quack` 包壳、`cutlass` ThrMma / make_fragment 别名。
- 🛡️ mask / CPU / fp32 / 不支持 head_dim 回退 SDPA。
- ✅ 零 GPU 机检 + ruff。