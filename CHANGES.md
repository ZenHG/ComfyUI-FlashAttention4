# CHANGES

> 开发者向的历史记录。用户可见的升级说明写在 [`RELEASE-NOTES.md`](RELEASE-NOTES.md)。

## 0.2.2 — 2026-11-24
- 🧹 开源卫生：README 改为英文、移除内部语境（无「母版 / 152s 档 / H3 结论」），
  工具提示改为英文；扫描无本机路径、无凭据、无 CJK。
- 📊 README 的 A/B 表改为实测值（Kitchen+Spark 77.4–141.8s / Kitchen only 86.9s /
  FA4 only 141.1–177.9s / FA4+Spark 85.6s）。
- 🧰 `tests/test_metadata.py` 的 README 断言与重写后的 README 对齐。
- 📦 首次 `git init` + 提交（此前只有工作目录，没有版本历史）。

## 0.2.1 — 2026-11-23
- 🐛 `container_function` 现在会 `.take()` 解包 `AttentionTensorContainer`；
  修掉 Spark 组合路径里的 `'AttentionTensorContainer' object has no attribute 'ndim'`。
- 🧪 `FlashAttention4Stats.read(reset=True)` 额外起 daemon 线程，30s 后打印
  `deferred stats`，解决「reset 节点先于采样器执行，立即打印全是 0」。

## 0.2.0 — 2026-11-23
- ✨ 新增 `FlashAttention4Stats`（7 个 INT 输出 + `reset`），A/B 时可读
  `fa4` / `sdpa_mask` / `sdpa_dtype` / `sdpa_gqa` / `sdpa_headdim` / `sdpa_meta` / `kernel_errors`。
- 🛡️ dispatch 表：mask / dtype / GQA / head_dim / meta 一次性判定，逐类计数；
  FA4 无 attn-mask 路径，mask 非空一律走 SDPA。
- 🐛 `nodes.py` 注册两个节点（`FlashAttention4ModelPatch` + `FlashAttention4Stats`）。
- ✅ 零 GPU 闸 `tools/check_zero_gpu.py` 覆盖 adapter / node-contract / metadata / ruff。
