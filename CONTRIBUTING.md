# CONTRIBUTING

## 一句话

改代码 → 跑下面的闸 → 全绿再提 PR。

```bash
python -m ruff check .
python tools/check_zero_gpu.py
```

`tools/check_zero_gpu.py` 覆盖：

1. 所有 `.py` 的 byte-compile；
2. `tests/test_adapter.py` —— dispatch 表 / mask / GQA / head_dim / container 解包；
3. `tests/test_node_contract.py` —— 注册 / MODEL patch 克隆语义 / Stats 契约；
4. `tests/test_metadata.py` —— pyproject / 版本同步 / README 事实；
5. `ruff check`。

## 判据纪律

- 期望值写在测试里，**不写行号**。
- 新写 / 改判据时做一次**手工变异测试**（把修复摘掉应当变红）。
- 口径数字（版本号、节点数、断言数）改动 ⇒ 四处同步：
  `__init__.py` / `pyproject.toml` / `CHANGES.md` / `README.md`。

## 提交

- 一件事一个提交；修 bug 别顺手重构。
- 提交信息写「条件 + 结论」，不写内部沟通、不写本机路径。
