# CONTRIBUTING

## 一句话

改代码 → 跑下面的闸 → 全绿再提 PR。

```bash
python -m ruff check .
python tools/check_zero_gpu.py
```

⚠️ 别用 `| tail` 看输出——退出码和计数会被吃掉。要退出码就重定向到文件后 `echo $?`。

## 闸覆盖什么

`tools/check_zero_gpu.py`（退出码 0 = 全绿，1 = 至少一条红）：

1. 所有 `.py` 的 byte-compile（语法闸）；
2. `tests/test_adapter.py` —— dispatch 表 / mask / GQA / head_dim / container 解包 / STATS 计数；
3. `tests/test_node_contract.py` —— 注册 / MODEL patch 克隆语义 / Stats / Stats Line 契约；
4. `tests/test_metadata.py` —— pyproject / 版本同步 / README 事实 / name；
5. `ruff check`（没装 ruff ⇒ SKIP，不算过）。

## 三档检查（三个「通过」互不相等）

| 档 | 怎么跑 | 覆盖什么 |
|---|---|---|
| **本地零 GPU** | 上面两条 | 语法 / 适配器 / 节点契约 / 元数据 / ruff |
| **CI** | `.github/workflows/ci.yml`（Python 3.10 / 3.11 / 3.12） | 干净宿主 + 跨 Python 版本 |
| **真机** | 起 ComfyUI → 插节点 → 跑一张图 → 读 Stats | FA4 真加载、无回退、画面不崩 |

🔴 **真机档没有自动化**：CI 无 GPU；本仓无前端 JS，`ui_smoke` 不适用。
真机验收看三件事：日志有 `FlashAttention 4 kernel loaded`、没有
`FlashAttention 4 failed, using SDPA`、`FlashAttention4Stats` 的 `fa4 > 0`。

## 改什么 → 至少跑什么

| 改了什么 | 至少跑 |
|---|---|
| 任意 Python | `python -m ruff check .` |
| `attention_fa4.py` dispatch | `python tests/test_adapter.py` |
| `attention_fa4.py` 的 `STATS` 键 | `python tests/test_adapter.py` + `python tests/test_node_contract.py` |
| `nodes.py` 的 `INPUT_TYPES` | `python tests/test_node_contract.py` |
| `pyproject.toml` / `README.md` | `python tests/test_metadata.py` |
| 版本号 | 四处同步：`__init__.py` / `pyproject.toml` / `CHANGES.md` 顶部 / `README.md` |
| `tools/` 脚本 | `python tools/check_zero_gpu.py`（会 byte-compile 全部 `.py`） |

## 判据纪律

- 期望值写在测试里，**不写行号**（会漂）。
- 新写 / 改判据时做一次**手工变异测试**（把修复摘掉应当变红）。
- 口径数字（版本号、节点数、断言数）改动 ⇒ **四处同步**：
  `__init__.py` / `pyproject.toml` / `CHANGES.md` / `README.md`。

## 开源卫生（三条硬要求）

1. **不写本机路径**：任何盘符绝对路径（`<盘符>:\...`、用户家目录）一律不进仓库；示例用 `<ComfyUI>` / `/path/to/...`。
2. **不写凭据**：registry PAT 只放 `<仓根>/.comfy_registry_token`（已 gitignore）。
3. **不留未跟踪文件**：`git status --short` 必须干净（`LOCAL-*.md`、`审计报告/` 已被 `.gitignore` 挡住）。

## 提交

- 一件事一个提交；修 bug 别顺手重构。
- 提交信息写「条件 + 结论」，不写内部沟通、不写本机路径。
- 不 `git add -A`；不替人提交、不替人推送、不重写历史。

## 发布

见 [`RELEASING.md`](RELEASING.md)。用户可见的升级说明写在
[`RELEASE-NOTES.md`](RELEASE-NOTES.md)（中英成对）。