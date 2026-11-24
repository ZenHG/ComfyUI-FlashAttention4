# AGENTS.md — ComfyUI-FlashAttention4 的作业入口

> 给**改这个仓库的 agent** 读。
> 🔴 **本文件不是规范源**。规范源是 [`CONTRIBUTING.md`](CONTRIBUTING.md)；本文件只做**路由与判据映射**，冲突时以 `CONTRIBUTING.md` 为准。
> 🔴 **本文件不写任何项数 / 断言数 / job 数**——写在这里的数字没有闸能抓，必然漂。要数字就去 `CONTRIBUTING.md` 的命令注释、`.github/workflows/ci.yml` 文件头，或现场跑。

---

## 0 · 30 秒版（只读这 5 条也能做对第一件事）

1. **先跑基线，别先读代码**：
   ```bash
   python -m ruff check .
   python tools/check_zero_gpu.py
   ```
   两条都绿才动代码。`check_zero_gpu.py` 的退出码就是结论（0 = 全绿，1 = 至少一条红）。

2. **改参数 / 节点 = 四处一起动**：
   `nodes.py` 的 `INPUT_TYPES()` · `README.md` 的节点清单 · `tests/test_metadata.py` 的 README 断言 · `pyproject.toml` 的 `[tool.comfy]`。
   新 widget **只能加在末尾**（`widgets_values` 按位置存）。

3. **改 STATS 键 = 三处一起动**：
   `attention_fa4.py` 的 `STATS` 字典 · `nodes.py` 的 `FlashAttention4Stats.INPUT_TYPES()` / `read()` · `tests/test_node_contract.py` 的 Stats 用例。
   漏一处 ⇒ 前端显示 0 或 `KeyError`。

4. **交 PR 前的最小集合**：
   ```bash
   python -m ruff check .
   python tools/check_zero_gpu.py
   ```
   ⚠️ 别用 `| tail` 看输出——退出码和计数会被吃掉。要计数就 `grep -E "失败 [0-9]+"`，要退出码就重定向到文件后 `echo $?`。

5. **三条禁令**：
   - 别把「SKIP / 没跑」写成「通过」；
   - 别在仓库里留未跟踪文件（`git status --short` 必须干净）；
   - 别改包名 `comfyui-fa4`（registry node id，发布后不可改）。

---

## 1 · 三档检查（三个「通过」互不相等）

| 档 | 怎么跑 | 覆盖什么 | 跑不了时 |
|---|---|---|---|
| **本地零 GPU** | `python tools/check_zero_gpu.py` | byte-compile / adapter dispatch / node 契约 / 元数据 / ruff | —— |
| **CI** | `.github/workflows/ci.yml`（Python 3.10 / 3.11 / 3.12 矩阵） | 干净宿主 + 跨 Python 版本 | 本地无法等价复现 runner ⇒ PR 以 CI 实跑为准 |
| **真机（GPU + 宿主）** | 起 ComfyUI → 插 `FlashAttention4 Model Patch` → 跑一张图 → 读 `FlashAttention4 Stats` | FA4 真的加载、无 `using SDPA`、画面不崩 | **点名说没跑**，不许当作已覆盖 |

🔴 **FA4 的真机档没有自动化**：CI 无 GPU，`ui_smoke` 之类不适用（本仓无前端 JS）。
真机验收只能靠人：看宿主日志有没有 `FlashAttention 4 kernel loaded`、有没有
`FlashAttention 4 failed, using SDPA`、Stats 的 `fa4` 是否 > 0。

---

## 2 · 改什么 → 至少跑什么

| 你改了什么 | 至少跑 | 为什么 |
|---|---|---|
| 任意 Python | `python -m ruff check .` | CI 第一道 |
| `attention_fa4.py` 的 dispatch 表 | `python tests/test_adapter.py` | 每条回退路径（mask / dtype / GQA / head_dim / meta）都要有断言 |
| `attention_fa4.py` 的 `STATS` 键 | `python tests/test_adapter.py` + `python tests/test_node_contract.py` | 键名是 A/B 的唯一读数口 |
| `nodes.py` 的 `INPUT_TYPES` | `python tests/test_node_contract.py` + `python tests/test_metadata.py` | 注册 / 克隆语义 / README 事实 |
| `pyproject.toml` 的 `[project]` / `[tool.comfy]` | `python tests/test_metadata.py` | name / version / license / PublisherId |
| `README.md` | `python tests/test_metadata.py` | README 事实断言（含版本号） |
| 版本号 | 四处：`__init__.py` / `pyproject.toml` / `CHANGES.md` 顶部 / `README.md` | 漏一处 = 文档与代码不一致 |
| `tools/` 脚本 | `python tools/check_zero_gpu.py`（会 byte-compile 全部 `.py`） | 语法 + ruff |

---

## 4 · 铁律 ↔ 执行体

| 规则 | 原文位置 | 执行体 | 强度 |
|---|---|---|---|
| **不许把本机的东西写进仓库** | `CONTRIBUTING.md` | 人工 grep + `tools/README.md` 的路径约定 | ⚠️ **无机检**（`H3e` 那种受控文本扫描本仓没有） |
| **不许把凭据写进受控文件** | `CONTRIBUTING.md` | 人工 + `.gitignore` 的 `.comfy_registry_token` | ⚠️ 无机检 |
| **硬错误必须 raise，绝不静默降级** | `CONTRIBUTING.md` | `tests/test_adapter.py` 的 fallback 断言 | 半机检（只覆盖 adapter） |
| **给建议必须把好与坏都写透** | `CONTRIBUTING.md` | README 的 Benchmarks + 「FA4 更慢」结论 | 人工 |
| **包名发布后不可改** | `CONTRIBUTING.md` | `tests/test_metadata.py` 的 name 断言 | 机检（只挡改动，不挡发布） |

🔴 **本仓最大的薄弱点**：**没有 `H3e` 那样的本机路径机检**。`LOCAL-*.md`、`审计报告/`、
`.comfyui_path` 都靠 `.gitignore` + 人工。加一条机检 = 新脚本 + 新腐化面，先量再定。

---

## 5 · 已知薄弱点（诚实列，别当已覆盖）

| # | 薄弱点 | 现状 |
|---|---|---|
| 1 | **本机路径无机检** | 靠 `.gitignore` + 人工 grep；`tools/README.md` 已把默认路径改成环境变量 |
| 2 | **凭据无机检** | 靠 `.gitignore` 的 `.comfy_registry_token`；提交前人工看 |
| 3 | **py3.10 只在 CI 跑** | 本机是 3.12；`requires-python = ">=3.10"` ⇒ 本地跑不了 3.10 |
| 4 | **真机档无机检** | 要活宿主 + GPU；CI 没有 |
| 5 | **无类型检查** | 无 mypy / pyright |
| 6 | **`tools/ab_quality*.py` 需要 cv2** | 不在 CI（CI 只跑 `check_zero_gpu.py`）；本地缺 cv2 会 SKIP |

---

## 6 · 提交

- 一件事一个提交；修 bug 别顺手重构。
- 提交信息写「条件 + 结论」，不写内部沟通、不写本机路径。
- 不 `git add -A`；不替人提交、不替人推送、不重写历史。