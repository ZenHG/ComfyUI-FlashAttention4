# RELEASING · 维护者发布手册

> 两个渠道：**GitHub**（仓库 + Release）与 **Comfy Registry**（节点市场）。
> 本文只写"为什么"和"失败怎么处置"；判据在脚本与 CI 里。

---

## 0. 一次发布要满足的前提

| 项 | 怎么满足 |
|---|---|
| 工作树干净 | 发出去的东西必须是**提交态** |
| 版本号四处一致 | `__init__.py` / `pyproject.toml` / `CHANGES.md` / `README.md` |
| `RELEASE-NOTES.md` 有本版一节 | 中英双语；中文 ≤700 字符、英文 ≤1700 字符 |
| 零 GPU 闸绿 | `python tools/check_zero_gpu.py` |
| `gh` 可用 | 建 Release、等 CI |
| `comfy`（comfy-cli）真能跑 | `comfy --version` 返回 0 |
| registry PAT | 只放 `<仓根>/.comfy_registry_token`（已 gitignore） |

---

## 1. 两个渠道，两种失败

- **只 push 不 publish** ⇒ GitHub 上是新版、Manager 里还是旧版。
- **只 publish 不 push** ⇒ registry 包的 `repository` 指向的代码里没有这一版。
- **push 了但 CI 红** ⇒ 已经公开的提交没通过自检。

三件事都要同时成立才算「发布完成」。

---

## 2. 步骤

```bash
# ① 预检
python tools/check_zero_gpu.py
python tools/check_release_ready.py

# ② 提交 + 打 tag
git add . && git commit -m "release: x.y.z"
git tag vX.Y.Z

# ③ 建远端（只做一次）
gh repo create ZenHG/ComfyUI-FlashAttention4 --public --source=. --remote=origin --push

# ④ 推
git push -u origin master --tags

# ⑤ 等 CI
gh run watch --exit-status

# ⑥ GitHub Release
gh release create vX.Y.Z --title "vX.Y.Z" --notes-file RELEASE-NOTES.md

# ⑦ registry（需要 comfy-cli + PAT）
comfy node publish --token "$(cat .comfy_registry_token)"
```

---

## 3. 失败处置

| 步 | 失败了怎么办 |
|---|---|
| ① 预检 | 直接修，**什么都没有发生** |
| ④ push | 网络层问题；先 `git ls-remote` 核对 |
| ⑤ CI | 红 ⇒ 修完重跑；**不要**发 registry |
| ⑦ registry | 版本接口只收 GET ⇒ changelog **发布后不可改**；发之前把 RELEASE-NOTES 写对 |

---

## 4. registry PAT

- 只放 `<仓根>/.comfy_registry_token`（已 gitignore）。
- 值只允许存在于这个文件；不进代码、不进提交信息。
