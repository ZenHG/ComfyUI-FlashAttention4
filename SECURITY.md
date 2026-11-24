# SECURITY

## 本包的攻击面（如实披露）

本包是本地 ComfyUI 自定义节点，运行在用户自己的进程里。

- **不联网**：没有遥测、自动更新、远程配置。
- **不下载权重**：节点不读任何远程模型文件。
- **不执行外部命令**：包内没有 `subprocess` / `eval` / `exec`。
- **不读用户目录**：除 ComfyUI 自己的模型 / 输出目录外不访问文件系统。
- **kernel 由 `kernels` 下载**：首次使用时从 Hugging Face 拉
  `SecondNatureComputing/flash-attn-4-sm120`；这是唯一的外发请求，且只发生在第一次 attention 调用。
- 包内不含任何模型权重、可执行文件或 pip 包。

## 报告漏洞

**请不要开公开 issue。** 用 GitHub 的 Private vulnerability reporting
（仓库 → Security → Report a vulnerability）。

## 依赖链

- `kernels` / `nvidia-cutlass-dsl` / `quack-kernels` / `apache-tvm-ffi` 均为上游项目，
  各自许可见其仓库；本包不对其做担保。
