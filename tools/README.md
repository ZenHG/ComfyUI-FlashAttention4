# tools/

开发 / 验收工具。**不进发布包**（`.comfyignore` 排除 `/tools/`）。

| 脚本 | 用途 | 需要 |
|---|---|---|
| `check_zero_gpu.py` | 零 GPU 闸：byte-compile + adapter / node-contract / metadata 测试 + ruff | `ruff`（没有则 SKIP） |
| `ab_quality.py` | A/B 质量第一轮：head/body 锐度、Laplacian 方差、接缝跳变、contact sheet | `cv2` + `numpy` + 已渲染的 mp4 |
| `ab_quality2.py` | A/B 质量第二轮：FFT 高频占比、8×8 块效应、RMS 对比度、时序 SSIM | `cv2` + `numpy` + 已渲染的 mp4 |

## 环境变量

`ab_quality*.py` 默认读 `I:\ComfyUI\output`（开发机路径，仅为默认值）：

| 变量 | 作用 | 默认 |
|---|---|---|
| `COMFYUI_OUTPUT` | 视频目录 | `I:\ComfyUI\output` |
| `AB_JSON_OUT` | 指标 JSON 输出路径 | `<OUT>/ab_quality.json` |
| `AB_SHEET` | contact sheet 输出路径 | `<OUT>/ab_contact_sheet.png` |

换机器时只改环境变量，不改脚本：

```bash
COMFYUI_OUTPUT=/path/to/ComfyUI/output \
AB_SHEET=/tmp/ab_sheet.png \
I:\python\python.exe tools/ab_quality.py
```

## 视频清单

脚本顶部 `VIDS` 字典是「标签 → 文件名」，按实际产物改：

```python
VIDS = {
    "3step":      "abT3_C2_4step_00001_.mp4",
    "6step":      "abT3_C2_00001_.mp4",
    "6step_spark":"abT3_A0_00001_.mp4",
    "fa4_6step":  "abT3_C1_cold_00001_.mp4",
}
```

## 判读

- `seam/body` ≈ 1 ⇒ 接缝与正常运动同量级（好）；>1.5 ⇒ 接缝跳。
- `blockiness` ≈ 1.0 ⇒ 无 8×8 块效应；>1.1 ⇒ 可疑。
- `fa4` 计数 >0 且 `sdpa_*` = 0 ⇒ FA4 真的在跑（见 `FlashAttention4Stats`）。