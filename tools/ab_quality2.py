# -*- coding: utf-8 -*-
"""Second-pass quality metrics: contrast, HF energy, blockiness, temporal SSIM.

Zero GPU. Complements tools_local_ab_quality.py with metrics that are less
sensitive to overall motion magnitude.

Run: I:\\python\\python.exe tools_local_ab_quality2.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

OUT = os.environ.get("COMFYUI_OUTPUT", r"I:\ComfyUI\output")
JSON_OUT = os.environ.get("AB_JSON_OUT", os.path.join(OUT, "ab_quality2.json"))

VIDS = {
    "3step": "abT3_C2_4step_00001_.mp4",
    "6step": "abT3_C2_00001_.mp4",
    "6step_spark": "abT3_A0_00001_.mp4",
    "fa4_6step": "abT3_C1_cold_00001_.mp4",
    "8step": "abT3_C2_8step_00001_.mp4",
}


def read_gray(path: str):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames


def hf_ratio(gray):
    """High-frequency energy / total energy via FFT (normalized)."""
    f = np.fft.fft2(gray.astype(np.float32))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = min(cy, cx)
    hi = mag[r > rmax * 0.5].sum()
    total = mag.sum() + 1e-8
    return float(hi / total)


def blockiness(gray):
    """Mean |diff| across 8x8 block boundaries minus inside-block diff."""
    g = gray.astype(np.float32)
    h, w = g.shape
    # vertical boundaries: diff between column x and x+1 at x = 7, 15, 23...
    cols = [np.abs(g[:, x + 1] - g[:, x]).mean() for x in range(7, w - 1, 8)]
    rows = [np.abs(g[y + 1, :] - g[y, :]).mean() for y in range(7, h - 1, 8)]
    if not cols or not rows:
        return 0.0
    bnd = (float(np.mean(cols)) + float(np.mean(rows))) / 2.0
    ins = []
    for y in range(0, h - 8, 8):
        for x in range(0, w - 8, 8):
            blk = g[y : y + 8, x : x + 8]
            ins.append(np.abs(blk[:, 1:] - blk[:, :-1]).mean())
            ins.append(np.abs(blk[1:, :] - blk[:-1, :]).mean())
    inner = float(np.mean(ins)) if ins else 1e-8
    return float(bnd / (inner + 1e-8))


def ssim(a, b):
    """Global SSIM (grayscale, single channel)."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (va + vb + c2)
    return float(num / den)


def main() -> int:
    rows = {}
    for name, fname in VIDS.items():
        path = os.path.join(OUT, fname)
        if not os.path.isfile(path):
            print("MISSING", path)
            continue
        fr = read_gray(path)
        n = len(fr)
        if n < 4:
            print("SHORT", path)
            continue
        head_n = max(1, int(n * 0.15))
        body = fr[head_n : max(head_n + 1, int(n * 0.85))]
        hf = float(np.mean([hf_ratio(g) for g in body]))
        blk = float(np.mean([blockiness(g) for g in body]))
        rms = float(np.mean([g.std() for g in body]))
        mean = float(np.mean([g.mean() for g in body]))
        # temporal SSIM: consecutive body frames, plus first 3 (seam)
        body_ssim = float(
            np.mean([ssim(body[i], body[i - 1]) for i in range(1, len(body))])
        )
        seam_ssim = float(np.mean([ssim(fr[i], fr[i - 1]) for i in range(1, min(4, n))]))
        rows[name] = {
            "file": fname,
            "frames": n,
            "hf_ratio": round(hf, 5),
            "blockiness": round(blk, 4),
            "rms_contrast": round(rms, 2),
            "mean": round(mean, 2),
            "temporal_ssim_body": round(body_ssim, 4),
            "temporal_ssim_seam": round(seam_ssim, 4),
        }

    print(
        "\n%-12s %6s %7s %6s %6s %8s %8s"
        % ("run", "hf", "block", "rms", "mean", "ssim_b", "ssim_s")
    )
    for name, r in rows.items():
        print(
            "%-12s %6.5f %7.4f %6.2f %6.2f %8.4f %8.4f"
            % (
                name,
                r["hf_ratio"],
                r["blockiness"],
                r["rms_contrast"],
                r["mean"],
                r["temporal_ssim_body"],
                r["temporal_ssim_seam"],
            )
        )
    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print("\njson ->", JSON_OUT)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
