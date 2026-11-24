# -*- coding: utf-8 -*-
"""Local A/B quality check: sharpness / seam / motion for the T3 runs.

Zero GPU. Reads the mp4s produced by the T3 matrix and prints a table plus a
contact sheet PNG for human review.

Run: COMFYUI_OUTPUT=/path/to/ComfyUI/output python tools/ab_quality.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

OUT = os.environ.get("COMFYUI_OUTPUT")
if not OUT:
    raise SystemExit("set COMFYUI_OUTPUT=/path/to/ComfyUI/output")
SHEET = os.environ.get("AB_SHEET", os.path.join(OUT, "ab_contact_sheet.png"))
JSON_OUT = os.environ.get("AB_JSON_OUT", os.path.join(OUT, "ab_quality.json"))

VIDS = {
    "3step": "abT3_C2_4step_00001_.mp4",
    "6step": "abT3_C2_00001_.mp4",
    "6step_spark": "abT3_A0_00001_.mp4",
    "fa4_6step": "abT3_C1_cold_00001_.mp4",
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


def sharpness(gray):
    """(laplacian variance, mean gradient magnitude)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = float(np.sqrt(gx * gx + gy * gy).mean())
    return float(lap.var()), grad


def main() -> int:
    rows = {}
    tiles = []
    for name, fname in VIDS.items():
        path = os.path.join(OUT, fname)
        if not os.path.isfile(path):
            print("MISSING", path)
            continue
        frames = read_gray(path)
        n = len(frames)
        if n == 0:
            print("EMPTY", path)
            continue
        sharp = [sharpness(g) for g in frames]
        head_n = max(1, int(n * 0.15))
        body = sharp[head_n : max(head_n + 1, int(n * 0.85))]
        head = sharp[:head_n]
        diffs = [
            float(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)).mean())
            for i in range(1, n)
        ]
        seam = float(np.mean(diffs[:3])) if len(diffs) >= 3 else 0.0
        body_diff = float(np.mean(diffs[3:])) if len(diffs) > 3 else 0.0
        rows[name] = {
            "file": fname,
            "frames": n,
            "head_sharp": round(float(np.mean([x[1] for x in head])), 3),
            "body_sharp": round(float(np.mean([x[1] for x in body])), 3),
            "lapvar": round(float(np.mean([x[0] for x in sharp])), 1),
            "seam_diff": round(seam, 3),
            "body_diff": round(body_diff, 3),
            "seam_over_body": round(seam / body_diff, 3) if body_diff else None,
            "brightness": round(float(np.mean([g.mean() for g in frames])), 2),
        }
        # tile: frames 0, n/4, n/2, 3n/4 downscaled
        idxs = [0, n // 4, n // 2, (3 * n) // 4]
        tiles.append((name, [frames[i] for i in idxs]))

    print("\n%-12s %5s %9s %9s %8s %8s %8s %7s" % ("run", "n", "head", "body", "lapvar", "seam", "bodyd", "s/b"))
    for name, r in rows.items():
        print(
            "%-12s %5d %9.3f %9.3f %8.1f %8.3f %8.3f %7s"
            % (
                name,
                r["frames"],
                r["head_sharp"],
                r["body_sharp"],
                r["lapvar"],
                r["seam_diff"],
                r["body_diff"],
                r["seam_over_body"],
            )
        )

    if tiles:
        h, w = tiles[0][1][0].shape
        scale = 0.35
        tw, th = int(w * scale), int(h * scale)
        pad = 6
        label_h = 22
        cols = 4
        rows_n = len(tiles)
        sheet = np.full(
            (rows_n * (th + label_h + pad) + pad, cols * (tw + pad) + pad),
            30,
            dtype=np.uint8,
        )
        for r_i, (name, imgs) in enumerate(tiles):
            y0 = pad + r_i * (th + label_h + pad)
            for c_i, img in enumerate(imgs):
                x0 = pad + c_i * (tw + pad)
                small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                sheet[y0 + label_h : y0 + label_h + th, x0 : x0 + tw] = small
                cv2.putText(
                    sheet,
                    "%s f%d" % (name, c_i),
                    (x0 + 2, y0 + label_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    230,
                    1,
                    cv2.LINE_AA,
                )
        cv2.imwrite(SHEET, sheet)
        print("\ncontact sheet ->", SHEET, sheet.shape)

    with open(JSON_OUT, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
    print("json ->", JSON_OUT)
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
