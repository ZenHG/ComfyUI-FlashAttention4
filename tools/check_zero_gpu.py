# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Zero-GPU gate for ComfyUI-FlashAttention4.

Runs, in order:
1. byte-compile every Python file (syntax gate);
2. tests/test_adapter.py;
3. tests/test_node_contract.py;
4. tests/test_metadata.py;
5. ruff check, when ruff is installed.

No ComfyUI, no CUDA, no FA4 kernel, no network.

Run: python tools/check_zero_gpu.py
Exit 0 = all green. Exit 1 = at least one red.
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def run(label, args):
    print("\n== %s ==" % label)
    proc = subprocess.run(args, cwd=ROOT)
    ok = proc.returncode == 0
    print("  %s %s" % ("[OK]" if ok else "[FAIL]", label))
    if not ok:
        FAILED.append(label)
    return ok


def main():
    print("FlashAttention4 zero-GPU gate")
    print("python: %s" % sys.executable)

    print("\n== 1. Byte-compile ==")
    compile_failed = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".git", "dist", ".ruff_cache")]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            try:
                py_compile.compile(path, doraise=True)
            except Exception as error:
                compile_failed.append("%s: %s" % (os.path.relpath(path, ROOT), error))
    for item in compile_failed:
        print("  [FAIL] %s" % item)
    if compile_failed:
        FAILED.append("byte-compile")
    else:
        print("  [OK] all .py files compile")

    for test in ("test_adapter.py", "test_node_contract.py", "test_metadata.py"):
        run("tests/%s" % test, [sys.executable, os.path.join(ROOT, "tests", test)])

    # ruff is optional; report SKIP, never count it as a pass.
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "."],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except Exception as error:
        print("\n== ruff ==\n  [SKIP] ruff unavailable: %s" % error)
    else:
        if proc.returncode == 0:
            print("\n== ruff ==\n  [OK] ruff check")
        elif "No module named" in (proc.stderr or ""):
            print("\n== ruff ==\n  [SKIP] ruff unavailable")
        else:
            print("\n== ruff ==\n  [FAIL] ruff check")
            print(proc.stdout)
            print(proc.stderr)
            FAILED.append("ruff")

    print("\n== Summary ==")
    if FAILED:
        print("  [FAIL] %d gate(s) red: %s" % (len(FAILED), ", ".join(FAILED)))
        return 1
    print("  [OK] zero-GPU gate green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
