# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
"""Metadata / documentation sync checks (zero GPU).

Run: python tests/test_metadata.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILED = []


def check(name, ok, detail=""):
    print(("  [OK]   " if ok else "  [FAIL] ") + name + (("  " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


def load_pyproject():
    path = os.path.join(ROOT, "pyproject.toml")
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    data = load_pyproject()
    project = data["project"]
    comfy = data["tool"]["comfy"]

    print("== 1. pyproject ==")
    check("name = comfyui-flash-attention4", project["name"] == "comfyui-flash-attention4", project["name"])
    check("version = 0.2.0", project["version"] == "0.2.0", project["version"])
    check("license 指向 LICENSE 文件", project["license"].get("file") == "LICENSE", str(project["license"]))
    check("requires-python >= 3.10", project["requires-python"] == ">=3.10", project["requires-python"])
    check("comfy PublisherId 存在", bool(comfy.get("PublisherId")))
    check("comfy requires-comfyui 存在", bool(comfy.get("requires-comfyui")))

    print("== 2. Version sync ==")
    init = {}
    with open(os.path.join(ROOT, "__init__.py"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                init["v"] = line.split("=", 1)[1].strip().strip('"').strip("'")
    check("__init__.__version__ == pyproject.version", init.get("v") == project["version"], str(init.get("v")))

    print("== 3. README facts ==")
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    for needle in (
        "152s",
        "140.2s",
        "640×832×124",
        "er_sde",
        "0.95",
        "BSD-3-Clause",
        "SecondNatureComputing/flash-attn-4-sm120",
        "flash4",
        "MODEL → MODEL",
    ):
        check("README 含 %s" % needle, needle in readme)

    print("== 4. Runtime files ==")
    for name in ("__init__.py", "nodes.py", "attention_fa4.py", "README.md", "pyproject.toml", "LICENSE"):
        check("%s 存在" % name, os.path.isfile(os.path.join(ROOT, name)))

    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
