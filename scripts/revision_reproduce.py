#!/usr/bin/env python3
"""Rebuild the binary paper from included numerical records.

The binary stage computes no model calls and performs no training.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_environment() -> dict[str, str]:
    expected = dict(
        line.strip().split("==", 1)
        for line in (ROOT / "requirements-lock.txt").read_text().splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    )
    actual = {name: importlib.metadata.version(name) for name in expected}
    differences = {name: [version, actual[name]] for name, version in expected.items() if actual[name] != version}
    if differences:
        raise SystemExit(f"Interpreter does not match requirements-lock.txt: {differences}")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["check", "binary"])
    args = parser.parse_args()
    packages = check_environment()
    if args.stage == "check":
        print("Pinned record-rebuild environment verified")
        return
    commands = [
        [sys.executable, "scripts/build_isdfs_binary.py"],
        ["tectonic", "-X", "compile", "latex/paper.tex", "--keep-logs", "--keep-intermediates"],
        [sys.executable, "scripts/validate_isdfs_binary.py"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    print("Completed record-based paper rebuild")


if __name__ == "__main__":
    main()
