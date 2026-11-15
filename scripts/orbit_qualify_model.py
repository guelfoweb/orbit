#!/usr/bin/env python3
"""Measure a hash-pinned experimental GGUF without production enablement."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from orbit.qualification.model_probe import main

if __name__ == "__main__":
    raise SystemExit(main(script=Path(__file__).resolve()))
