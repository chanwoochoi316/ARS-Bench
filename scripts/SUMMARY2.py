#!/usr/bin/env python3
"""Compatibility entry point: use the new result-bundle evaluator CLI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ars_bench.cli import main

raise SystemExit(main(["evaluate", *sys.argv[1:]]))
