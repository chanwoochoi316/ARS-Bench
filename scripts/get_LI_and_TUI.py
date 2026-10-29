#!/usr/bin/env python3
"""Paired paper LI/TUI, never inferred from aggregate wall time."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ars_bench.io import load_bundle
from ars_bench.metrics import pair_records, efficiency

p = argparse.ArgumentParser()
p.add_argument("result", type=Path)
args = p.parse_args()
try:
    b = load_bundle(args.result)
    pairs = pair_records(b["clean"]["records"], b["attacked"]["records"])
    missing = False
    for name, key in [("LI", "reasoning_time_s"), ("TUI", "reasoning_tokens")]:
        try:
            print(f"{name}: {100 * efficiency(pairs, key):.4f}%")
        except ValueError as exc:
            missing = True
            print(f"{name}: MISSING ({exc})")
    raise SystemExit(2 if missing else 0)
except (ValueError, OSError) as exc:
    print(f"Error: {exc}", file=sys.stderr)
    raise SystemExit(1)
