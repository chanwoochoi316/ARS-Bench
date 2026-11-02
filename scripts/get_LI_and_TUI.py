#!/usr/bin/env python3
"""Benchmark LI/TUI from original total wall time and text-token heuristic."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ars_bench.io import load_bundle
from ars_bench.metrics import pair_records
from ars_bench import benchmark_metrics, metrics

p = argparse.ArgumentParser()
p.add_argument("result", type=Path)
p.add_argument("--metric-profile", choices=["benchmark", "instrumented"], default="benchmark")
args = p.parse_args()
try:
    b = load_bundle(args.result)
    pairs = pair_records(b["clean"]["records"], b["attacked"]["records"])
    source = b if args.metric_profile == "benchmark" else pairs
    efficiency = (benchmark_metrics if args.metric_profile == "benchmark" else metrics).efficiency
    missing = False
    for name, key in [("LI", "reasoning_time_s"), ("TUI", "reasoning_tokens")]:
        try:
            print(f"{name}: {100 * efficiency(source, key):.4f}%")
        except ValueError as exc:
            missing = True
            print(f"{name}: MISSING ({exc})")
    raise SystemExit(2 if missing else 0)
except (ValueError, OSError) as exc:
    print(f"Error: {exc}", file=sys.stderr)
    raise SystemExit(1)
