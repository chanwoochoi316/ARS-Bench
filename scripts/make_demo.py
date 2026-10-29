"""Create explicitly synthetic, offline evaluation examples for installation checks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_core import fixture, FakeClient, FakeShield, FakeEmbedding
from ars_bench.evaluation import evaluate, export_report
from ars_bench.io import write

root = Path(__file__).resolve().parents[1] / "examples"
obj = fixture()
report = evaluate(obj, FakeClient(), FakeShield(), FakeEmbedding())
write(root / "synthetic_result.json", obj)
export_report(report, root / "synthetic_report.json")
print("Created synthetic test fixtures; these are not measured research results.")
