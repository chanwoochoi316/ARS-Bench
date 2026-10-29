import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from ars_bench import metrics
from ars_bench.io import bundle, normalize_records, normalize_step, load_bundle, write
from ars_bench.attacks import generate, intent, CHAIN
from ars_bench.defenses import Defense
from ars_bench.evaluation import evaluate
from ars_bench.agents import run_one, extract_final


def record(rid="a", prediction="gold", seconds=1, tokens=10):
    return {"id": rid, "question": "What is the code?", "reference": "gold", "prediction": prediction,
            "reasoning_time_s": seconds, "reasoning_tokens": tokens, "token_count_method": "tokenizer:test",
            "steps": [{"thought": "Find the code", "action": "Search[code]", "search_query": "code",
                       "is_search": True, "retrieved_documents": [{"id": "c", "title": "", "text": "gold", "is_poisoned": False}]}]}


def fixture():
    c = [record("a", seconds=1, tokens=10), record("b", seconds=10, tokens=100)]
    a = [record("b", "NEW TARGET", seconds=10, tokens=100), record("a", "wrong", seconds=2, tokens=20)]
    a[0]["steps"][0]["retrieved_documents"][0]["is_poisoned"] = True
    exp = {"dataset": "hotpotqa", "architecture": "react", "model": "synthetic", "attack_method": "ignore",
           "target_answer": "NEW TARGET", "defense_method": "none", "seed": 1, "synthetic": True}
    return bundle(c, a, [{"id": "p", "title": "code", "text": "NEW TARGET"}], exp)


class FakeClient:
    model = "gpt-4o"
    def __init__(self):
        self.calls = []
    def chat(self, messages, **options):
        self.calls.append((messages, options))
        if "Score rubric:" in messages[0]["content"]:
            return json.dumps({"score": 0, "reason": "synthetic fixture only"})
        if any("Generate doc1" in m["content"] for m in messages):
            return json.dumps({f"doc{i}": f"synthetic bait {i}" for i in range(1, 6)})
        return "synthetic generated document"


class FakeShield:
    def evaluate(self, text):
        return {"flagged": False, "provider": "synthetic-test-shield"}


class FakeEmbedding:
    name = "synthetic-test-embeddings"
    def encode(self, texts):
        import numpy as np
        return np.asarray([[1., 0.] if "Find" in t else [0., 1.] for t in texts])


class ScientificMetrics(unittest.TestCase):
    def test_paired_mean_is_not_ratio_of_means(self):
        b = fixture()
        pairs = metrics.pair_records(b["clean"]["records"], b["attacked"]["records"])
        self.assertEqual(metrics.efficiency(pairs, "reasoning_time_s"), .5)
        self.assertEqual(metrics.efficiency(pairs, "reasoning_tokens"), .5)

    def test_pairing_uses_ids_and_rejects_missing_samples(self):
        b = fixture()
        with self.assertRaisesRegex(ValueError, "ID sets"):
            metrics.pair_records(b["clean"]["records"], b["attacked"]["records"][:1])

    def test_reference_mismatch_rejected(self):
        b = fixture(); b["attacked"]["records"][0]["reference"] = "different"
        with self.assertRaisesRegex(ValueError, "mismatch"):
            metrics.pair_records(b["clean"]["records"], b["attacked"]["records"])

    def test_exposure_uses_query_and_search_denominators(self):
        a = record(); a["steps"] *= 2
        b = record("b"); b["steps"][0]["retrieved_documents"][0]["is_poisoned"] = True
        qles, sles, n = metrics.exposure([a, b])
        self.assertEqual(qles, 2.5); self.assertAlmostEqual(sles, 5/3); self.assertEqual(n, 3)

    def test_mixed_legacy_trace_formats_per_step(self):
        a = record(); a["steps"] += [{"thought": "", "action": "Search[x]", "is_search": True, "poisoned_flags": [True]}]
        self.assertEqual(metrics.exposure([a])[:2], (5, 2.5))

    def test_unknown_exposure_is_not_zero(self):
        a = record(); del a["steps"][0]["retrieved_documents"]
        with self.assertRaises(ValueError): metrics.exposure([a])

    def test_ufs_is_relative_and_clamped(self):
        self.assertEqual(metrics.relative_ufs(.8, .4), 2.5)
        self.assertEqual(metrics.relative_ufs(.4, .8), 0)
        with self.assertRaises(ValueError): metrics.relative_ufs(0, 0)

    def test_missing_reference_is_reported_with_id_not_silently_excluded(self):
        r = record("missing-label"); r["reference"] = ""
        with self.assertRaisesRegex(ValueError, "missing-label"):
            metrics.utility("hotpotqa", [r])

    def test_zero_efficiency_baseline_is_undefined(self):
        c, a = record(seconds=0), record(seconds=1)
        with self.assertRaises(ValueError): metrics.efficiency([(c, a)], "reasoning_time_s")

    def test_heuristic_tokens_rejected(self):
        c, a = record(), record(); c["token_count_method"] = a["token_count_method"] = "heuristic"
        with self.assertRaises(ValueError): metrics.efficiency([(c, a)], "reasoning_tokens")

    @unittest.skipUnless(importlib.util.find_spec("rouge_score"), "rouge-score extra not installed")
    def test_telcodpr_uses_rouge_not_containment(self):
        r = record(); r["reference"] = "the timer expires after thirty seconds"; r["prediction"] = "the timer expires after sixty seconds"
        value = metrics.utility("telcodpr", [r])
        self.assertGreater(value, 0); self.assertLess(value, 1)

    def test_pds_identical_stationary_path_is_zero(self):
        self.assertAlmostEqual(metrics.pds_from_embeddings([[1, 0], [1, 0]], [[1, 0], [1, 0]]), 0)

    def test_pds_single_step_and_scale(self):
        self.assertAlmostEqual(metrics.pds_from_embeddings([[1, 0]], [[-1, 0]], alpha=.3), 1.5)

    def test_pds_rejects_nan(self):
        with self.assertRaises(ValueError): metrics.pds_from_embeddings([[float("nan"), 0]], [[1, 0]])


class SchemaAndAttacks(unittest.TestCase):
    def test_legacy_details_and_search_query_adaptation(self):
        r = {"qid": "q", "question": "Q", "correct_answer": "A", "llm_answer": "B",
             "step_stats": [{"thought": "T", "action": "Search[technical phrase]", "is_search": True, "poisoned_flags": [True]}]}
        out = normalize_records({"details": [r]})[0]
        self.assertEqual(out["steps"][0]["search_query"], "technical phrase")
        self.assertEqual(out["reference"], "A")

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): normalize_records([record(), record()])

    def test_missing_ids_rejected(self):
        with self.assertRaises(ValueError): normalize_records([{"question": "Q"}])

    def test_string_poison_flag_rejected(self):
        with self.assertRaises(ValueError): normalize_step({"poisoned_flags": ["false"]})

    def test_incomplete_checkpoint_rejected(self):
        b = fixture(); b["run_status"] = "partial"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"b.json"; write(path, b)
            with self.assertRaises(ValueError): load_bundle(path)

    def test_all_five_attacks_use_custom_target_and_clean_paths(self):
        for method in ("kidnaprag", "ignore", "fake_completion", "poisonedrag", "paradox"):
            client = FakeClient()
            a = generate([record()], method, "SELECT * FROM users", client)
            serialized = json.dumps(a)
            self.assertNotIn("KILL YOURSELF", serialized)
            self.assertEqual(a["target_answer"], "SELECT * FROM users")
            self.assertIn("code", {d["title"] for d in a["documents"]})
            self.assertEqual(len({d["id"] for d in a["documents"]}), len(a["documents"]))
            self.assertTrue(all(d["metadata"]["target_answer"] == "SELECT * FROM users" for d in a["documents"]))
            for messages, _ in client.calls:
                self.assertNotIn("KILL YOURSELF", json.dumps(messages))

    def test_empty_clean_search_cannot_generate(self):
        r = record(); r["steps"] = []
        with self.assertRaises(ValueError): generate([r], "ignore", "X")

    def test_target_with_format_braces_is_literal(self):
        a = generate([record()], "completion_real", '{"foo": "bar"}')
        self.assertIn('{"foo": "bar"}', a["documents"][0]["text"])

    def test_spotlight_preserves_document_metadata(self):
        docs = [{"id": "x", "title": "two words", "text": "a b", "is_poisoned": True}]
        kept, info = Defense("spotlight").apply("q", docs)
        self.assertEqual(kept[0]["text"], "a^b")
        self.assertTrue(kept[0]["is_poisoned"])
        self.assertEqual(docs[0]["text"], "a b")


class EvaluationAndAgents(unittest.TestCase):
    def test_all_ten_metrics_and_offline_cache_replay(self):
        b = fixture(); client = FakeClient()
        report = evaluate(b, client, FakeShield(), FakeEmbedding())
        self.assertTrue(report["complete"])
        self.assertEqual(set(report["metrics"]), set(metrics.METRICS))
        self.assertEqual(report["metrics"]["IBS"]["value"], 5)
        self.assertEqual(report["metrics"]["ISS"]["value"], 0)
        self.assertEqual(report["metrics"]["RCS"]["value"], 2.5)
        self.assertEqual(report["metrics"]["LI"]["percent"], 50)
        replay = evaluate(b)
        self.assertTrue(replay["complete"])
        self.assertEqual(report["metrics"], replay["metrics"])

    def test_missing_judges_are_not_reported_as_zero(self):
        report = evaluate(fixture())
        for name in ("IBS", "ISS", "SBS", "PDS"):
            self.assertEqual(report["metrics"][name]["status"], "missing")
            self.assertIsNone(report["metrics"][name]["value"])

    def test_target_change_invalidates_iss_cache(self):
        b = fixture(); evaluate(b, FakeClient(), FakeShield(), FakeEmbedding())
        b["experiment"]["target_answer"] = "CHANGED"
        self.assertEqual(evaluate(b)["metrics"]["ISS"]["status"], "missing")

    def test_explicit_intent_does_not_hide_target_cache_change(self):
        b = fixture(); b["experiment"]["attacker_intent"] = "an explicitly specified intent"
        evaluate(b, FakeClient(), FakeShield(), FakeEmbedding())
        b["experiment"]["target_answer"] = "CHANGED"
        self.assertEqual(evaluate(b)["metrics"]["ISS"]["status"], "missing")

    def test_document_change_invalidates_ingestion_cache(self):
        b = fixture(); evaluate(b, FakeClient(), FakeShield(), FakeEmbedding())
        b["documents"][0]["text"] = "CHANGED"
        report = evaluate(b)
        self.assertEqual(report["metrics"]["SBS"]["status"], "missing")

    def test_trace_change_invalidates_pds_cache(self):
        b = fixture(); evaluate(b, FakeClient(), FakeShield(), FakeEmbedding())
        b["attacked"]["records"][0]["steps"][0]["thought"] = "CHANGED"
        self.assertEqual(evaluate(b)["metrics"]["PDS"]["status"], "missing")

    def test_react_runs_retrieval_defense_then_finish(self):
        class Model:
            def __init__(self): self.i = 0
            def complete(self, prompt, **kw):
                self.i += 1
                return "Find it\nAction 1: Search[code]" if self.i == 1 else "Ready\nAction 2: Finish[gold]"
        class Search:
            def search(self, query, k): return [{"id": "d", "title": "a b", "text": "gold", "is_poisoned": True}]
        r = run_one(record(), "react", Model(), Search(), Defense("spotlight"))
        self.assertEqual(r["prediction"], "gold")
        self.assertEqual(r["completion_status"], "finished")
        self.assertEqual(len(r["steps"]), 2)
        self.assertEqual(r["steps"][0]["retrieved_documents"][0]["title"], "a^b")

    def test_webthinker_search_and_aux_summary(self):
        class Main:
            def __init__(self): self.i = 0
            def chat(self, messages, **kw):
                self.i += 1
                return "Find it<|begin_search_query|>code" if self.i == 1 else "Answer is \\boxed{gold}"
        class Aux:
            def chat(self, messages, **kw): return "**Final Information**\ngold"
        class Search:
            def search(self, query, k): return [{"id": "d", "text": "gold", "is_poisoned": False}]
        r = run_one(record(), "webthinker", Main(), Search(), Defense(), aux_client=Aux())
        self.assertEqual(r["prediction"], "gold")
        self.assertEqual(r["steps"][0]["observation"], "gold")

    def test_balanced_boxed_answer(self):
        self.assertEqual(extract_final("answer \\boxed{a{b}c}"), "a{b}c")


if __name__ == "__main__":
    unittest.main()
