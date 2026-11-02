"""Portable traces, legacy adapters, atomic JSON writes, and content identities."""
import hashlib
import json
import math
import re
from pathlib import Path

SCHEMA = "ars-bench/1"
DATASETS = {"hotpotqa", "legalbenchrag", "telcodpr", "covidqa", "tatqa"}


def read(path):
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    if p.suffix == ".jsonl":
        if not isinstance(value, list):
            raise ValueError("JSONL output requires a list of records")
        with tmp.open("w", encoding="utf-8") as stream:
            for row in value:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    else:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def nonempty(row, keys, default=""):
    return next((row[k] for k in keys if row.get(k) not in (None, "")), default)


def rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "data", "details", "results", "samples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Expected a record list or records/data/details/results/samples wrapper")


def normalize_step(s):
    action = str(s.get("action") or "")
    query = s.get("search_query")
    match = re.search(r"search\[(.*?)\]", action, re.I | re.S)
    if not query and match:
        query = match.group(1)
    is_search = s.get("is_search", bool(query or match))
    if type(is_search) is not bool:
        raise ValueError("is_search must be boolean")
    out = {"thought": str(s.get("thought") or ""), "action": action,
           "search_query": query, "is_search": is_search}
    for k in ("observation", "source", "search_intent", "extracted_info", "reasoning_time_s", "reasoning_tokens", "token_count_method", "raw_documents", "defense"):
        if k in s:
            out[k] = s[k]
    documents = s.get("retrieved_documents", s.get("search_documents"))
    if isinstance(documents, list):
        out["retrieved_documents"] = documents
    flags = s.get("poisoned_flags")
    if isinstance(flags, list):
        if not all(type(flag) is bool for flag in flags):
            raise ValueError("poisoned_flags must contain booleans")
        out["poisoned_flags"] = flags
    return out


def normalize_records(payload):
    result = []
    seen = set()
    for r in rows(payload):
        rid = nonempty(r, ("id", "qid"), None)
        if rid is None:
            raise ValueError("Every record needs an id or qid; positional matching is not allowed")
        rid = str(rid)
        if rid in seen:
            raise ValueError(f"Duplicate record id: {rid}")
        seen.add(rid)
        steps = r.get("steps", r.get("step_stats", []))
        if not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps):
            raise ValueError(f"Invalid steps for {rid}")
        out = {"id": rid, "question": str(nonempty(r, ("question", "query"))),
               "reference": nonempty(r, ("reference", "correct answer", "correct_answer", "answer")),
               "prediction": str(nonempty(r, ("prediction", "llm_answer", "final_answer"))),
               "steps": [normalize_step(s) for s in steps]}
        for k in ("reasoning_time_s", "reasoning_tokens", "token_count_method", "target_answer", "attacker_intent", "error", "completion_status", "benchmark_record", "benchmark_trajectory", "benchmark_output", "generation_calls", "generation_warnings", "web_explorer", "n_calls", "n_badcalls"):
            if k in r:
                out[k] = r[k]
        if "benchmark_record" not in out and "step_stats" in r:
            out["benchmark_record"] = {"id": rid, "question": out["question"],
                "correct answer": out["reference"], "prediction": out["prediction"], "step_stats": r["step_stats"]}
        result.append(out)
    return result


def normalize_documents(payload):
    docs = payload.get("documents") if isinstance(payload, dict) else payload
    if not isinstance(docs, list):
        raise ValueError("Expected documents list or JSONL")
    out = []
    seen = set()
    for i, doc in enumerate(docs):
        if not isinstance(doc, dict):
            raise ValueError(f"Document {i} must be a JSON object")
        rid = str(doc.get("id", doc.get("_id", f"doc-{i}")))
        if rid in seen:
            rid = f"{rid}-{i}"
        seen.add(rid)
        d = {"id": rid, "title": str(doc.get("title", "")),
             "text": str(doc.get("text", doc.get("contents", ""))),
             "is_poisoned": True, "metadata": dict(doc.get("metadata", {}))}
        if "evaluations" in doc:
            d["evaluations"] = doc["evaluations"]
        out.append(d)
    return out


def bundle(clean, attacked, documents, experiment):
    ds = experiment.get("dataset")
    if ds not in DATASETS:
        raise ValueError(f"dataset must be one of {sorted(DATASETS)}")
    if not str(experiment.get("target_answer", "")).strip():
        raise ValueError("An explicit nonempty target_answer is required")
    result = {"schema_version": SCHEMA, "experiment": experiment,
            "documents": normalize_documents(documents),
            "clean": {"records": normalize_records(clean)},
            "attacked": {"records": normalize_records(attacked)}}
    for name, payload in (("clean", clean), ("attacked", attacked)):
        if isinstance(payload, dict):
            summary = payload.get("source_summary", payload.get("summary", payload.get("metrics")))
            if isinstance(summary, dict):
                result[name]["source_summary"] = summary
    return result


def load_bundle(path, clean=None):
    obj = read(path)
    paired = isinstance(obj, dict) and obj.get("schema_version") == SCHEMA
    trace = isinstance(obj, dict) and obj.get("artifact_version") == "ars-trace/1"
    if not paired and not trace:
        raise ValueError("Use import-results to convert legacy files to an ars-bench/1 bundle")
    attacked = obj["attacked"] if paired else obj
    if obj.get("run_status") == "partial" or obj.get("expected_record_count", len(rows(attacked))) != len(rows(attacked)):
        raise ValueError("Run is an incomplete checkpoint")
    if clean is None and not paired:
        raise ValueError("This inference result has no paired clean run; supply --clean <clean-result.json>")
    baseline = clean if clean is not None else obj["clean"]
    if isinstance(baseline, dict):
        if baseline.get("run_status") == "partial" or baseline.get("expected_record_count", len(rows(baseline))) != len(rows(baseline)):
            raise ValueError("Clean run is an incomplete checkpoint")
        baseline_experiment = baseline.get("experiment", {})
        for key in ("dataset", "architecture", "model", "seed"):
            if baseline_experiment.get(key) is not None and obj["experiment"].get(key) is not None and baseline_experiment[key] != obj["experiment"][key]:
                raise ValueError(f"Clean {key} differs from attacked configuration")
    out = bundle(baseline, attacked, obj.get("documents", []), obj["experiment"])
    if "evaluation_cache" in obj:
        out["evaluation_cache"] = obj["evaluation_cache"]
    return out


def finite(value, label, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{label} cannot be boolean")
    n = float(value)
    if not math.isfinite(n) or n < minimum:
        raise ValueError(f"Invalid {label}: {value}")
    return n
