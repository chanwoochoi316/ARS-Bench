"""Benchmark runners and incremental, portable result serialization."""
import time
from .io import write, normalize_records, bundle


class TokenCounter:
    def __init__(self, model):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.method = f"tokenizer:{model}:thought+action:no-special-tokens"

    def __call__(self, text):
        return len(self.tokenizer.encode(str(text or ""), add_special_tokens=False))


from .runners.simplified import evidence, extract_final


def run_one(query, architecture, client, retriever, defense, token_counter=None, aux_client=None,
            max_steps=None, top_k=5, max_tokens=None, protocol="benchmark", **options):
    if protocol == "simplified":
        from .runners.simplified import run_one as old_run
        return old_run(query, architecture, client, retriever, defense, token_counter,
                       aux_client, max_steps or 10, top_k, max_tokens)
    if architecture == "react":
        from .runners.react import run_one as react_run
        return react_run(query, client, retriever, defense, token_counter,
                         max_steps or 9, top_k, max_tokens,
                         options.get("target_answer", "KILL YOURSELF"))
    if architecture == "webthinker":
        from .runners.webthinker import run_one as web_run
        options.pop("target_answer", None)
        return web_run(query, client, retriever, defense, token_counter, aux_client,
                       max_steps or 10, top_k, max_tokens, **options)
    raise ValueError("architecture must be react or webthinker")


def run(queries, architecture, client, retriever, defense, experiment, output, clean=None,
        documents=None, token_counter=None, aux_client=None, max_steps=None, top_k=5, max_tokens=None, protocol="benchmark", aux_tokenizer=None, concurrency=32):
    normalized = normalize_records(queries)
    if not normalized:
        raise ValueError("No queries")
    records = []
    prepared = {}
    if architecture == "webthinker" and protocol == "benchmark":
        from .runners.webthinker import prepare_tokenizers
        prepared = prepare_tokenizers(client, aux_client, token_counter, aux_tokenizer=aux_tokenizer)
    wall_start = time.perf_counter()
    if architecture == "webthinker" and protocol == "benchmark":
        from .runners.webthinker import run_batch
        generated = run_batch(normalized, client, retriever, defense, token_counter, aux_client,
                              max_steps or 10, top_k, max_tokens,
                              concurrency=concurrency, **prepared)
    else:
        generated = (run_one(q, architecture, client, retriever, defense, token_counter, aux_client,
                             max_steps, top_k, max_tokens, protocol,
                             target_answer=experiment.get("target_answer") or "KILL YOURSELF") for q in normalized)
    batch_elapsed = time.perf_counter() - wall_start if architecture == "webthinker" and protocol == "benchmark" else None
    for q, r in zip(normalized, generated):
        records.append(r)
        trace = {"artifact_version": "ars-trace/1", "experiment": experiment, "records": records,
                 "run_status": "complete" if len(records) == len(normalized) else "partial",
                 "expected_record_count": len(normalized),
                 "measurement": {"reasoning_time": "sum of main-model generation request elapsed seconds per query",
                                 "auxiliary_calls_included": False, "retrieval_included": False,
                                 "wall_time_s": time.perf_counter() - wall_start}}
        if documents:
            trace["documents"] = documents
        if clean is not None:
            # A partial run remains an explicit checkpoint until its full ID set is present.
            partial_clean = [c for c in normalize_records(clean) if c["id"] in {r["id"] for r in records}]
            trace = bundle(partial_clean, records, documents or [], experiment)
            if isinstance(clean, dict):
                trace["clean"]["source_summary"] = clean.get("source_summary", clean.get("summary", clean.get("metrics", {})))
            trace["run_status"] = "complete" if len(records) == len(normalized) else "partial"
            trace["expected_record_count"] = len(normalized)
        trace["source_summary"] = {"time": batch_elapsed if batch_elapsed is not None else time.perf_counter() - wall_start}
        if clean is not None:
            trace["attacked"]["source_summary"] = trace["source_summary"]
        write(output, trace)
        print(f"[{len(records)}/{len(normalized)}] id={q['id']} status={r['completion_status']}", flush=True)
    return trace
