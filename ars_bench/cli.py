import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from .io import read, write, normalize_records, normalize_documents, bundle, load_bundle, DATASETS

ROOT = Path(__file__).resolve().parent.parent


def endpoint_args(p, prefix=""):
    flag = f"{prefix}-" if prefix else ""
    p.add_argument(f"--{flag}base-url", default="http://localhost:8000/v1")
    p.add_argument(f"--{flag}model", required=not bool(prefix), default=None)
    p.add_argument(f"--{flag}key-env", default="OPENAI_API_KEY")


def parser():
    p = argparse.ArgumentParser(description="ARS-Bench: independent Agentic RAG attacks, defenses, and paper evaluation")
    sub = p.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="Generate attack documents from a previously recorded clean path")
    g.add_argument("--clean", required=True, type=Path)
    g.add_argument("--attack", required=True)
    g.add_argument("--target-answer", required=True)
    g.add_argument("--output", required=True, type=Path, help=".jsonl saves documents only; .json saves a self-contained attack artifact")
    g.add_argument("--base-url", default="http://localhost:8000/v1")
    g.add_argument("--model", default="Qwen3-30B-A3B-Instruct-2507")
    g.add_argument("--key-env", default="OPENAI_API_KEY")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--count", type=int, default=None, help="Defaults: KidnapRAG/Ignore/Fake Completion/PoisonedRAG=5; PARADOX=4")
    g.add_argument("--scope", choices=["all", "first"], default="all")
    g.add_argument("--max-tokens", type=int, default=2500)

    i = sub.add_parser("index", help="Build a portable dense index containing its document metadata")
    i.add_argument("--corpus", required=True, type=Path)
    i.add_argument("--output", required=True, type=Path)
    i.add_argument("--model", default="intfloat/e5-large-v2")
    i.add_argument("--device", default="cpu")
    i.add_argument("--batch-size", type=int, default=64)

    r = sub.add_parser("run", help="Run clean or attacked ReAct/WebThinker, with a retrieval defense")
    r.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    r.add_argument("--architecture", choices=["react", "webthinker"], required=True)
    r.add_argument("--queries", type=Path)
    r.add_argument("--corpus", type=Path, help="Clean document corpus; must match the dense index when supplied")
    r.add_argument("--index", type=Path)
    r.add_argument("--retriever", choices=["dense", "bm25"], default="dense")
    r.add_argument("--retrieval-model")
    r.add_argument("--retrieval-device", default="cpu")
    r.add_argument("--attack-documents", type=Path, help="Poisoned document JSONL/JSON or an ars-attack/1 artifact; omitted for clean execution")
    r.add_argument("--clean", type=Path, help="Clean inference result for paired evaluation (separate from --corpus)")
    r.add_argument("--attack", help="Attack method; inferred from document metadata when available")
    r.add_argument("--target-answer", help="Attack target; inferred from document metadata when available")
    r.add_argument("--attacker-intent", help="Optional explicit intent for evaluation")
    r.add_argument("--defense", choices=["none", "piguard", "ragdefender", "spotlight"], default="none")
    r.add_argument("--defense-model", type=Path)
    r.add_argument("--defense-device", default="cpu")
    r.add_argument("--defense-embedding-model", type=Path)
    r.add_argument("--tokenizer", help="Main tokenizer path/name (WebThinker chat rendering and optional exact token instrumentation)")
    r.add_argument("--max-steps", type=int, default=None, help="Benchmark defaults: ReAct=9, WebThinker=10")
    r.add_argument("--protocol", choices=["benchmark", "simplified"], default="benchmark")
    r.add_argument("--aux-tokenizer", help="Auxiliary tokenizer path/name; defaults to auxiliary model mapping")
    r.add_argument("--concurrency", type=int, default=32, help="WebThinker request concurrency, as in Benchmark")
    r.add_argument("--max-tokens", type=int)
    r.add_argument("--top-k", type=int, default=5)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--limit", type=int)
    r.add_argument("--output", required=True, type=Path)
    endpoint_args(r)
    endpoint_args(r, "aux")

    e = sub.add_parser("evaluate", help="Evaluate ten metrics from attacked results; use --clean to pair a separate clean run")
    e.add_argument("result", type=Path)
    e.add_argument("--clean", type=Path, help="Clean inference result, if not already paired in the attacked result")
    e.add_argument("--metric-profile", choices=["benchmark", "instrumented"], default="benchmark")
    e.add_argument("--document-limit", type=int, default=100, help="Benchmark IBS evaluates first 100 documents; -1 evaluates all")
    e.add_argument("--moderation-base-url", default="https://api.openai.com/v1")
    e.add_argument("--moderation-model", default="omni-moderation-latest")
    e.add_argument("--moderation-key-env", default="OPENAI_API_KEY")
    e.add_argument("--output", required=True, type=Path)
    e.add_argument("--enriched-output", type=Path, help="Bundle with expensive evaluation caches; default <report>.bundle.json")
    e.add_argument("--judge-base-url", default=os.environ.get("ARS_JUDGE_BASE_URL") or ("https://api.openai.com/v1" if os.environ.get("OPENAI_API_KEY") else None),
                   help="LLM judge endpoint; auto-enabled when OPENAI_API_KEY or ARS_JUDGE_BASE_URL is set")
    e.add_argument("--judge-model", default="gpt-4o")
    e.add_argument("--judge-key-env", default="OPENAI_API_KEY")
    e.add_argument("--azure-shield", action="store_true")
    e.add_argument("--offline", action="store_true", help="Use stored evaluation caches only; never call services or load models")
    e.add_argument("--shield-endpoint")
    e.add_argument("--shield-key-env", default="AZURE_CONTENT_SAFETY_KEY")
    e.add_argument("--embedding-model", help="PDS model name/path; defaults to models/pds if present, otherwise matching cached PDS only")
    e.add_argument("--device", default="cpu")
    e.add_argument("--tokenizer", help="Recount recorded Thought+Action text with this tokenizer")
    e.add_argument("--alpha", type=float, default=0.3)
    e.add_argument("--require-all", action="store_true", help="Return exit code 2 when any metric is missing")

    b = sub.add_parser("import-results", help="Convert clean/attack legacy files to a portable result bundle")
    b.add_argument("--clean", required=True, type=Path)
    b.add_argument("--attacked", required=True, type=Path)
    b.add_argument("--documents", type=Path)
    b.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    b.add_argument("--architecture", required=True)
    b.add_argument("--model", required=True)
    b.add_argument("--attack", required=True)
    b.add_argument("--target-answer", required=True)
    b.add_argument("--attacker-intent")
    b.add_argument("--defense", default="none")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--output", required=True, type=Path)

    v = sub.add_parser("validate", help="Validate IDs, clean pairing and metric readiness")
    v.add_argument("result", type=Path)
    v.add_argument("--clean", type=Path, help="Clean inference result for an unpaired attack trace")
    v.add_argument("--metric-profile", choices=["benchmark", "instrumented"], default="benchmark")

    a = sub.add_parser("aggregate", help="Collect report files into a single experiment table")
    a.add_argument("reports", type=Path, nargs="+")
    a.add_argument("--output", type=Path, required=True)

    f = sub.add_parser("defend", help="Apply one defense to a retrieved passage list")
    f.add_argument("--input", type=Path, required=True)
    f.add_argument("--query", required=True)
    f.add_argument("--method", choices=["none", "piguard", "ragdefender", "spotlight"], required=True)
    f.add_argument("--model-path", type=Path)
    f.add_argument("--embedding-model", type=Path)
    f.add_argument("--device", default="cpu")
    f.add_argument("--output", type=Path, required=True)

    s = sub.add_parser("serve", help="Start a user-selected model with vLLM's OpenAI-compatible server")
    s.add_argument("--model", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--gpus", default="0")
    s.add_argument("--tensor-parallel-size", type=int, default=1)
    s.add_argument("--max-model-len", type=int, default=30000)

    q = sub.add_parser("pipeline", help="Run clean -> generate -> attack -> evaluate using a JSON configuration")
    q.add_argument("--config", required=True, type=Path)
    q.add_argument("--target-answer", help="Override config's target without editing filenames or prompts")
    return p


def execute(args):
    from .client import Client
    if args.command == "index":
        from .retrieval import build_index
        build_index(args.corpus, args.output, args.model, args.device, args.batch_size)
    elif args.command == "generate":
        from .attacks import generate
        clean = read(args.clean)
        if isinstance(clean, dict) and clean.get("run_status") == "partial":
            raise ValueError("Cannot generate from an incomplete clean run")
        artifact = generate(clean, args.attack, args.target_answer,
                            Client(args.base_url, args.model, args.key_env, args.seed),
                            args.count, args.seed, args.scope, args.max_tokens)
        if isinstance(clean, dict) and "experiment" in clean:
            artifact["clean_experiment"] = clean["experiment"]
        write(args.output, artifact["documents"] if args.output.suffix == ".jsonl" else artifact)
        print(f"Generated {len(artifact['documents'])} documents; target={args.target_answer!r}")
    elif args.command == "run":
        from .agents import run, TokenCounter
        from .retrieval import DenseRetriever, BM25Retriever
        from .defenses import Defense
        if args.concurrency < 1 or (args.max_steps is not None and args.max_steps < 1):
            raise ValueError("concurrency and max-steps must be positive")
        if args.max_steps is None:
            args.max_steps = 9 if args.architecture == "react" and args.protocol == "benchmark" else 10
        from .attack_inputs import resolve_attack
        if args.attack_documents:
            attack = resolve_attack(read(args.attack_documents), args.attack, args.target_answer, args.attacker_intent)
        else:
            if args.clean or args.attack or args.target_answer or args.attacker_intent:
                raise ValueError("--clean/--attack/--target-answer/--attacker-intent require --attack-documents")
            attack = None
        clean = read(args.clean) if args.clean else (attack["clean"] if attack else None)
        if isinstance(clean, dict) and clean.get("run_status") == "partial":
            raise ValueError("Cannot pair with an incomplete clean run")
        queries = (read(args.queries) if args.queries else
                   normalize_records(clean) if clean is not None else
                   read(ROOT / "data" / args.dataset / "queries.json"))
        if args.limit:
            if args.limit <= 0:
                raise ValueError("--limit must be positive")
            queries = normalize_records(queries)[:args.limit]
        docs = attack["documents"] if attack else []
        if clean is not None:
            from .metrics import pair_records
            pair_records(normalize_records(clean), normalize_records(queries))
            cm = (clean.get("experiment", {}) if args.clean and isinstance(clean, dict) else attack.get("clean_experiment", {}))
            for key, current in [("dataset", args.dataset), ("architecture", args.architecture), ("model", args.model), ("seed", args.seed)]:
                if key in cm and cm[key] is not None and cm[key] != current:
                    raise ValueError(f"Clean {key} differs from attacked configuration")
        retriever = (BM25Retriever(args.corpus or ROOT / "data" / args.dataset / "corpus.jsonl", attack_documents=docs)
                     if args.retriever == "bm25" else
                     DenseRetriever(args.index or ROOT / "indexes" / args.dataset / "e5", args.retrieval_model,
                                    args.retrieval_device, docs, corpus=args.corpus))
        defense = Defense(args.defense, str(args.defense_model) if args.defense_model else None,
                          args.defense_device, str(args.defense_embedding_model) if args.defense_embedding_model else None)
        experiment = {"dataset": args.dataset, "architecture": args.architecture, "model": args.model,
                      "protocol": args.protocol, "concurrency": args.concurrency,
                      "attack_documents_source": str(args.attack_documents) if args.attack_documents else None,
                      "clean_source": str(args.clean) if args.clean else None,
                      "attack_method": attack["attack_method"] if attack else "clean",
                      "target_answer": attack["target_answer"] if attack else "",
                      "attacker_intent": attack["attacker_intent"] if attack else "",
                      "defense_method": args.defense, "seed": args.seed, "retriever": args.retriever,
                      "top_k": args.top_k, "max_steps": args.max_steps,
                      "max_tokens": args.max_tokens or (100 if args.architecture == "react" else 4096),
                      "aux_tokenizer": args.aux_tokenizer,
                      "baseline_protocol": cm.get("protocol", "legacy/unknown") if clean is not None else None,
                      "retrieval_model": getattr(getattr(retriever, "encoder", None), "model_name", "bm25"),
                      "aux_model": args.aux_model, "tokenizer": args.tokenizer,
                      "reasoning_latency_scope": "main-model request elapsed time, excluding retrieval and auxiliary calls"}
        aux = Client(args.aux_base_url, args.aux_model, args.aux_key_env, args.seed) if args.aux_model else None
        run(queries, args.architecture, Client(args.base_url, args.model, args.key_env, args.seed),
            retriever, defense, experiment, args.output, clean, docs,
            TokenCounter(args.tokenizer) if args.tokenizer else None, aux, args.max_steps, args.top_k, args.max_tokens,
            protocol=args.protocol, aux_tokenizer=args.aux_tokenizer, concurrency=args.concurrency)
    elif args.command == "import-results":
        from .attacks import canonical_method, intent
        from .metrics import pair_records
        method = canonical_method(args.attack)
        experiment = {"dataset": args.dataset, "architecture": args.architecture, "model": args.model,
                      "attack_method": method, "target_answer": args.target_answer,
                      "attacker_intent": args.attacker_intent or intent(method, args.target_answer),
                      "defense_method": args.defense, "seed": args.seed, "imported": True}
        obj = bundle(read(args.clean), read(args.attacked), read(args.documents) if args.documents else [], experiment)
        # The explicitly selected target takes precedence over stale legacy target fields.
        conflicts = 0
        for record in obj["clean"]["records"] + obj["attacked"]["records"]:
            if record.get("target_answer") and record["target_answer"] != args.target_answer:
                conflicts += 1
            record.pop("target_answer", None)
            record.pop("attacker_intent", None)
        if conflicts:
            experiment["legacy_target_fields_overridden"] = conflicts
        pair_records(obj["clean"]["records"], obj["attacked"]["records"])
        write(args.output, obj)
        print("Imported. Legacy aggregate wall times are not converted into per-query reasoning times.")
    elif args.command == "evaluate":
        from .evaluation import evaluate, export_report
        from .judges import AzureShield, OpenAIModeration
        from .metrics import Embeddings
        from .agents import TokenCounter
        obj = load_bundle(args.result, read(args.clean) if args.clean else None)
        enriched = args.enriched_output or args.output.with_suffix(".bundle.json")
        if enriched.resolve() == args.output.resolve():
            raise ValueError("Report and enriched bundle must have different paths")
        judge = Client(args.judge_base_url, args.judge_model, args.judge_key_env) if args.judge_base_url and not args.offline else None
        shield_enabled = args.azure_shield or (os.environ.get("AZURE_CONTENT_SAFETY_ENDPOINT") and os.environ.get(args.shield_key_env))
        if args.metric_profile == "benchmark":
            shield = OpenAIModeration(args.moderation_base_url, args.moderation_model, args.moderation_key_env) if not args.offline else None
        else:
            shield = AzureShield(args.shield_endpoint, args.shield_key_env) if shield_enabled and not args.offline else None
        embedding_model = args.embedding_model or (str(ROOT / "models/pds") if (ROOT / "models/pds").is_dir() else None)
        embedder = Embeddings(embedding_model, args.device) if embedding_model and not args.offline else None
        report = evaluate(obj, judge, shield, embedder, args.judge_model, args.alpha,
                          TokenCounter(args.tokenizer) if args.tokenizer else None, lambda obj: write(enriched, obj),
                          profile=args.metric_profile, document_limit=args.document_limit)
        export_report(report, args.output)
        for name, value in report["metrics"].items():
            print(f"{name}: {value['value']:.4f}" if value["status"] == "ok" else f"{name}: MISSING ({value['reason']})")
        return 2 if args.require_all and not report["complete"] else 0
    elif args.command == "validate":
        from .metrics import pair_records
        from . import metrics as selected
        if args.metric_profile == "benchmark":
            from . import benchmark_metrics as selected
        exposure, efficiency = selected.exposure, selected.efficiency
        obj = load_bundle(args.result, read(args.clean) if args.clean else None)
        pairs = pair_records(obj["clean"]["records"], obj["attacked"]["records"])
        out = {"schema": "ok", "paired_queries": len(pairs), "attack_documents": len(obj["documents"]), "metric_profile": args.metric_profile, "readiness": {}}
        out["missing_reference_ids"] = [c["id"] for c, a in pairs if not str(c.get("reference", "")).strip()]
        for name, fn in [("retrieval", lambda: exposure(obj["attacked"]["records"])),
                         ("LI", lambda: efficiency(obj if args.metric_profile == "benchmark" else pairs, "reasoning_time_s")),
                         ("TUI", lambda: efficiency(obj if args.metric_profile == "benchmark" else pairs, "reasoning_tokens"))]:
            try:
                fn(); out["readiness"][name] = "ready"
            except ValueError as exc:
                out["readiness"][name] = str(exc)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif args.command == "aggregate":
        import csv
        from .metrics import METRICS
        from .benchmark_metrics import METRICS as BENCHMARK_METRICS
        metric_columns = list(BENCHMARK_METRICS) + ["SBS"]
        columns = ["dataset", "architecture", "model", "attack_method", "target_answer", "defense_method", "seed", "metric_profile"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns + metric_columns + [k + "_status" for k in metric_columns])
            writer.writeheader()
            for path in args.reports:
                obj = read(path)
                if obj.get("schema_version") != "ars-report/1":
                    raise ValueError(f"Not an ARS report: {path}")
                row = {k: obj["experiment"].get(k) for k in columns}
                row["metric_profile"] = obj.get("metric_profile", "instrumented")
                for k in metric_columns:
                    value = obj["metrics"].get(k, {"value": None, "status": "not_applicable"})
                    row[k] = value["value"]
                    row[k + "_status"] = value["status"]
                writer.writerow(row)
    elif args.command == "defend":
        from .defenses import Defense
        payload = read(args.input)
        docs = payload.get("documents", []) if isinstance(payload, dict) else payload
        kept, info = Defense(args.method, str(args.model_path) if args.model_path else None,
                             args.device, str(args.embedding_model) if args.embedding_model else None).apply(args.query, docs)
        write(args.output, {"query": args.query, "raw_documents": docs, "documents": kept, "defense": info})
    elif args.command == "serve":
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
        return subprocess.call([sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
                                "--served-model-name", args.name, "--port", str(args.port),
                                "--tensor-parallel-size", str(args.tensor_parallel_size),
                                "--max-model-len", str(args.max_model_len)], env=env)
    elif args.command == "pipeline":
        from .pipeline import pipeline
        return pipeline(args.config, args.target_answer)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return execute(args)
    except (ValueError, OSError, ImportError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
