"""One configuration drives clean -> attacks -> defended runs -> evaluation."""
from pathlib import Path
from .io import read, write, digest


def pipeline(config_path, target_override=None):
    from .cli import main, ROOT
    cfg = read(config_path)
    target = target_override or cfg["target_answer"]
    if not target.strip():
        raise ValueError("target_answer must not be empty")
    required = ("dataset", "architecture", "model", "base_url", "tokenizer")
    if any(not cfg.get(k) for k in required):
        raise ValueError(f"Pipeline config requires {required}")
    # Separate targets and configurations cannot overwrite one another's runs.
    identity = {**cfg, "target_answer": target}
    run_dir = ROOT / "runs" / cfg["dataset"] / (cfg.get("name", "experiment") + "-" + digest(identity)[:12])
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "clean.json").exists():
        raise ValueError(f"Run already exists: {run_dir}; change name/config or explicitly reuse artifacts with individual commands")

    def invoke(command):
        print("ARS stage:", command[0], flush=True)
        code = main([str(x) for x in command])
        if code:
            raise RuntimeError(f"Stage {command[0]} exited with code {code}; artifacts remain at {run_dir}")

    corpus = cfg.get("corpus", ROOT / "data" / cfg["dataset"] / "corpus.jsonl")
    index = cfg.get("index", ROOT / "indexes" / cfg["dataset"] / "e5")
    retriever = cfg.get("retriever", "dense")
    if retriever == "dense" and not (Path(index) / "manifest.json").exists():
        invoke(["index", "--corpus", corpus, "--output", index,
                "--model", cfg.get("retrieval_model", "intfloat/e5-large-v2"),
                "--device", cfg.get("retrieval_device", "cpu")])
    common = ["run", "--dataset", cfg["dataset"], "--architecture", cfg["architecture"],
              "--model", cfg["model"], "--base-url", cfg["base_url"], "--tokenizer", cfg["tokenizer"],
              "--retriever", retriever, "--corpus", corpus, "--index", index,
              "--retrieval-device", cfg.get("retrieval_device", "cpu"),
              "--seed", cfg.get("seed", 1), "--max-steps", cfg.get("max_steps", 10),
              "--top-k", cfg.get("top_k", 5)]
    if cfg.get("queries"):
        common += ["--queries", cfg["queries"]]
    if cfg["architecture"] == "webthinker":
        common += ["--aux-model", cfg["aux_model"], "--aux-base-url", cfg["aux_base_url"]]
    clean = run_dir / "clean.json"
    invoke(common + ["--output", clean])
    reports = []
    incomplete = False
    shared_cache = {}
    for method in cfg.get("attacks", ["kidnaprag", "ignore", "fake_completion", "poisonedrag", "paradox"]):
        artifact = run_dir / f"documents-{method}.json"
        invoke(["generate", "--clean", clean, "--attack", method, "--target-answer", target,
                "--base-url", cfg.get("generation_base_url", cfg["base_url"]),
                "--model", cfg.get("generation_model", "Qwen3-30B-A3B-Instruct-2507"),
                "--seed", cfg.get("seed", 1), "--scope", cfg.get("attack_scope", "all"),
                "--count", cfg.get("documents_per_search", 5), "--output", artifact])
        for defense in cfg.get("defenses", ["none", "ragdefender", "piguard", "spotlight"]):
            result = run_dir / f"result-{method}-{defense}.json"
            report = run_dir / f"report-{method}-{defense}.json"
            invoke(common + ["--attack-documents", artifact, "--defense", defense,
                             "--defense-device", cfg.get("defense_device", "cpu"), "--output", result])
            obj = read(result)
            obj["evaluation_cache"] = shared_cache
            write(result, obj)
            command = ["evaluate", result, "--output", report,
                       "--embedding-model", cfg.get("embedding_model", ROOT / "models/pds"),
                       "--device", cfg.get("evaluation_device", "cpu")]
            if cfg.get("judge_base_url"):
                command += ["--judge-base-url", cfg["judge_base_url"], "--judge-model", cfg.get("judge_model", "gpt-4o")]
            if cfg.get("azure_shield", False):
                command += ["--azure-shield"]
            invoke(command)
            shared_cache = read(report.with_suffix(".bundle.json")).get("evaluation_cache", {})
            reports.append(report)
            incomplete = incomplete or not read(report)["complete"]
    invoke(["aggregate", *reports, "--output", run_dir / "summary.csv"])
    print(f"Results: {run_dir}")
    return 2 if incomplete else 0
