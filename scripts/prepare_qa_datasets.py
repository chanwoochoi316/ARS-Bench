#!/usr/bin/env python3
"""Build COVID-QA and TAT-QA in ARS-Bench's corpus/query file format.

Requires pyarrow for the Hugging Face parquet. Never overwrites dataset folders.
Run from the ARS-Bench root; --source-cache can reuse verified downloaded inputs.
"""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile
from datetime import datetime, timezone
import urllib.request

SOURCES = {
    "covid_data.parquet": "https://huggingface.co/datasets/deepset/covid_qa_deepset/resolve/main/covid_qa_deepset/train-00000-of-00001.parquet",
    "covid_readme.md": "https://huggingface.co/datasets/deepset/covid_qa_deepset/raw/main/README.md",
    "tat_train.json": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json",
    "tat_dev.json": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json",
    "tat_test_gold.json": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_test_gold.json",
    "tat_readme.md": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/README.md",
}


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fetch(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        p = cache / name
        if p.exists():
            if p.stat().st_size == 0:
                raise ValueError(f"Empty cached source: {p}")
            continue
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    b = response.read()
                tmp = p.with_suffix(p.suffix + ".tmp")
                tmp.write_bytes(b)
                tmp.replace(p)
                break
            except Exception:
                if attempt == 2:
                    raise


def windows(text, size=3000, overlap=1000):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cuts = list(re.finditer(r"\s", text[start + size - 300:end]))
            if cuts:
                end = start + size - 300 + cuts[-1].end()
        yield start, end
        if end == len(text):
            break
        following = end - overlap
        while following > start + 1 and not text[following - 1].isspace():
            following -= 1
        if following <= start:
            raise ValueError("Chunking did not advance")
        start = following


def covid(cache):
    import pyarrow.parquet as pq
    raw = pq.read_table(cache / "covid_data.parquet").to_pylist()
    if len(raw) != 2019:
        raise ValueError(f"Unexpected COVID-QA row count: {len(raw)}")
    contexts, docs, queries, annotations = {}, [], [], {}
    repairs, multiple = 0, 0
    for row in raw:
        did, text = str(row["document_id"]), row["context"]
        if did in contexts:
            if contexts[did]["text"] != text:
                raise ValueError(f"Conflicting COVID-QA context: {did}")
            continue
        title = text.splitlines()[0].strip()
        match = re.search(r"https?://[^\s]+", text)
        chunks = []
        for i, (start, end) in enumerate(windows(text)):
            doc = {"_id": f"covidqa/{did}##{i}", "title": title, "text": text[start:end],
                   "metadata": {"dataset": "covidqa", "source_document_id": did,
                                "source_split": "train", "chunk_idx": i,
                                "char_start": start, "char_end": end,
                                "url": match.group(0) if match else ""}}
            chunks.append(doc)
            docs.append(doc)
        contexts[did] = {"text": text, "chunks": chunks}
    for row in raw:
        qid, did = str(row["id"]), str(row["document_id"])
        answers = row["answers"]
        if row["is_impossible"] or not answers["text"]:
            raise ValueError(f"Unanswerable COVID-QA row: {qid}")
        text = row["context"]
        aligned, relevant = [], set()
        for answer, original in zip(answers["text"], answers["answer_start"]):
            if not answer.strip():
                raise ValueError(f"Empty COVID-QA answer: {qid}")
            positions = [m.start() for m in re.finditer(re.escape(answer), text)]
            if not positions:
                raise ValueError(f"Answer absent from context: {qid}")
            actual = min(positions, key=lambda p: (abs(p - original), p))
            repairs += actual != original
            multiple += len(positions) > 1
            aligned.append(actual)
            covered = [d["_id"] for d in contexts[did]["chunks"]
                       if d["metadata"]["char_start"] <= actual
                       and actual + len(answer) <= d["metadata"]["char_end"]]
            if not covered:
                raise ValueError(f"Answer crosses all chunk boundaries: {qid}")
            relevant.update(covered)
        queries.append({"id": qid, "question": row["question"], "reference": answers["text"][0]})
        if qid in annotations:
            raise ValueError(f"Duplicate COVID-QA question: {qid}")
        annotations[qid] = {"source_split": "train", "source_document_id": did,
                            "corpus_ids": sorted(relevant), "gold_answers": answers["text"],
                            "original_answer_start": answers["answer_start"],
                            "aligned_answer_start": aligned, "is_impossible": row["is_impossible"]}
    detail = {"source_documents": len(contexts), "chunk_size_chars": 3000,
              "chunk_overlap_chars": 1000, "source_answer_offsets_repaired": repairs,
              "answers_with_multiple_occurrences": multiple,
              "offset_policy": "Preserve original offsets; choose nearest exact text occurrence for alignment. Ties use first occurrence. No answer text is changed."}
    return docs, {"train": queries}, annotations, detail, "train"


def reference(answer, scale):
    values = answer if isinstance(answer, list) else [answer]
    result = []
    for value in values:
        text = str(value).strip()
        # Units only decorate numeric answers. Source text spans stay verbatim.
        if scale and re.fullmatch(r"[-+]?\d[\d,.]*(?:\.\d+)?", text):
            text += "%" if scale == "percent" else " " + scale
        result.append(text)
    return "; ".join(result)


def tat(cache):
    docs, splits, annotations = [], {}, {}
    all_doc_ids = set()
    counts = {}
    for split, filename in (("train", "tat_train.json"), ("dev", "tat_dev.json"), ("test", "tat_test_gold.json")):
        raw = load(cache / filename)
        queries = []
        counts[split] = len(raw)
        for item in raw:
            table = item["table"]
            uid = str(table["uid"])
            did = "tatqa/" + uid
            if did in all_doc_ids:
                raise ValueError(f"Duplicate TAT-QA context: {did}")
            all_doc_ids.add(did)
            paragraphs = sorted(item["paragraphs"], key=lambda p: p["order"])
            stream = io.StringIO(newline="")
            csv.writer(stream, delimiter="\t", lineterminator="\n").writerows(table["table"])
            table_text = stream.getvalue()
            if list(csv.reader(io.StringIO(table_text), delimiter="\t")) != table["table"]:
                raise ValueError(f"Lossy table serialization: {did}")
            text = "Table (tab-separated cells):\n" + table_text + "\nParagraphs:\n" + "\n\n".join(
                f"[{p['order']}] {p['text']}" for p in paragraphs)
            docs.append({"_id": did,
                         "title": (paragraphs[0]["text"][:240] if paragraphs else "Financial report") + " [TAT-QA " + uid + "]",
                         "text": text,
                         "metadata": {"dataset": "tatqa", "source_split": split,
                                      "table_uid": uid, "table": table["table"],
                                      "paragraphs": [{"uid": p["uid"], "order": p["order"]} for p in paragraphs],
                                      "url": "https://github.com/NExTplusplus/TAT-QA/tree/master/dataset_raw"}})
            for q in item["questions"]:
                qid = str(q["uid"])
                if qid in annotations or "answer" not in q:
                    raise ValueError(f"Duplicate or unlabeled TAT-QA question: {qid}")
                ref = reference(q["answer"], q.get("scale", ""))
                queries.append({"id": qid, "question": q["question"], "reference": ref})
                # Preserve every original annotation, including arithmetic derivations.
                annotations[qid] = {**q, "source_split": split, "corpus_ids": [did]}
        splits[split] = queries
    detail = {"source_contexts_by_split": counts,
              "context_policy": "One complete table with every related paragraph per corpus document; no questions, answers or derivations in corpus.",
              "reference_policy": "Join multi-span answers with '; '; numeric percent answers get %, and other numeric scaled answers get the source scale. Raw typed answers and scales are preserved in query_metadata.json.",
              "test_source": "Official tatqa_dataset_test_gold.json (published test ground truth)."}
    return docs, splits, annotations, detail, "dev"


def validate(root, docs, splits, annotations, sample_size):
    # Re-read installed-format files through the same public loaders used by ARS-Bench.
    from ars_bench.io import normalize_records
    from ars_bench.retrieval import load_corpus, BM25Retriever
    corpus = load_corpus(root / "corpus.jsonl")
    allq = load(root / "queries_all.json")
    selected = load(root / "queries.json")
    ids = {d["id"] for d in corpus}
    assert len(ids) == len(corpus) == len(docs)
    assert len(selected) == sample_size
    assert len(allq) == sum(map(len, splits.values())) == len(annotations)
    assert len({q["id"] for q in allq}) == len(allq)
    assert {q["id"] for q in selected} <= {q["id"] for q in allq}
    assert all(set(q) == {"id", "question", "reference"} for q in allq)
    assert all(isinstance(q[k], str) and q[k].strip() for q in allq for k in q)
    assert len(normalize_records(allq)) == len(allq)
    assert all(set(a["corpus_ids"]) <= ids and a["corpus_ids"] for a in annotations.values())
    by_id = {d["id"]: d for d in corpus}
    if root.name == "covidqa":
        for q in allq:
            a = annotations[q["id"]]
            assert all(any(answer in by_id[did]["text"] for did in a["corpus_ids"])
                       for answer in a["gold_answers"])
    retriever = BM25Retriever(root / "corpus.jsonl")
    checks = []
    for q in selected[:3]:
        hits = retriever.search(q["question"], k=5)
        assert hits and all(h["id"] in ids and h["text"].strip() for h in hits)
        checks.append({"query_id": q["id"], "returned_documents": len(hits),
                       "gold_document_in_top5": bool(set(annotations[q["id"]]["corpus_ids"]) & {h["id"] for h in hits})})
    return {"corpus_documents": len(corpus), "total_queries": len(allq),
            "selected_queries": len(selected), "unique_query_ids": len(annotations),
            "missing_reference_ids": [], "missing_corpus_links": [],
            "schema_and_source_integrity": "passed", "loader_and_bm25_smoke": checks}


def build(name, cache, target, count, seed):
    docs, splits, annotations, detail, selected_split = (covid if name == "covidqa" else tat)(cache)
    target.mkdir()
    raw_dir = target / "raw"
    raw_dir.mkdir()
    prefix = "covid_" if name == "covidqa" else "tat_"
    sources = []
    for filename, url in SOURCES.items():
        if not filename.startswith(prefix):
            continue
        shutil.copyfile(cache / filename, raw_dir / filename)
        p = raw_dir / filename
        sources.append({"file": "raw/" + filename, "url": url,
                        "sha256": sha(p), "bytes": p.stat().st_size})
    with (target / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False, allow_nan=False) + "\n")
    allq = [q for queries in splits.values() for q in queries]
    pool = splits[selected_split]
    indices = sorted(random.Random(seed).sample(range(len(pool)), count))
    selected = [pool[i] for i in indices]
    dump(target / "queries.json", selected)
    dump(target / "queries_all.json", allq)
    for split, queries in splits.items():
        dump(target / f"queries_{split}.json", queries)
    dump(target / "query_metadata.json", annotations)
    quality = validate(target, docs, splits, annotations, count)
    dump(target / "quality.json", quality)
    outputs = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
               for p in sorted(target.iterdir()) if p.is_file()}
    manifest = {"dataset": name, "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "builder_sha256": sha(Path(__file__)), "sources": sources,
                "source_questions_by_split": {s: len(q) for s, q in splits.items()},
                "corpus_documents": len(docs), "sample": {"size": count, "seed": seed,
                    "source_split": selected_split, "method": "Python random.Random(seed).sample; retain source order",
                    "query_ids": [q["id"] for q in selected]},
                "license": "Apache-2.0 (dataset; individual paper licenses remain applicable)" if name == "covidqa" else "CC-BY-4.0",
                "processing": detail, "outputs": outputs}
    dump(target / "manifest.json", manifest)
    (target / "README.md").write_text(
        f"# {name}\n\n"
        "ARS-Bench-compatible data. `corpus.jsonl`: `_id`, `title`, `text`, `metadata`.\n"
        "`queries*.json`: lists of `id`, `question`, `reference` (all strings).\n\n"
        f"`queries.json` contains {count} questions sampled from `{selected_split}` with seed {seed}.\n"
        "`queries_all.json` preserves every labeled source question; `queries_<split>.json` preserves splits.\n"
        "Do not treat COVID-QA's only train split as an official held-out evaluation split.\n\n"
        "`query_metadata.json` maps original question IDs to corpus IDs and preserves original gold annotations.\n"
        "TAT-QA derivations and gold answers occur only in annotations/raw QA, never in corpus.\n"
        "TAT-QA references are convenient strings, not the official numeric/multi-span evaluator; use raw answer/scale metadata for official scoring.\n\n"
        "`manifest.json` records processing, sample IDs, source URLs and SHA-256 hashes.\n"
        "`quality.json` records source/link/schema checks and three BM25 retrieval smoke checks; retrieval recall is not guaranteed.\n"
        "`raw/` preserves downloaded source files and upstream attribution README.\n\n"
        "The application dataset whitelist and evaluation code have not been modified.\n"
        "To rebuild into a separate data directory, from the repository root:\n\n"
        "```sh\nPYTHONPATH=.cache/qa_data_build/deps:. python scripts/prepare_qa_datasets.py --data-root .cache/qa_data_build/rebuilt-data --source-cache .cache/qa_data_build/sources\n```\n",
        encoding="utf-8")
    print(json.dumps({"dataset": name, "quality": quality, "processing": detail}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--source-cache", type=Path, default=Path(".cache/qa_data_build/sources"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 1:
        parser.error("sample-size must be positive")
    args.data_root.mkdir(parents=True, exist_ok=True)
    names = ("covidqa", "tatqa")
    for name in names:
        if (args.data_root / name).exists():
            raise FileExistsError(f"Refusing to overwrite {args.data_root / name}")
    fetch(args.source_cache)
    stage = Path(tempfile.mkdtemp(prefix=".qa-build-", dir=args.data_root))
    for name in names:
        build(name, args.source_cache, stage / name, args.sample_size, args.seed)
    # Publish only after BOTH datasets pass all checks. Keep failures for diagnosis.
    for name in names:
        os.rename(stage / name, args.data_root / name)
    stage.rmdir()
    print("BUILD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
