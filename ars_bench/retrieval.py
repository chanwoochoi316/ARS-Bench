"""Portable dense indexes and a small dependency-free BM25 baseline."""
import collections
import math
import re
from pathlib import Path
from .io import read, write, digest


def resolve_device(device):
    if device == "auto":
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def iter_corpus(path, poisoned=False):
    import json
    path = Path(path)
    def normalize(row, index):
        if not isinstance(row, dict):
            raise ValueError(f"Corpus document {index} must be a JSON object")
        return {"id": str(row.get("id", row.get("_id", index))), "title": str(row.get("title", "")),
                "text": str(row.get("text", row.get("contents", ""))), "is_poisoned": poisoned}
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            index = 0
            for line in stream:
                if line.strip():
                    yield normalize(json.loads(line), index)
                    index += 1
    else:
        documents = read(path)
        if not isinstance(documents, list):
            raise ValueError("Corpus must be JSONL or JSON list")
        for index, row in enumerate(documents):
            yield normalize(row, index)


def load_corpus(path, poisoned=False):
    documents = list(iter_corpus(path, poisoned))
    if not documents:
        raise ValueError("Corpus is empty")
    return documents


def validate_clean_corpus(path, indexed_documents):
    from itertools import zip_longest
    for index, (provided, stored) in enumerate(zip_longest(iter_corpus(path), indexed_documents)):
        if provided != stored:
            raise ValueError(f"--corpus does not match the clean index at document {index}; use the matching corpus/index or rebuild the index")


def build_index(corpus, output, model="intfloat/e5-large-v2", device="auto", batch_size=1024, shard_size=100000):
    from .encoder import SimpleEncoder
    import torch
    output = Path(output)
    if (output / "manifest.json").exists():
        raise ValueError("Index already exists; choose a new output directory")
    device = resolve_device(device)
    docs = load_corpus(corpus)
    encoder = SimpleEncoder(model, device=device, batch_size=batch_size)
    output.mkdir(parents=True, exist_ok=True)
    # A copy of document metadata is stored in the index; no runtime corpus dependency.
    write(output / "documents.json", docs)
    for i, start in enumerate(range(0, len(docs), shard_size)):
        tensor = encoder.encode_corpus(docs[start:start + shard_size])
        torch.save(tensor, output / f"embedding-shard-{i}.pt")
    write(output / "manifest.json", {"version": 1, "model": model, "count": len(docs),
                                      "documents_digest": digest(docs),
                                      "encoding": {"device": device, "dtype": "float16" if "cuda" in device else "float32",
                                                   "max_length": 512, "batch_size": batch_size}})


class DenseRetriever:
    def __init__(self, index, model=None, device="auto", attack_documents=None, corpus=None):
        import torch
        from .encoder import SimpleEncoder
        device = resolve_device(device)
        self.torch, self.device = torch, device
        root = Path(index)
        manifest = read(root / "manifest.json")
        self.documents = read(root / "documents.json")
        if digest(self.documents) != manifest["documents_digest"]:
            raise ValueError("Index documents differ from their manifest")
        if corpus is not None:
            validate_clean_corpus(corpus, self.documents)
        selected_model = model or manifest["model"]
        if model is not None and model != manifest["model"]:
            raise ValueError("Retrieval model must match the index model")
        files = sorted(root.glob("embedding-shard-*.pt"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        if not files:
            raise ValueError("Dense index has no embedding shards")
        self.embeddings = torch.cat([torch.load(f, map_location="cpu", weights_only=True) for f in files])
        if len(self.documents) != len(self.embeddings):
            raise ValueError("Index document and embedding counts differ")
        expected_dtype = torch.float16 if "cuda" in device else torch.float32
        if self.embeddings.dtype != expected_dtype:
            raise ValueError("Index precision does not match the retrieval encoder; rebuild the index with the same CPU/CUDA device type as retrieval")
        self.encoder = SimpleEncoder(selected_model, device=device, batch_size=32)
        if attack_documents:
            poisoned = [{**d, "is_poisoned": True} for d in attack_documents]
            self.documents.extend(poisoned)
            # Match the clean/poison indexer's batch size, then restore query batching.
            self.encoder.batch_size = 1024
            try:
                poison_embeddings = self.encoder.encode_corpus(poisoned).cpu()
            finally:
                self.encoder.batch_size = 32
            self.embeddings = torch.cat([self.embeddings, poison_embeddings])
        self.embeddings = self.embeddings.to(device)

    def search(self, query, k=5):
        q = self.encoder.encode_queries([query]).to(self.device, dtype=self.embeddings.dtype)
        scores, indices = self.torch.topk((q @ self.embeddings.T)[0], min(k, len(self.documents)))
        return [{**self.documents[i], "score": float(s)} for s, i in zip(scores.tolist(), indices.tolist())]


class BM25Retriever:
    def __init__(self, corpus=None, documents=None, attack_documents=None):
        self.documents = list(documents) if documents is not None else load_corpus(corpus)
        self.documents += [{**d, "is_poisoned": True} for d in (attack_documents or [])]
        if not self.documents:
            raise ValueError("Corpus is empty")
        self.tf = []
        self.postings = collections.defaultdict(list)
        for i, d in enumerate(self.documents):
            tf = collections.Counter(re.findall(r"\w+", (d.get("title", "") + " " + d.get("text", "")).lower()))
            self.tf.append(tf)
            for token, frequency in tf.items():
                self.postings[token].append((i, frequency))
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.average = sum(self.lengths) / len(self.tf) or 1

    def search(self, query, k=5):
        scores = collections.defaultdict(float)
        n = len(self.documents)
        for token in set(re.findall(r"\w+", query.lower())):
            posting = self.postings.get(token, [])
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, frequency in posting:
                scores[i] += idf * frequency * 2.5 / (frequency + 1.5 * (0.25 + 0.75 * self.lengths[i] / self.average))
        chosen = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [{**self.documents[i], "score": scores[i]} for i in chosen]
