"""Instrumented ReAct and WebThinker runners using configurable model endpoints."""
import re
import time
from pathlib import Path
from .io import read, write, normalize_records, bundle
from .defenses import SPOTLIGHT_INSTRUCTION
from .agent_prompts import (get_multiqa_search_o1_instruction, get_task_instruction_openqa,
                           get_webpage_to_reasonchain_instruction)

BEGIN_QUERY, END_QUERY = "<|begin_search_query|>", "<|end_search_query|>"
BEGIN_RESULT, END_RESULT = "<|begin_search_result|>", "<|end_search_result|>"


class TokenCounter:
    def __init__(self, model):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.method = f"tokenizer:{model}:thought+action:no-special-tokens"

    def __call__(self, text):
        return len(self.tokenizer.encode(str(text or ""), add_special_tokens=False))


def extract_final(text):
    # Balanced braces handle boxed answers containing nested LaTeX expressions.
    start = text.rfind("\\boxed{")
    if start >= 0:
        depth = 1
        for i in range(start + 7, len(text)):
            depth += (text[i] == "{") - (text[i] == "}")
            if depth == 0:
                return text[start + 7:i].strip()
    match = re.search(r"(?:\*\*ANSWER:\s*|Final Answer:\s*)(.*)", text, re.I | re.S)
    return match.group(1).strip().strip("*") if match else text.strip()


def evidence(documents):
    return "\n\n".join(f"Document {i+1}\n{d.get('title','')}\n{d.get('text',d.get('contents',''))}" for i, d in enumerate(documents))


def run_one(query, architecture, client, retriever, defense, token_counter=None, aux_client=None,
            max_steps=10, top_k=5, max_tokens=None):
    if architecture not in {"react", "webthinker"}:
        raise ValueError("architecture must be react or webthinker")
    if architecture == "webthinker" and aux_client is None:
        raise ValueError("WebThinker needs an auxiliary evidence-summary client")
    record = {"id": query["id"], "question": query["question"], "reference": query["reference"],
              "prediction": "", "steps": [], "reasoning_time_s": 0.0,
              "completion_status": "step_limit"}
    if token_counter:
        record.update(reasoning_tokens=0, token_count_method=token_counter.method)
    marking = "\n" + SPOTLIGHT_INSTRUCTION if defense.method == "spotlight" else ""
    if architecture == "react":
        config = Path(__file__).resolve().parent / "resources/react_fewshot.json"
        fewshot = read(config)["webthink_simple6"]
        prompt = ("Solve the question using interleaved Thought, Action, Observation. "
                  "Available actions: Search[query], Finish[answer]. Do not use Lookup.\n" + marking + "\n" +
                  fewshot + f"\nQuestion: {query['question']}\n")
    else:
        prompt = get_multiqa_search_o1_instruction(max_steps) + marking + get_task_instruction_openqa(query["question"])
    for index in range(1, max_steps + 1):
        started = time.perf_counter()
        if architecture == "react":
            raw = client.complete(prompt + f"Thought {index}:", temperature=0, top_p=1,
                                  max_tokens=max_tokens or 100,
                                  stop=[f"\nObservation {index}:", "\nQuestion:"])
        else:
            raw = client.chat([{"role": "user", "content": prompt}], temperature=0.7, top_p=0.8,
                              max_tokens=max_tokens or 4096, stop=[END_QUERY])
        elapsed = time.perf_counter() - started
        record["reasoning_time_s"] += elapsed
        if architecture == "react":
            match = re.search(r"(?:^|\n)Action\s*\d*:\s*(Search|Finish)\[(.*?)\]", raw, re.I | re.S)
            if match is None:
                match = re.match(r"(Search|Finish)\[(.*?)\]", raw.strip(), re.I | re.S)
            if not match:
                step = {"thought": raw, "action": "", "search_query": None, "is_search": False,
                        "reasoning_time_s": elapsed, "parse_error": True}
                record["steps"].append(step)
                prompt += f"Thought {index}: {raw}\nObservation {index}: Use Search[query] or Finish[answer].\n"
                if token_counter:
                    record["reasoning_tokens"] += token_counter(raw)
                continue
            thought = re.sub(r"\n?Action\s*\d*:\s*$", "", raw[:match.start()]).strip()
            kind, argument = match.group(1).lower(), match.group(2).strip()
            action = f"{kind.title()}[{argument}]"
            search_query = argument if kind == "search" else None
            completed = kind == "finish"
        else:
            match = re.search(re.escape(BEGIN_QUERY) + r"(.*?)(?:" + re.escape(END_QUERY) + r"|$)", raw, re.S)
            if match:
                thought, search_query = raw[:match.start()].strip(), match.group(1).strip()
                if not search_query:
                    raise ValueError("WebThinker generated an empty search query")
                action, completed = f"Search[{search_query}]", False
            else:
                thought, search_query = raw, None
                action, completed = f"Finish[{extract_final(raw)}]", True
        step = {"thought": thought, "action": action, "search_query": search_query,
                "is_search": bool(search_query), "reasoning_time_s": elapsed}
        if token_counter:
            step["reasoning_tokens"] = token_counter(thought) + token_counter(action)
            step["token_count_method"] = token_counter.method
            record["reasoning_tokens"] += step["reasoning_tokens"]
        record["steps"].append(step)
        if completed:
            record["prediction"] = argument if architecture == "react" else extract_final(raw)
            record["completion_status"] = "finished"
            break
        raw_docs = retriever.search(search_query, k=top_k)
        docs, defense_info = defense.apply(search_query, raw_docs)
        step.update(raw_documents=raw_docs, retrieved_documents=docs, defense=defense_info)
        observation = evidence(docs) or "No documents found."
        if architecture == "webthinker":
            summary_prompt = get_webpage_to_reasonchain_instruction(thought, search_query, observation)
            if defense.method == "spotlight":
                summary_prompt = SPOTLIGHT_INSTRUCTION + "\n" + summary_prompt
            # Auxiliary model calls and retrieved text are outside reasoning-time/token metrics.
            summary = aux_client.chat([{"role": "user", "content": summary_prompt}], temperature=0, max_tokens=4096)
            match = re.search(r"\*?\*?Final Information\*?\*?\s*(.*)", summary, re.I | re.S)
            observation = match.group(1).strip() if match else summary
            step["auxiliary_summary"] = observation
            prompt += raw + (END_QUERY if not raw.endswith(END_QUERY) else "") + f"\n{BEGIN_RESULT}{observation}{END_RESULT}\n"
        else:
            prompt += f"Thought {index}: {thought}\nAction {index}: {action}\nObservation {index}: {observation}\n"
        step["observation"] = observation
    return record


def run(queries, architecture, client, retriever, defense, experiment, output, clean=None,
        documents=None, token_counter=None, aux_client=None, max_steps=10, top_k=5, max_tokens=None):
    normalized = normalize_records(queries)
    if not normalized:
        raise ValueError("No queries")
    records = []
    wall_start = time.perf_counter()
    for q in normalized:
        r = run_one(q, architecture, client, retriever, defense, token_counter, aux_client, max_steps, top_k, max_tokens)
        records.append(r)
        trace = {"artifact_version": "ars-trace/1", "experiment": experiment, "records": records,
                 "run_status": "complete" if len(records) == len(normalized) else "partial",
                 "expected_record_count": len(normalized),
                 "measurement": {"reasoning_time": "sum of main-model generation request elapsed seconds per query",
                                 "auxiliary_calls_included": False, "retrieval_included": False,
                                 "wall_time_s": time.perf_counter() - wall_start}}
        if clean is not None:
            # A partial run remains an explicit checkpoint until its full ID set is present.
            partial_clean = [c for c in normalize_records(clean) if c["id"] in {r["id"] for r in records}]
            trace = bundle(partial_clean, records, documents or [], experiment)
            trace["run_status"] = "complete" if len(records) == len(normalized) else "partial"
            trace["expected_record_count"] = len(normalized)
        write(output, trace)
        print(f"[{len(records)}/{len(normalized)}] id={q['id']} status={r['completion_status']}", flush=True)
    return trace
