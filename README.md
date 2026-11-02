# ARS-Bench

ARS-Bench runs ReAct and WebThinker agents over a document collection, generates attack documents, and evaluates the resulting agent behavior.

This guide covers the complete workflow:

1. Index the clean document collection.
2. Run inference without attack documents.
3. Generate attack documents from the clean run.
4. Run inference with the attack documents.
5. Evaluate the results.

## Conda setup

Use a Linux server with NVIDIA GPUs and Conda installed. Run the following commands in Bash:

```bash
git clone https://github.com/chanwoochoi316/ARS-Bench.git
cd ARS-Bench
conda env create -f environment.yml
conda activate ars-bench
export ARS_PYTHON="$CONDA_PREFIX/bin/python"
chmod +x ars
./ars --help
```

The environment includes PyTorch, Transformers, vLLM, retrieval, and evaluation dependencies. Model weights and datasets are downloaded or prepared separately.

In **every additional terminal**, activate the environment and work from the repository root:

```bash
cd /path/to/ARS-Bench
conda activate ars-bench
export ARS_PYTHON="$CONDA_PREFIX/bin/python"
```

Check the installation:

```bash
python -m pip check
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The examples below use `telcodpr`. To use another dataset, replace the dataset name and its paths throughout the commands. Supported names are `hotpotqa`, `legalbenchrag`, `telcodpr`, `covidqa`, and `tatqa`.

Choose **either ReAct or WebThinker** in step 2 and use the same agent and model in step 4. The examples write one experiment to `runs/telcodpr/`; use a different directory for each additional experiment.

## 1. Index the clean documents

Prepare these two files:

```text
data/telcodpr/corpus.jsonl
data/telcodpr/queries.json
```

`corpus.jsonl` contains one clean document per line:

```json
{"id":"doc-1","title":"Timer specification","text":"The timer duration is thirty seconds."}
{"id":"doc-2","title":"Timer configuration","text":"The timer can be configured by the operator."}
```

`queries.json` contains a list of questions with stable IDs and reference answers:

```json
[
  {"id":"q-1","question":"What is the timer duration?","reference":"thirty seconds"}
]
```

Build the dense retrieval index:

```bash
./ars index \
  --corpus data/telcodpr/corpus.jsonl \
  --output indexes/telcodpr/e5 \
  --model intfloat/e5-large-v2 \
  --device cpu
```

Use `--device cuda:0` to index on an available GPU. The index stores the document metadata and embeddings. Build it once and reuse it for clean and attacked inference. To rebuild it, choose a new output directory.

## 2. Run inference without attack documents

Start the model server in a separate terminal and leave it running. Adjust `--gpus` and `--tensor-parallel-size` for your available GPUs. `--model` in a `run` command must match the server's `--name`.

### Option A: ReAct

Start the agent model server:

```bash
./ars serve \
  --model Qwen/Qwen2.5-32B-Instruct \
  --name Qwen2.5-32B-Instruct \
  --port 8000 \
  --gpus 0,1 \
  --tensor-parallel-size 2
```

After the server is ready, run clean inference in another terminal:

```bash
./ars run \
  --dataset telcodpr \
  --architecture react \
  --queries data/telcodpr/queries.json \
  --corpus data/telcodpr/corpus.jsonl \
  --index indexes/telcodpr/e5 \
  --retrieval-device cpu \
  --model Qwen2.5-32B-Instruct \
  --base-url http://localhost:8000/v1 \
  --tokenizer Qwen/Qwen2.5-32B-Instruct \
  --output runs/telcodpr/clean.json
```

### Option B: WebThinker

WebThinker requires a main model server and an auxiliary model server. Start each in its own terminal on separate GPUs:

```bash
./ars serve \
  --model Qwen/QwQ-32B \
  --name QwQ-32B \
  --port 8000 \
  --gpus 0,1 \
  --tensor-parallel-size 2
```

```bash
./ars serve \
  --model Qwen/Qwen2.5-32B-Instruct \
  --name Qwen2.5-32B-Instruct \
  --port 8002 \
  --gpus 2,3 \
  --tensor-parallel-size 2
```

After both servers are ready:

```bash
./ars run \
  --dataset telcodpr \
  --architecture webthinker \
  --queries data/telcodpr/queries.json \
  --corpus data/telcodpr/corpus.jsonl \
  --index indexes/telcodpr/e5 \
  --retrieval-device cpu \
  --model QwQ-32B \
  --base-url http://localhost:8000/v1 \
  --tokenizer Qwen/QwQ-32B \
  --aux-model Qwen2.5-32B-Instruct \
  --aux-base-url http://localhost:8002/v1 \
  --aux-tokenizer Qwen/Qwen2.5-32B-Instruct \
  --output runs/telcodpr/clean.json
```

Both options produce `runs/telcodpr/clean.json`, containing predictions and retrieval/reasoning traces. Wait for clean inference to finish before generating attack documents.

## 3. Generate attack documents

The following example generates KidnapRAG documents. The generation server is needed only for this step. If these GPUs are occupied by agent servers, stop those servers first. After generation, stop the generation server and restart the same agent servers for step 4.

Start the generation model in a separate terminal:

```bash
./ars serve \
  --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --name Qwen3-30B-A3B-Instruct-2507 \
  --port 8001 \
  --gpus 2,3 \
  --tensor-parallel-size 2
```

Run generation after the server is ready:

```bash
./ars generate \
  --clean runs/telcodpr/clean.json \
  --attack kidnaprag \
  --target-answer 'KILL YOURSELF' \
  --model Qwen3-30B-A3B-Instruct-2507 \
  --base-url http://localhost:8001/v1 \
  --output runs/telcodpr/poisoned_corpus.jsonl
```

Set `--target-answer` to the exact output string you want the attack to induce. KidnapRAG generates five bait documents per clean search. Only the bait documents require model generation. The program also appends 25 fixed documents once per attack artifact: 20 chain documents and five final MAL-INS documents.

Supported `--attack` values are `kidnaprag`, `ignore`, `fake_completion`, `poisonedrag`, and `paradox`. Ignore and Fake Completion do not require a running generation model.

The output `poisoned_corpus.jsonl` contains **only attack documents**, one JSON object per line. Each document has an ID, title, text, and metadata identifying the attack and target. The clean inference result remains in `clean.json` and is not embedded in this file.

Example document structure:

```json
{"id":"kidnaprag-000000","title":"A clean search query","text":"The attack document body","is_poisoned":true,"metadata":{"attack_method":"kidnaprag","target_answer":"KILL YOURSELF","kind":"bait"}}
```

Pass the JSONL path directly to `--attack-documents` in step 4.

## 4. Run inference with attack documents

Use the same questions, clean index, architecture, model, and tokenizers as in step 2. Start the agent server or servers again if you stopped them for attack generation.

### ReAct

```bash
./ars run \
  --dataset telcodpr \
  --architecture react \
  --queries data/telcodpr/queries.json \
  --corpus data/telcodpr/corpus.jsonl \
  --index indexes/telcodpr/e5 \
  --retrieval-device cpu \
  --model Qwen2.5-32B-Instruct \
  --base-url http://localhost:8000/v1 \
  --tokenizer Qwen/Qwen2.5-32B-Instruct \
  --attack-documents runs/telcodpr/poisoned_corpus.jsonl \
  --defense none \
  --output runs/telcodpr/attacked.json
```

### WebThinker

```bash
./ars run \
  --dataset telcodpr \
  --architecture webthinker \
  --queries data/telcodpr/queries.json \
  --corpus data/telcodpr/corpus.jsonl \
  --index indexes/telcodpr/e5 \
  --retrieval-device cpu \
  --model QwQ-32B \
  --base-url http://localhost:8000/v1 \
  --tokenizer Qwen/QwQ-32B \
  --aux-model Qwen2.5-32B-Instruct \
  --aux-base-url http://localhost:8002/v1 \
  --aux-tokenizer Qwen/Qwen2.5-32B-Instruct \
  --attack-documents runs/telcodpr/poisoned_corpus.jsonl \
  --defense none \
  --output runs/telcodpr/attacked.json
```

Attack documents are embedded and added to the clean retrieval candidates automatically. You do not need to rebuild or modify the clean index. The output stores attacked predictions, traces, and attack documents. The clean inference result is supplied separately during evaluation in step 5.

The two document inputs are:

- `--corpus`: the normal document collection used to build the clean index.
- `--attack-documents`: the poisoned document JSONL to add to retrieval.

If you already have a poisoned document JSONL, you can skip generation and use its path directly. If its metadata does not specify the attack method or target, add `--attack ignore --target-answer 'KILL YOURSELF'` with the appropriate values.

`--clean` means a **clean inference result**, not the normal document collection. It is optional for inference. Add `--clean runs/telcodpr/clean.json` here to save a paired result, or supply it at evaluation time as shown below. Self-contained `attack.documents.json` files are also accepted.

## 5. Evaluate the results

Download the embedding model used for PDS:

```bash
hf download sentence-transformers/all-mpnet-base-v2 --local-dir models/pds
```

Configure an OpenAI API key for moderation and the LLM judges, then evaluate the attacked result bundle:

```bash
export OPENAI_API_KEY='YOUR_OPENAI_API_KEY'

./ars evaluate runs/telcodpr/attacked.json \
  --clean runs/telcodpr/clean.json \
  --judge-base-url https://api.openai.com/v1 \
  --judge-model gpt-4o \
  --embedding-model models/pds \
  --device cpu \
  --output runs/telcodpr/report.json \
  --require-all
```

Evaluation computes **MBS, IBS, QLES, SLES, PDS, ISS, UFS, RCS, LI, and TUI**. It calls the moderation API and LLM judges. IBS evaluates the first 100 attack documents by default; add `--document-limit -1` to evaluate all attack documents.

Outputs:

```text
runs/telcodpr/report.json         Metric values, status, and evaluation details
runs/telcodpr/report.csv          Metric table
runs/telcodpr/report.bundle.json  Result bundle with reusable evaluation caches
```

Unavailable metrics are reported as `null` with a reason. With `--require-all`, the command returns exit code 2 if any metric is unavailable.

To reuse completed evaluations without API calls or model loading:

```bash
./ars evaluate runs/telcodpr/report.bundle.json \
  --offline \
  --output runs/telcodpr/report-replayed.json \
  --require-all
```
