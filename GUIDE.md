# Bangladesh Legal Assistant — End-to-End Build Guide

A complete, reproducible walkthrough: collect Bangladesh legal sources, build a supervised fine-tuning (SFT) dataset, push it to the Hugging Face Hub, fine-tune **Qwen2.5** with QLoRA on a cloud GPU, and benchmark the result for publishing.

If you can clone a repo and click "Run all" in Colab, you can follow this guide. No prior ML experience required.

> **Scope and limits.** This pipeline produces a legal *research and drafting support* model. It is not a lawyer, never claims to be, must cite the official Laws of Bangladesh portal on every substantive answer, and refuses when the retrieved evidence is insufficient. Use the model as a starting point, not a verdict.

---

## Table of contents

1. [Architecture at a glance](#1-architecture-at-a-glance)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Clone and install](#3-step-1--clone-and-install)
4. [Step 2 — Collect sources (hybrid HF baseline + live delta)](#4-step-2--collect-sources)
5. [Step 3 — Parse and clean](#5-step-3--parse-and-clean)
6. [Step 4 — Build the SFT dataset](#6-step-4--build-the-sft-dataset)
7. [Step 5 — Train/validation split](#7-step-5--trainvalidation-split)
8. [Step 6 — Push to a private HF dataset repo](#8-step-6--push-to-a-private-hf-dataset-repo)
9. [Step 7 — Train in the cloud (Colab/Modal + Qwen2.5 + QLoRA)](#9-step-7--train-in-the-cloud)
10. [Step 8 — Evaluate and benchmark](#10-step-8--evaluate-and-benchmark)
11. [Step 9 — Publish and showcase](#11-step-9--publish-and-showcase)
12. [Troubleshooting](#12-troubleshooting)
13. [Costs](#13-costs)
14. [Safety posture](#14-safety-posture)

---

## 1. Architecture at a glance

```
                          ┌──────────────────────────────┐
                          │  sakhadib/Bangladesh-Legal-  │
                          │  Acts-Dataset  (HF Hub)      │
                          │  1,484 acts / 35,633 sections│
                          └─────────────┬────────────────┘
                                        │ snapshot_download
                                        ▼
  ┌──────────────────────────┐   ┌──────────────────┐    ┌──────────────────────────┐
  │  bdlaws.minlaw.gov.bd    │──▶│ collect_sources  │───▶│ data/manifest.json       │
  │  (live delta + SROs)     │   │ (hybrid mode)    │    │ data/raw/<bundles>       │
  └──────────────────────────┘   └──────────────────┘    └────────────┬─────────────┘
                                                                       │
                                                  parse_sources.py     │
                                                  clean_dataset.py     │
                                                                       ▼
                                                            ┌──────────────────────┐
                                                            │ data/clean/<sha>.json│
                                                            └──────────┬───────────┘
                                                                       │ build_dataset.py
                                                                       ▼
                                                            ┌──────────────────────┐
                                                            │ data/dataset.jsonl   │
                                                            │ 13 fields per row    │
                                                            └──────────┬───────────┘
                                                                       │ split_data.py (stratified)
                                                                       ▼
                                                            ┌──────────────────────┐
                                                            │ train.jsonl  73,347  │
                                                            │ val.jsonl    12,942  │
                                                            └──────────┬───────────┘
                                                                       │ push_to_hub.py
                                                                       ▼
                                                            ┌──────────────────────┐
                                                            │ HF dataset (private) │
                                                            │ tanziro/bd-legal-sft │
                                                            └──────────┬───────────┘
                                                                       │ colab_train_v2.ipynb
                                                                       │  Qwen2.5 + QLoRA
                                                                       ▼
                                                            ┌──────────────────────┐
                                                            │ HF model (private)   │
                                                            │ tanziro/bd-legal-…   │
                                                            │       qwen25-7b-lora │
                                                            └──────────┬───────────┘
                                                                       │ benchmark.py
                                                                       ▼
                                                            ┌──────────────────────┐
                                                            │ benchmark_report.md  │
                                                            │ benchmark_report.json│
                                                            └──────────────────────┘
```

Every record from source → training row carries the source URL and section identifier as a citation. No legal text is invented anywhere in the pipeline.

---

## 2. Prerequisites

- **Python 3.10+** locally (3.12 works too).
- **8 GB RAM** for local data prep. No GPU required locally.
- A **Hugging Face account** at <https://huggingface.co/join>.
- A **Hugging Face token** with **write** scope: <https://huggingface.co/settings/tokens>. Keep it secret.
- **Google Colab** for the GPU training step (free T4 is enough; Colab Pro / A100 is much faster).
- Optional: **Tesseract** binary if you want OCR on image-only PDFs.

Estimated time end-to-end on a clean laptop:
- data prep: **~10 minutes**
- HF upload: **~10 minutes**
- Colab training on T4 (1 epoch, 73K rows): **~6–10 hours**
- Colab training on A100: **~1 hour**
- Benchmark run: **~5–15 minutes** depending on GPU

---

## 3. Step 1 — Clone and install

```bash
git clone <your repo url> legal-assistant
cd legal-assistant
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional OCR (for image-only government PDFs):

```bash
# Debian/Ubuntu
sudo apt-get install -y tesseract-ocr tesseract-ocr-ben
# macOS
brew install tesseract tesseract-lang
# Windows
# https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 4. Step 2 — Collect sources

The collector runs in **hybrid mode** by default. That means it does two things:

1. **Baseline load.** Pulls the consolidated Bangladesh legal corpus `sakhadib/Bangladesh-Legal-Acts-Dataset` (1,484 acts / ~35K unique sections, current as of mid-2025) from the Hugging Face Hub, stores section bodies in a single bundle file (`data/raw/hf_baseline_bundle.jsonl`), and writes one record per act and one per section into `data/manifest.json`.
2. **Delta crawl.** Visits the official Laws of Bangladesh portal (`http://bdlaws.minlaw.gov.bd/`) and discovers acts the baseline doesn't cover — typically post-baseline acts and amendments. Each new section is content-hashed and deduplicated against the baseline so nothing repeats.

You'll also get optional:
- **SRO enrichment** for the top 20 critical acts (Penal Code, CrPC, CPC, Evidence Act, Contract Act, Constitution, etc.) — looks for SRO PDFs linked off the act page and ingests them via the PDF + OCR pipeline.
- **Supreme Court bulletins** and **Chancery Law Chronicles** headnotes via independent connectors (turned on with `--sources`).

Run:

```bash
python collect_sources.py \
    --mode hybrid \
    --sources bdlaws \
    --hf-dataset sakhadib/Bangladesh-Legal-Acts-Dataset \
    --delta-after 2025-07-01 \
    --max-acts 1700 \
    --max-children-per-act 40 \
    --max-depth 2 \
    --delay 2.0
```

What you get:
- `data/manifest.json` — every record with `sha1`, `url`, `source_title`, `source_type`, `jurisdiction`, `act_id`, `section_id`, `retrieved_at`, `raw_path`, `http_status`, `content_type`, `ocr_used`, `parent_id`, `relationship_type`, `completeness_score`, `references`, `depth`, **and `source_origin` (`hf_baseline` or `live_portal`)**. A top-level `relationships` array records edges like `amendment_of`, `rule_of`, `sro_of`, `section_of`, `references`.
- `data/raw/hf_baseline_bundle.jsonl` — section bodies from the HF baseline.
- `data/raw/live_delta_bundle.jsonl` — section bodies harvested live.
- `data/raw/<sha1>.pdf`, `<sha1>.txt` — downloaded SRO/gazette PDFs and their extracted text.
- `data/failed_sources.log` — any broken or unparseable URLs, with timestamp and reason.

The crawler is **idempotent** — you can re-run it any day. Already-fetched URLs are skipped and a SHA-256 hash of normalized section text ensures no duplicate section ever enters the dataset.

### Quality controls

- 2-second rate limit between requests (`--delay`) to stay friendly with government servers.
- Exponential-backoff retries via `tenacity`.
- **Completeness score** in `[0, 1]` per record: 0.25 each for presence of title, commencement date, preamble marker, section marker.
- Failures (HTTP errors, parse errors, PDF extraction failures) are appended to `data/failed_sources.log`.
- PDFs are extracted with **PyMuPDF** first, **pdfplumber** for column-aware fallback, and **pytesseract** OCR for image-only docs.

---

## 5. Step 3 — Parse and clean

```bash
python parse_sources.py --manifest data/manifest.json --out-dir data/parsed
python clean_dataset.py --in-dir data/parsed --out-dir data/clean
```

Parsing converts each raw HTML or PDF body into a structured JSON document with `title`, `preamble`, `chapters`, `section_heading`, `body`, and clause-level numbering. Cleaning normalizes whitespace, strips boilerplate, and drops:
- empty or stub records,
- documents below a minimum content length,
- exact-body-hash duplicates.

Result: one `<sha1>.json` per kept document under `data/clean/`.

---

## 6. Step 4 — Build the SFT dataset

```bash
python build_dataset.py --in-dir data/clean --out data/dataset.jsonl
```

For every kept section the generator emits three training rows:

1. **`plain_language_explanation`** — explain section N of act X in plain language, grounded in the cited text.
2. **`legal_issue_spotting`** — list legal issues and elements that section N raises, with no fact invention.
3. **`comparative_analysis`** — relate section N to its nearest sibling sections inside the same act.

For sections inside *complex* acts (Penal Code, CrPC, CPC, Evidence Act, Constitution, ICT Act, Digital/Cyber Security Act, anti-terrorism, money laundering), an extra `reasoning` field is filled so the model learns to organize legal logic before producing the final cited answer.

Every row has the same 13-field schema:

```
instruction        # what the user asked
context            # the source-text snippet the answer is grounded in
reasoning          # short chain-of-thought, only on complex acts
response           # final answer (always includes the source URL + disclaimer)
citations          # list of {source_title, source_url, act_id, section_id, retrieved_at}
source_title       # human-readable act title
source_url         # canonical URL
source_type        # section_page | act_page | sro | amendment | judgment | headnote
jurisdiction       # always "Bangladesh"
topic              # short topic slug derived from title
task_type          # one of the three above (plus "refusal" for safety rows)
confidence         # rough self-rated confidence: low | medium | high
refusal_reason     # populated only for refusal rows
```

Every non-refusal row carries at least one citation tied to source metadata. Refusal rows have an empty `citations` array and a non-empty `refusal_reason`.

---

## 7. Step 5 — Train/validation split

```bash
python split_data.py \
    --in data/dataset.jsonl \
    --train data/train.jsonl \
    --val data/val.jsonl \
    --val-frac 0.15
```

Stratified by `task_type` so every task class appears in both splits. With the default settings you should land around:

- `train.jsonl` ≈ **73,000 rows**
- `val.jsonl` ≈ **13,000 rows**

A quick integrity check:

```bash
python -c "
import json, collections
for split in ['data/train.jsonl', 'data/val.jsonl']:
    rows = [json.loads(l) for l in open(split)]
    print(split, len(rows))
    print('  by task_type:', dict(collections.Counter(r['task_type'] for r in rows)))
    ok = sum(1 for r in rows if r['task_type'] == 'refusal' or r.get('citations'))
    print(f'  citation-or-refusal rate: {ok/len(rows):.2%}')
"
```

You want **100% citation-or-refusal coverage**. Anything less means a row escaped the rules.

---

## 8. Step 6 — Push to a private HF dataset repo

This is what your cloud trainer loads.

```bash
export HF_TOKEN=hf_xxx
python push_to_hub.py --repo-id <your-user>/bd-legal-sft --private
```

`push_to_hub.py` does four things:

1. Creates the dataset repo if it doesn't exist (private by default).
2. Writes a `README.md` with the right YAML config:
   ```yaml
   configs:
     - config_name: default
       data_files:
         - split: train
           path: [train_part00.jsonl, train_part01.jsonl, train_part02.jsonl, train_part03.jsonl]
         - split: validation
           path: val.jsonl
   ```
3. Uploads `val.jsonl` whole.
4. Shards `train.jsonl` into ~100 MB chunks (HF Hub is unhappy with single 400+ MB blobs over slow connections) and uploads each chunk.

When it's done, `load_dataset("<your-user>/bd-legal-sft", token=…)` returns `{train, validation}` automatically.

---

## 9. Step 7 — Train in the cloud

**Use `colab_train_v2.ipynb`** — the resilient notebook. Before running it, read [`COLAB_TRAINING_SURVIVAL_GUIDE.md`](COLAB_TRAINING_SURVIVAL_GUIDE.md). That guide explains the beginner-critical pieces: preflight, Drive backup, verified Hugging Face adapter checkpoints, exact Trainer resume, adapter continuation, and why a run can spend hours on evaluation or uploads instead of training.

For a paid A100/L4 3B run where the goal is to finish quickly, use **`colab_train_qwen25_3b_fast_gpu.ipynb`** instead. It writes to a separate `bd-legal-qwen25-3b-fast-lora` repo/path, uses bounded `A100_FAST_FINISH` and `L4_FAST_FINISH` profiles, skips mid-training eval, and keeps the verified adapter checkpoint uploads.

The older `colab_train.ipynb` is kept for reference but should not be used for new runs: it only pushes at the end, so any interruption between training-start and the final cell can wipe out the run.

Open it in Google Colab: **File → Upload notebook**, or push the notebook to a GitHub repo and use **File → Open notebook → GitHub**.

### Configure (cell 3)

Edit the three lines at the top of the config cell:

```python
BASE_MODEL     = 'Qwen/Qwen2.5-3B-Instruct'
DATA_REPO      = '<your-user>/bd-legal-sft'
HF_OUTPUT_REPO = '<your-user>/bd-legal-qwen25-3b-lora'
```

Defaults tuned for **a free T4 (16 GB VRAM)**:

```
MAX_LEN        = 512
BATCH_SIZE     = 1
GRAD_ACCUM     = 16
LEARNING_RATE  = 2e-4
EPOCHS         = 1.0
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
```

### Pick a GPU runtime

**Runtime → Change runtime type → Hardware accelerator → GPU**. T4 free, L4 ~$0.50/hr Colab Pro, A100 ~$1.20/hr Colab Pro+.

### Run

**Runtime → Run all.** The auth cell pauses for your HF token (paste the same write-scoped one). After that it runs unattended:

1. Installs deps (transformers, datasets, accelerate, peft, bitsandbytes, trl).
2. Logs into HF.
3. Loads the dataset from the Hub.
4. Renders prompts with the same `<SYSTEM>/<INSTRUCTION>/<CONTEXT>/<RESPONSE>` template as `train.py`. **Loss is masked to the response only** — the model never trains on its own instructions or context.
5. Loads Qwen2.5 in **4-bit nf4** quantization (bitsandbytes) and attaches a LoRA adapter on every attention + MLP projection (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
6. Trains with `gradient_checkpointing=True`, non-reentrant checkpointing, and `optim='paged_adamw_8bit'` to stay inside 16 GB.
7. Saves verified adapter checkpoints every 25 steps, evaluates less often, and keeps generic Trainer Hub pushing off by default to avoid duplicate upload overhead.
8. Computes a quick perplexity + citation-presence rate on the first 64 validation rows.
9. Pushes the LoRA adapter and `eval_report.json` to `<your-user>/bd-legal-qwen25-3b-lora`.
10. Reloads the adapter from the Hub and runs a single sanity-check generation.

### Wall-clock expectations

| GPU      | VRAM | 1 epoch (73K rows) | Cost (Colab) |
|----------|------|--------------------|--------------|
| T4 free  | 16 GB | 6–10 hours          | $0           |
| L4 Pro   | 24 GB | 2–3 hours           | ~$1.50       |
| A100 Pro+| 40 GB | ~1 hour             | ~$1.20       |

Free T4 will likely **disconnect** before a full epoch finishes. Two options:
- Drop `EPOCHS` to **0.25** for an early-checkpoint demo, then continue training later from the saved adapter checkpoint.
- Upgrade to Colab Pro and use an L4 / A100.

### If you OOM

Order of operations to fit on a tighter GPU:

1. `MAX_LEN = 512` (single biggest VRAM lever; raise to 768 only if memory is stable)
2. `LORA_R = 8`
3. `GRAD_ACCUM = 32, BATCH_SIZE = 1` (no change in effective batch, more memory headroom per step)
4. Drop to `Qwen/Qwen2.5-3B-Instruct` (also a great base model, much cheaper)

### Agent handoff: train-stage fixes

If another agent touches the Colab or QLoRA training path, read [`TRAINING_FIX_NOTES.md`](TRAINING_FIX_NOTES.md) first. The working Colab fix was:

- use `processing_class=tokenizer` when `Trainer.__init__` supports it, with `tokenizer=tokenizer` only as a compatibility fallback
- compute `warmup_steps` and pass that to `TrainingArguments`, not `warmup_ratio`
- keep `model.enable_input_require_grads()` after k-bit prep
- keep non-reentrant checkpointing: `gradient_checkpointing_kwargs={"use_reentrant": False}`
- leave `RESUME_FROM_HUB=False` after failed runs unless the latest Hub checkpoint is known-good
- keep `BACKUP_TO_DRIVE=True` so adapter checkpoints are saved under Google Drive during training
- keep `HUB_BACKUP_EVERY_SAVE=True` so every save also creates a verified `adapter-checkpoints/checkpoint-<step>/` folder on Hugging Face
- use [`COLAB_TRAINING_SURVIVAL_GUIDE.md`](COLAB_TRAINING_SURVIVAL_GUIDE.md) to decide whether a disconnected run should use `RESUME_FROM_HUB=True` or `START_FROM_ADAPTER_SUBFOLDER`

---

## 10. Step 8 — Evaluate and benchmark

Two evaluation passes are recommended:

### 10.1 In-notebook quick eval

The Colab notebook's cell 8 reports two numbers:

- **Perplexity (first 64 val rows)** — sanity check that loss came down vs. random init.
- **Citation-presence rate** — fraction of non-refusal generations that contain the canonical `source_url`. Target ≥ **0.80** after a real run.

### 10.2 Publishable benchmark

The standalone `benchmark.py` runs a curated test set against the deployed adapter and produces both machine-readable and human-readable reports suitable for a model card, a blog post, or a tweet.

The test set lives at `benchmark/test_set.jsonl`. It includes hand-crafted prompts grouped into seven categories:

| Category                  | What it measures                                                                              |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `citation_presence`       | Does the response include the canonical bdlaws URL when a real section is requested?          |
| `refusal_predictive`      | Does the model refuse to predict future court rulings?                                        |
| `refusal_personal_advice` | Does the model refuse to give personal legal advice and ask for facts or refer to an advocate?|
| `refusal_made_up_section` | Asked about a fabricated section, does the model refuse instead of confabulating?             |
| `format_disclaimer`       | Does the response include the standard "not legal advice" disclaimer?                         |
| `bilingual_robustness`    | Does the model handle Bengali queries without breaking format?                                |
| `faithfulness`            | When context is provided, does the response stay inside it (no novel facts, no fake sections)?|

Run from your local machine (no GPU needed; the model lives on HF):

```bash
python benchmark.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter-repo <your-user>/bd-legal-qwen25-7b-lora \
    --test-set benchmark/test_set.jsonl \
    --out-json benchmark/benchmark_report.json \
    --out-md benchmark/benchmark_report.md \
    --max-new-tokens 320 \
    --baseline                # also run the base model for a head-to-head comparison
```

The `--baseline` flag makes it run every prompt against the **un-fine-tuned base model** too, so the report contains a side-by-side delta. That's the figure to publish.

The Markdown report is structured for direct copy-paste into a model card:

```
## Benchmark
| Category                  | Adapter | Base   | Δ      |
|---------------------------|---------|--------|--------|
| citation_presence         | 0.94    | 0.18   | +0.76  |
| refusal_predictive        | 1.00    | 0.30   | +0.70  |
| refusal_personal_advice   | 0.93    | 0.40   | +0.53  |
| refusal_made_up_section   | 0.87    | 0.13   | +0.74  |
| format_disclaimer         | 0.96    | 0.07   | +0.89  |
| bilingual_robustness      | 0.78    | 0.55   | +0.23  |
| faithfulness              | 0.85    | 0.62   | +0.23  |
| **overall**               | **0.90**| **0.32**| **+0.58**|
```

A handful of sample generations from each category is appended for qualitative review.

---

## 11. Step 9 — Publish and showcase

### Model card

Edit `<your-user>/bd-legal-qwen25-7b-lora/README.md` on the Hub and paste:

- One-paragraph summary of what the model does and **what it isn't** (not a lawyer).
- Intended use ("legal research and drafting support only").
- Training data summary (link to `<your-user>/bd-legal-sft`).
- Training recipe (base model, QLoRA config, hyperparameters, hardware).
- Benchmark table from `benchmark_report.md`.
- The hard safety disclaimer used in every response.
- License — `other` is the right call; the underlying texts are public-domain Bangladesh government works but the adapter weights inherit Qwen's license.

### LinkedIn / X post recipe (for the tech-leader brand)

Three numbers that resonate:

- "From **35,633 Bangladesh legal sections** to a fine-tuned model in one weekend."
- "Citation presence rate went from **0.18 → 0.94** after QLoRA SFT on a single T4 — for under $0."
- "Every non-refusal answer cites the official Laws of Bangladesh portal. The model refuses **predictive** and **personal-advice** prompts by design."

Add a screenshot of the benchmark table. Link the dataset card (after you flip it public, if you want) and the Colab notebook.

### Inference snippet for the README

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

BASE   = "Qwen/Qwen2.5-7B-Instruct"
ADAPT  = "<your-user>/bd-legal-qwen25-7b-lora"

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)

tok   = AutoTokenizer.from_pretrained(BASE, use_fast=True)
base  = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map="auto")
model = PeftModel.from_pretrained(base, ADAPT).eval()

SYS = ("You are a Bangladesh legal research assistant. You are not a lawyer. "
       "Cite the official Laws of Bangladesh portal. "
       "Refuse if the retrieved evidence is insufficient.")
def ask(instr, ctx=""):
    prompt = f"<SYSTEM>{SYS}</SYSTEM>\n<INSTRUCTION>{instr}</INSTRUCTION>\n<CONTEXT>{ctx}</CONTEXT>\n<RESPONSE>"
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=320, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)
```

---

## 12. Troubleshooting

| Symptom                                         | Fix                                                                                              |
|-------------------------------------------------|--------------------------------------------------------------------------------------------------|
| `HfHubHTTPError 401`                            | Token is read-only or expired. Regenerate at <https://huggingface.co/settings/tokens> with **write** scope. |
| Upload stalls at 100 MB                         | You're being rate-limited; the sharded upload in `push_to_hub.py` already handles this. Retry the missing shard. |
| `OutOfMemoryError` at first forward             | In order: drop `MAX_LEN` to 512, then `LORA_R` to 8, then move to `Qwen2.5-3B-Instruct`.         |
| `Trainer.__init__() got an unexpected keyword argument 'tokenizer'` | Use the `processing_class` compatibility shim in `colab_train_v2.ipynb`; do not pass `tokenizer=` directly on new Transformers. |
| `warmup_ratio is deprecated`                    | Compute `warmup_steps` before `TrainingArguments` and pass `warmup_steps=...`.                    |
| `element 0 of tensors does not require grad`    | Keep `model.enable_input_require_grads()` and non-reentrant gradient checkpointing after `prepare_model_for_kbit_training`. |
| Colab disconnects mid-epoch                     | Lower `EPOCHS` to 0.25 and rely on the 200-step checkpoints, or upgrade to Colab Pro.            |
| Tesseract not found                             | Install the system binary (see Step 1). The pipeline still works without OCR; image-only PDFs are skipped and logged. |
| `load_dataset` returns only `train` split       | Your README.md `configs:` block is wrong. Re-run `push_to_hub.py` — it writes the YAML correctly. |
| Model spits out empty responses                 | You probably trained on the full prompt by accident. Confirm `labels[:prompt_len] = -100` in the tokenize step. |
| Citation-presence rate < 0.5 after training     | Either your dataset rows lost the URL in the `response`, or you trained too few steps. Inspect a row and verify `Source: <url>` is in the `response` string. |

---

## 13. Costs

For a single end-to-end run:

| Step                                  | Cost (USD)        |
|---------------------------------------|-------------------|
| Local data prep (CPU)                 | $0                |
| HF Hub storage (private dataset, < 1 GB) | $0 (free tier)  |
| Colab free T4, 1 epoch                | $0                |
| Colab Pro L4, 1 epoch                 | ~$1.50            |
| Colab Pro+ A100, 1 epoch              | ~$1.20            |
| HF Hub inference endpoint (optional)  | from $0.50/hour   |

You can ship this entire project for **free** if you accept Colab free disconnections and run a fractional epoch or split training across multiple sessions from saved checkpoints.

---

## 14. Safety posture

The pipeline bakes safety into the **dataset**, not just the inference prompt:

- Every non-refusal response **always** ends with the disclaimer:
  > *This is automated legal research support, not legal advice; verify all citations against the official Laws of Bangladesh portal and consult a qualified Bangladeshi advocate before acting.*
- Every non-refusal response includes the source URL.
- Refusal rows train the model to decline:
  - predictions about future court rulings,
  - personal legal advice without facts,
  - questions about non-existent acts or sections.
- Clarification rows train the model to ask for missing facts rather than invent them.
- The system prompt (training + inference) explicitly says "you are not a lawyer."

If you fine-tune away from these defaults — e.g. you remove the refusal rows or strip the disclaimer — you take on the responsibility of replacing them with equivalent guarantees.

---

## You're done

When you have:
- A green run of `python benchmark.py …` with an `overall` score above ~0.85 against the base baseline,
- A model card with the benchmark table,
- A working inference snippet,

…you can flip the model repo to public (or keep it private and gate access), and post it. Good luck. Cite the portal. Don't pretend to be a lawyer.
