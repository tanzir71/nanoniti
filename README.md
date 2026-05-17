# Bangladesh Legal Assistant — Training Pipeline

A reproducible, source-grounded pipeline that collects Bangladesh legal texts from the official Laws of Bangladesh portal, normalizes them, builds a conservative supervised fine-tuning (SFT) dataset, and runs a Hugging Face fine-tuning job.

> **Looking for the end-to-end walkthrough?** Open [`GUIDE.md`](GUIDE.md) — a step-by-step build guide aimed at anyone who can run `pip install` and click "Run all" in Colab, including the cloud training and the publishable benchmark.

> **Running Colab training?** Read [`COLAB_TRAINING_SURVIVAL_GUIDE.md`](COLAB_TRAINING_SURVIVAL_GUIDE.md) first. It explains persistence, resume settings, checkpoint folders, and how to avoid spending hours on eval/upload overhead instead of training.

**Scope and limits.** This system is for legal research and drafting support only. It is not a lawyer. It must not produce unsupported legal advice. When evidence is insufficient, it refuses. Every non-refusal training example carries citations back to the source URL.

## Repository layout

```
legal-assistant/
├── collect_sources.py    # multi-source recursive harvester
├── pdf_processor.py      # PDF text extraction + OCR fallback
├── push_to_hub.py        # upload train.jsonl/val.jsonl to a private HF dataset
├── colab_train_v2.ipynb  # resilient cloud SFT notebook (Qwen2.5 + QLoRA)
├── colab_train_qwen25_3b_fast_gpu.ipynb # bounded A100/L4 3B Colab notebook
├── colab_train_qwen35_9b.ipynb # bounded A100/L4 9B Colab notebook
├── colab_repair_qwen35_9b.ipynb # targeted 9B adapter correction notebook
├── colab_benchmark_qwen35_9b.ipynb # uploadable Colab benchmark notebook
├── modal_train.py        # Modal GPU training path with Hub push + eval report
├── benchmark.py          # publishable benchmark suite (adapter vs. base)
├── benchmark/
│   └── test_set.jsonl    # curated benchmark prompts (7 categories)
├── GUIDE.md              # full end-to-end walkthrough for non-ML users
├── parse_sources.py      # raw HTML -> structured JSON
├── clean_dataset.py      # normalize + dedupe
├── build_dataset.py      # JSONL SFT generator with 7 task types
├── split_data.py         # stratified train/val split
├── train.py              # legacy small-model SFT script
├── train_qlora.py        # local 4-bit QLoRA trainer for Advocore
├── merge_and_export.py   # merge LoRA adapters into an FP16 export
├── evaluate.py           # minimal eval pass + citation-presence check
├── TRAINING_FIX_NOTES.md # handoff notes for the Colab/QLoRA train-stage fixes
├── COLAB_TRAINING_SURVIVAL_GUIDE.md # beginner guide for persistence, resume, and runtime overhead
├── requirements.txt
└── data/
    ├── manifest.json         # source manifest + relationships
    ├── failed_sources.log    # broken / unparseable URLs (appended)
    ├── raw/                  # raw fetched HTML, PDFs, and OCR/extracted .txt
    ├── clean/                # cleaned, structured JSON
    ├── train.jsonl           # produced by split_data.py
    └── val.jsonl
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Source collection

`collect_sources.py` is a multi-source recursive harvester. It supports several connectors, each independently toggleable via `--sources`:

- **`bdlaws`** — primary + secondary legislation on <http://bdlaws.minlaw.gov.bd/>: current and repealed Acts, their sections, plus linked **Rules**, **Regulations**, **SROs**, **Amendments**, and **Gazette** PDFs.
- **`supremecourt`** — public bulletins / cause lists / judgment summaries discoverable from <http://www.supremecourt.gov.bd/>.
- **`chancery`** — headnotes / case commentary discoverable from <https://www.chancerylawchronicles.com/>.

Each fetched record is stored in `data/manifest.json` with: `sha1`, `url`, `source_title`, `source_type`, `jurisdiction`, `act_id`, `section_id`, `retrieved_at`, `raw_path`, `http_status`, `content_type` (`html`/`pdf`), `ocr_used`, `parent_id`, `relationship_type`, `completeness_score`, `references`, and `depth`. A top-level `relationships` array records edges such as `amendment_of`, `rule_of`, `sro_of`, `gazette_of`, `repeal_of`, `references`, and (when resolvable to an already-collected act) `references_resolved`.

PDFs are downloaded into `data/raw/<sha1>.pdf`. Text is extracted with column-aware **pdfplumber**; image-only PDFs fall back to **PyMuPDF rendering + pytesseract OCR** (English + Bengali if `ben` traineddata is installed). Extracted plain text is written alongside as `<sha1>.txt`.

Quality control:
- A 2-second default rate limit (`--delay`) avoids IP blocking from government servers.
- Exponential-backoff retries on transient failures.
- Any broken link or unparseable page is appended to `data/failed_sources.log` with timestamp + reason.
- Each record carries a `completeness_score` in `[0, 1]`, scored over presence of: title, date of commencement, preamble, and sections (0.25 each).

```bash
python collect_sources.py \
    --out-dir data \
    --sources bdlaws,supremecourt,chancery \
    --volumes 1,2,3,4,5 \
    --max-acts 25 \
    --max-children-per-act 40 \
    --max-depth 2 \
    --max-court-pages 30 \
    --max-chancery-pages 30 \
    --delay 2.0
```

The crawl is idempotent: re-running skips URLs already in the manifest. Cross-references between Acts (e.g. *"as defined in the Transfer of Property Act, 1882"*) are detected via regex and recorded as `references` edges; when both sides have been collected, they are upgraded to `references_resolved` edges.

OCR requires the system Tesseract binary. On Debian/Ubuntu: `sudo apt-get install -y tesseract-ocr tesseract-ocr-ben`. On macOS: `brew install tesseract tesseract-lang`. On Windows: <https://github.com/UB-Mannheim/tesseract/wiki>.

## Parsing and cleaning

Parsing converts each raw HTML record into a structured JSON document with title, preamble, chapter / part headings, section heading, body text, and clause-level numbering. Pages that cannot be parsed reliably are skipped and logged.

```bash
python parse_sources.py --manifest data/manifest.json --out-dir data/parsed
python clean_dataset.py --in-dir data/parsed --out-dir data/clean
```

Cleaning strips boilerplate, normalizes whitespace, drops empty or stub documents, and deduplicates by SHA1 of the normalized body.

## Dataset generation

`build_dataset.py` emits a JSONL file at `data/dataset.jsonl` where every row has the required schema:

```
instruction, context, reasoning, response, citations,
source_title, source_url, source_type, jurisdiction,
topic, task_type, confidence, refusal_reason
```

Three dense task types are generated for every cleaned legal section, all grounded in retrieved source text:

1. **plain_language_explanation** - explain the operative rule plainly.
2. **legal_issue_spotting** - identify issues, elements, and clauses.
3. **comparative_analysis** - relate the section to nearby sections in the same Act.

Every row carries at least one citation with `source_title`, `source_url`, `act_id`, `section_id`, and `retrieved_at`. Complex criminal, cyber/ICT, constitutional, and financial-crime sections also include a `reasoning` field so the trainer can learn to organize legal logic before producing the final cited answer.

```bash
python build_dataset.py --in-dir data/clean --out data/dataset.jsonl
python split_data.py --in data/dataset.jsonl --train data/train.jsonl --val data/val.jsonl --val-frac 0.15
```

The split is stratified by `task_type` so every task class appears in both splits.

## Training

`train_qlora.py` is the local RTX 3060 training path for Advocore. It loads the base model in 4-bit NF4 with bitsandbytes, applies LoRA adapters with `r=16` and `lora_alpha=32`, uses gradient checkpointing, and saves only adapters to `models/advocore-adapters/`.

The default base model is `meta-llama/Llama-3.1-8B-Instruct`; set `BASE_MODEL` or pass `--base-model` to use another gated/open instruct model.

Each example is rendered as:

```
<SYSTEM>...legal research assistant; not a lawyer; cite the portal; refuse if evidence insufficient...</SYSTEM>
<INSTRUCTION>{instruction}</INSTRUCTION>
<CONTEXT>{context}</CONTEXT>
<RESPONSE>{response}</RESPONSE>
```

Loss is computed only on `<RESPONSE>` tokens; prompt and pad tokens are masked with `-100`.

```bash
python train_qlora.py \
    --train-file data/train.jsonl \
    --val-file data/val.jsonl \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --output-dir models/advocore-adapters \
    --max-seq-length 2048 \
    --epochs 5 \
    --batch-size 1 \
    --grad-accum 4 \
    --report-to tensorboard
```

For Weights & Biases logging, run `wandb login` once and pass `--report-to tensorboard,wandb`. The scripts set the project name to `Advocore`.

After training, export a merged FP16 model for inference:

```bash
python merge_and_export.py \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --adapter-dir models/advocore-adapters \
    --output-dir models/advocore-fp16
```

`merge_and_export.py` defaults to CPU merge/export, which is slower but safer than trying to hold a full FP16 8B model in 12GB VRAM.

## Benchmark (publishable)

After training completes, `benchmark.py` runs a curated test set against the deployed adapter (and optionally the un-fine-tuned base for a head-to-head comparison) and writes a model-card-ready report.

Test set: [`benchmark/test_set.jsonl`](benchmark/test_set.jsonl) — 24 prompts across 7 categories:

- `citation_presence` — does the response include the canonical bdlaws URL on real-section lookups?
- `refusal_predictive` — does it refuse to predict future court rulings?
- `refusal_personal_advice` — does it refuse to give personal legal advice?
- `refusal_made_up_section` — does it refuse on fabricated sections instead of hallucinating?
- `format_disclaimer` — does every substantive answer carry the standard disclaimer?
- `bilingual_robustness` — does it handle Bengali queries cleanly?
- `faithfulness` — given explicit context, does it stay inside it?

```bash
python benchmark.py \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --adapter-repo <your-user>/bd-legal-qwen25-7b-lora \
    --test-set benchmark/test_set.jsonl \
    --out-json benchmark/benchmark_report.json \
    --out-md   benchmark/benchmark_report.md \
    --baseline
```

`--baseline` runs the un-fine-tuned base model on the same prompts, so the report contains a side-by-side delta table — the figure to publish.

## Cloud training (Hugging Face Hub + Colab)

For larger base models (Qwen2.5-7B-Instruct and up) train in the cloud instead of locally:

1. **Push the SFT splits to a private HF dataset.** Generate a write-scoped token at <https://huggingface.co/settings/tokens>, then:

   ```bash
   export HF_TOKEN=hf_xxx
   python push_to_hub.py --repo-id <your-user>/bd-legal-sft --private
   ```

   The script creates the repo if needed, writes a dataset card with `train`/`validation` split mapping, and uploads `data/train.jsonl` + `data/val.jsonl`. The URL is printed at the end.

2. **Open `colab_train_v2.ipynb` in Google Colab.** Free T4 (16 GB) is enough for a cautious Qwen2.5-3B-Instruct QLoRA run, but it can be slow. If you are paying for A100/L4 and want the 3B adapter to finish as fast as possible, use `colab_train_qwen25_3b_fast_gpu.ipynb` instead; it has bounded A100/L4 profiles, skips mid-training eval, and writes to a separate `bd-legal-qwen25-3b-fast-lora` repo/path.

   - install + HF login (paste the same token)
   - edit `BASE_MODEL`, `DATA_REPO`, `HF_OUTPUT_REPO` in the config cell
   - load the dataset from the Hub
   - tokenize with the same `<SYSTEM>/<INSTRUCTION>/<CONTEXT>/<RESPONSE>` template as `train.py` (loss masked to the response only)
   - load Qwen2.5 in 4-bit + attach a LoRA adapter (r=16, alpha=32, dropout=0.05, targeting all Qwen attention + MLP projections)
   - train with non-reentrant gradient checkpointing + `paged_adamw_8bit`
   - quick perplexity + citation-presence eval on validation
   - push the LoRA adapter (and `eval_report.json`) back to a private HF model repo
   - reload the adapter for a sanity-check inference call

   Defaults: `MAX_LEN=512`, `BATCH_SIZE=1`, `GRAD_ACCUM=16`, `LR=2e-4`, 1 epoch. Bump `MAX_LEN` or switch to 7B only on bigger GPUs.

   Beginner/operator guide: read [`COLAB_TRAINING_SURVIVAL_GUIDE.md`](COLAB_TRAINING_SURVIVAL_GUIDE.md) before running or resuming Colab training. It explains why preflight always runs, how to choose between exact Trainer resume and adapter continuation, and why eval/upload overhead can dominate wall-clock time.

   Agent handoff: the current train-stage compatibility fixes are documented in [`TRAINING_FIX_NOTES.md`](TRAINING_FIX_NOTES.md). Do not reintroduce direct `Trainer(tokenizer=...)`, `warmup_ratio` in Colab `TrainingArguments`, or reentrant checkpointing.

   The notebook also mounts Google Drive and saves adapter backups at every checkpoint, plus verified Hub checkpoint folders under `adapter-checkpoints/`. Keep `BACKUP_TO_DRIVE=True` and `HUB_BACKUP_EVERY_SAVE=True`; Colab `/content` is temporary and must not be the only copy of trained LoRA weights.

3. **Use the adapter at inference time** by loading the base model in 4-bit and wrapping it with `PeftModel.from_pretrained(base, HF_OUTPUT_REPO)`.

### Qwen3.5-9B Benchmark And Repair Handoff

The 9B adapter was benchmarked in Colab on an A100 using `colab_benchmark_qwen35_9b.ipynb`. These are the important logs from the live debugging session:

| Report | Benchmark state | Adapter overall | Base overall | Notes |
|---|---|---:|---:|---|
| `benchmark_report_qwen35_9b.json` | adapter only, first run | `0.333` | n/a | Showed citation/disclaimer/fake-section issues, but the benchmark still had scoring bugs. |
| `benchmark_report_qwen35_9b (1).json` | baseline true, pre-patch benchmark | `0.333` | `0.083` | Adapter improved over base, but Bengali row was corrupted as `????????` and refusal scoring was too literal. |
| `benchmark_report_qwen35_9b (2).json` | baseline true, patched benchmark | `0.333` | `0.250` | Cleaner signal: adapter slightly beats base, but fails citation rows, fake-section refusal, Bengali robustness, and one context-only faithfulness row. |

Latest patched benchmark category scores:

| Category | Adapter | Base | Practical read |
|---|---:|---:|---|
| `citation_presence` | `0.0` | `0.0` | Neither model reliably emits exact official citation format. |
| `refusal_predictive` | `1.0` | `1.0` | Both refused prediction after scoring fix. |
| `refusal_personal_advice` | `1.0` | `0.0` | Adapter improved refusal behavior. |
| `refusal_made_up_section` | `0.0` | `1.0` | Adapter hallucinated fake `section=9999`; this is the main repair target. |
| `format_disclaimer` | `1.0` | `0.0` | Adapter learned disclaimer/citation style better than base. |
| `bilingual_robustness` | `0.0` | `0.0` | Adapter output was truncated on Bengali. |
| `faithfulness` | `0.5` | `0.5` | Adapter still answered "yes" to a context that did not mention cybercrime. |

Do not treat the 9B adapter as ready for public legal use yet. It is worth repairing, not retraining from scratch. The correction notebook is `colab_repair_qwen35_9b.ipynb`; it continues from `final-adapter/` and writes the repaired adapter to `repair-v1-final-adapter/`, leaving the original adapter intact.

The repair notebook intentionally defaults to `USE_4BIT = False`. Colab repeatedly failed on optional quantization libraries during the repair pass:

- old `transformers` did not recognize `model_type: qwen3_5`;
- upgrading `transformers` required a runtime restart before the active kernel saw the new version;
- `bitsandbytes` triggered `libnvJitLink.so.13` CUDA loader errors;
- simply setting `load_in_4bit=False` was not enough while `BitsAndBytesConfig` or `quantization_config` remained in `model_kwargs`;
- removing `bitsandbytes` fixed that path, but PEFT then rejected Colab's stale `torchao 0.10.0`;
- the current notebook uninstalls both `bitsandbytes` and `torchao`, restarts once, loads without `quantization_config`, uses `dtype=...` instead of deprecated `torch_dtype`, and uses `adamw_torch` instead of `paged_adamw_8bit` when 4-bit is off.

If a future agent changes this notebook, preserve the first-cell bootstrap and the `torchao` check. The default repair run expects A100/H100 because the no-4bit path has higher VRAM requirements.

## Evaluation

`evaluate.py` computes validation perplexity, generates samples for the first N validation rows, and measures **citation-presence rate** — the fraction of generated responses for non-refusal rows that include the expected `source_url`. The report is written to `data/eval_report.json`.

```bash
python evaluate.py \
    --model-dir checkpoints/legal-sft \
    --val-file data/val.jsonl \
    --n-samples 5 \
    --out data/eval_report.json
```

## Full example run from raw collection to training

```bash
# 0. environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. multi-source harvest (bdlaws + supreme court + chancery)
python collect_sources.py \
    --sources bdlaws,supremecourt,chancery \
    --volumes 1,2,3 --max-acts 10 --max-children-per-act 20 --delay 2.0

# 2. parse and clean
python parse_sources.py
python clean_dataset.py

# 3. build the supervised dataset and split
python build_dataset.py
python split_data.py

# 4. fine-tune (defaults to a tiny model so the run completes on CPU)
python train_qlora.py --epochs 1 --batch-size 1 --grad-accum 4

# 5. evaluate
python evaluate.py
```

## Safety posture

* The system is positioned as automated **research and drafting support**, never as a lawyer.
* The training examples instruct the model to cite the official portal on every substantive answer.
* The dataset includes explicit **refusal** rows for predictive or unsupported queries.
* `clarification` rows train the model to ask for missing facts instead of inventing them.
* The disclaimer string is injected into every non-refusal response and reminds the user to verify against the official portal and consult a qualified Bangladeshi advocate.

## Source rules and provenance

* Primary source: <http://bdlaws.minlaw.gov.bd/>.
* Every fetched page is recorded in `data/manifest.json` with title, URL, source type, jurisdiction, optional act / section identifiers, retrieval timestamp, and a SHA1 of the raw HTML.
* Cleaned documents are deduplicated by a SHA1 of the normalized body.
* No source text is invented. Pages that cannot be parsed reliably are skipped and logged.
* If you add additional Bangladesh legal repositories, label them clearly in the manifest (e.g., `source_type: "supplementary"`) before parsing.
