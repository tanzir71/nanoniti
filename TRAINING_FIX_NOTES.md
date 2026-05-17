# Colab Training Fix Notes

Last confirmed working: 2026-05-15, Google Colab T4, `colab_train_v2.ipynb`.

This note exists so future agents do not rediscover the same training-stage failures.

Beginner/operator handoff lives in [`COLAB_TRAINING_SURVIVAL_GUIDE.md`](COLAB_TRAINING_SURVIVAL_GUIDE.md). Use that guide during live Colab runs; this file is the lower-level fix log.

## What Failed

The notebook reached the train cell and failed before `trainer.train(...)`:

```text
warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
TypeError: Trainer.__init__() got an unexpected keyword argument 'tokenizer'
```

Before that, the likely first-train-step crash was:

```text
RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
```

The Colab Jupyter server/kernel warnings about websocket timeout, frozen modules, and schema validation were noise. The useful error was the red traceback from the notebook cell containing `Trainer(...)`.

## Root Causes

1. Recent `transformers` removed or stopped accepting `Trainer(tokenizer=...)`; use `processing_class=tokenizer` when supported.
2. `warmup_ratio` is deprecated in the Colab runtime's `TrainingArguments`; compute and pass `warmup_steps` instead.
3. QLoRA + `prepare_model_for_kbit_training` + gradient checkpointing can leave inputs without gradients unless input grads and non-reentrant checkpointing are explicit.
4. Auto-resuming from a stale or incomplete Hub checkpoint can produce confusing train-stage failures after a failed Colab run.

## What Worked

Use `colab_train_v2.ipynb`, not `colab_train.ipynb`.

Known-good Colab defaults:

```python
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 16
RESUME_FROM_HUB = False
```

For a paid A100/L4 3B run optimized to finish quickly, use `colab_train_qwen25_3b_fast_gpu.ipynb`. It writes to `tanziro/bd-legal-qwen25-3b-fast-lora` and keeps a separate Drive path so it does not collide with the older 3B run. Default profile:

```python
TRAINING_PROFILE = "A100_FAST_FINISH"
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
SUBSET_TRAIN = 20000
BATCH_SIZE = 4
GRAD_ACCUM = 8
MAX_STEPS = 625
SAVE_STEPS = 25
EVAL_STEPS = 0
```

Use `L4_FAST_FINISH` if Colab gives L4. Use `PERSISTENCE_SMOKE` only to prove Drive/Hub persistence. The fast notebook keeps verified adapter uploads and disables generic Trainer Hub pushes.

For the separate 9B experiment, use `colab_train_qwen35_9b.ipynb`. It intentionally keeps a different repo, output path, Drive path, shorter sequence length, lower LoRA rank, paid-GPU profiles, dynamic padding, and verified adapter-only uploads:

```python
BASE_MODEL = "Qwen/Qwen3.5-9B"
HF_OUTPUT_REPO = "tanziro/bd-legal-qwen35-9b-lora"
TRAINING_PROFILE = "A100_FINISH_TODAY"
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/legal-assistant-bd-legal-qwen35-9b-lora"
```

Available 9B profiles:

```text
PERSISTENCE_SMOKE     512 rows, max_len 256, max_steps 10,  save_steps 5,  eval_steps 0
A100_FINISH_TODAY     8k rows,  max_len 384, max_steps 300, save_steps 10, eval_steps 0
L4_FINISH_TODAY       5k rows,  max_len 384, max_steps 180, save_steps 10, eval_steps 0
A100_STRONG_HALF_DAY  16k rows, max_len 384, max_steps 600, save_steps 25, eval_steps 250
```

Do not treat the 9B notebook as a simple search-and-replace copy of v2. It uses the same persistence machinery, but the paid-GPU defaults are bounded because 9B has a much tighter memory and runtime margin. Confirm `PERSISTENCE PREFLIGHT PASSED`, then confirm the first `adapter-checkpoints/checkpoint-*` folder exists on the Hub before leaving a full run unattended. The notebook disables Trainer's generic `push_to_hub` path for the 9B run, relies on the verified emergency-style `upload_adapter_copy(...)` path to avoid duplicate checkpoint uploads, and uses a projected-runtime guard to stop cleanly if the first steps predict another money-burning crawl.

Known-good k-bit prep pattern:

```python
try:
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
except TypeError:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
```

Known-good Trainer construction pattern:

```python
trainer_kwargs = dict(
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    data_collator=collator,
)
if "processing_class" in inspect.signature(Trainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer
trainer = Trainer(**trainer_kwargs)
```

Known-good warmup handling:

```python
steps_per_epoch = (len(ds_tok["train"]) + (BATCH_SIZE * GRAD_ACCUM) - 1) // (BATCH_SIZE * GRAD_ACCUM)
total_train_steps = max(1, int(steps_per_epoch * EPOCHS))
warmup_steps = max(1, int(total_train_steps * WARMUP_RATIO)) if WARMUP_RATIO else 0

args = TrainingArguments(
    ...,
    warmup_steps=warmup_steps,
)
```

## Do Not Revert

- Do not reintroduce `Trainer(..., tokenizer=tokenizer)` directly.
- Do not pass `warmup_ratio` to `TrainingArguments` in Colab.
- Do not pass `group_by_length` or `length_column_name` to Colab `TrainingArguments` on the Transformers 5 runtime; the 9B notebook keeps dynamic padding in the collator instead.
- Do not remove `model.enable_input_require_grads()`.
- Do not remove `gradient_checkpointing_kwargs={"use_reentrant": False}`.
- Do not set `RESUME_FROM_HUB=True` immediately after a failed run unless the latest checkpoint is known-good.

## Verification

Local static verification used for these fixes:

```bash
python -m unittest tests.test_training_infra
python -m py_compile train_qlora.py modal_train.py tests/test_training_infra.py
```

The Colab runtime then advanced past `Trainer(...)` and began training properly.

## Push-Stage Fix

Confirmed issue after training completed: training could finish locally in Colab while nothing obvious appeared on Hugging Face. The safer pattern is to save a clean adapter-only directory and verify the remote repo after upload.

Do not rely only on `Trainer(push_to_hub=True)` or on uploading the whole Trainer output directory. The notebook now saves:

```python
FINAL_ADAPTER_DIR = "/content/legal-sft-final-adapter"
trainer.model.save_pretrained(FINAL_ADAPTER_DIR, safe_serialization=True)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
```

Then it uploads that clean folder and asserts the remote repo contains:

```text
adapter_config.json
adapter_model.safetensors   # or adapter_model.bin
```

If Colab already trained but did not push, run the final push/recovery cell only. No retraining is needed if `/content/legal-sft-out` or `/content/legal-sft-final-adapter` still exists in the live runtime.

## Persistent Backup Rule

After two completed Colab runs produced no saved adapter, the notebook was hardened to use Google Drive as the persistent source of truth:

```python
BACKUP_TO_DRIVE = True
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/legal-assistant-bd-legal-qwen25-3b-lora"
```

The training cell now mounts Drive before training and refuses to continue if Drive cannot mount. A `DriveAdapterBackupCallback` saves LoRA adapter files at every Trainer save event:

```text
DRIVE_BACKUP_DIR/checkpoint-<global_step>/
```

The final adapter is also saved to:

```text
DRIVE_BACKUP_DIR/final-adapter/
```

The notebook also writes verified Hub adapter checkpoints at every save:

```python
SAVE_STEPS = 25
HUB_BACKUP_EVERY_SAVE = True
HUB_CHECKPOINT_PREFIX = "adapter-checkpoints"
```

Each save writes and verifies:

```text
adapter-checkpoints/checkpoint-<global_step>/adapter_config.json
adapter-checkpoints/checkpoint-<global_step>/adapter_model.safetensors
```

Future agents must not disable Drive or Hub checkpoint backup unless the user explicitly accepts the risk. Colab `/content` is never a persistence layer.

Adapter uploads must use the same pattern that worked for the emergency save:

```python
create_repo(..., exist_ok=True)
upload_folder(..., folder_path=adapter_dir, path_in_repo=...)
api.list_repo_files(...)
assert f"{path_in_repo}/adapter_config.json" in remote_files
assert f"{path_in_repo}/adapter_model.safetensors" in remote_files
```

The notebook's `upload_adapter_copy(...)` helper owns this pattern and is used for both checkpoint uploads and the final adapter upload to `final-adapter/`. Do not replace it with a bare `upload_folder(...)` call.

## Continuing From A Partial Adapter

If training was interrupted but an adapter was saved, it is not a full Trainer checkpoint. Optimizer and scheduler state are gone, but the learned LoRA weights are usable. Continue training by loading the adapter trainably:

```python
START_FROM_ADAPTER_REPO = HF_OUTPUT_REPO
START_FROM_ADAPTER_SUBFOLDER = "emergency-current-adapter"
```

The notebook uses:

```python
PeftModel.from_pretrained(
    model,
    START_FROM_ADAPTER_REPO,
    subfolder=START_FROM_ADAPTER_SUBFOLDER,
    token=os.environ["HF_TOKEN"],
    is_trainable=True,
)
```

This continues from the partial LoRA weights with a fresh optimizer. It is not byte-for-byte equivalent to resuming a Trainer checkpoint, but it avoids throwing away the completed training work.

## Qwen3.5-9B Benchmark And Repair Handoff

Last updated: 2026-05-16.

The 9B adapter was benchmarked with `colab_benchmark_qwen35_9b.ipynb` on an A100. The benchmark notebook itself was also fixed during this debugging cycle, so read the results as a sequence:

| Report | Benchmark state | Adapter overall | Base overall | Meaning |
|---|---|---:|---:|---|
| `benchmark_report_qwen35_9b.json` | adapter only, first run | `0.333` | n/a | First signal that the adapter had serious citation/refusal/Bengali issues. |
| `benchmark_report_qwen35_9b (1).json` | baseline true, pre-patch benchmark | `0.333` | `0.083` | Adapter clearly beat base, but the benchmark still had bugs. |
| `benchmark_report_qwen35_9b (2).json` | baseline true, patched benchmark | `0.333` | `0.250` | Cleaner signal: adapter slightly beat base, but remained unsafe. |

Benchmark bugs that were fixed before the latest run:

- Bengali row in `colab_benchmark_qwen35_9b.ipynb` had been corrupted as question marks.
- Predictive-refusal scoring punished "will not definitely overturn" because it contained `definitely overturn`.
- Disclaimer scoring required exact `not a lawyer` even when the model said `not legal advice`.
- One faithfulness row incorrectly passed an adapter answer that began "Yes, the text mentions cybercrime" when the context only discussed murder.
- Response cleanup now strips leaked `</RESPONSE>`, `<SYSTEM>`, `<INSTRUCTION>`, and `<CONTEXT>` continuations before scoring.

Latest patched benchmark category scores:

| Category | Adapter | Base | Takeaway |
|---|---:|---:|---|
| `citation_presence` | `0.0` | `0.0` | Exact official citation behavior is still weak. |
| `refusal_predictive` | `1.0` | `1.0` | Scoring fix made this pass for both. |
| `refusal_personal_advice` | `1.0` | `0.0` | Adapter improved here. |
| `refusal_made_up_section` | `0.0` | `1.0` | Adapter hallucinated fabricated provision URLs; this is the biggest safety issue. |
| `format_disclaimer` | `1.0` | `0.0` | Adapter learned disclaimer/citation style. |
| `bilingual_robustness` | `0.0` | `0.0` | Adapter output was truncated on Bengali. |
| `faithfulness` | `0.5` | `0.5` | Adapter still contradicted context on the cybercrime row. |

Decision: do not retrain the whole 9B run from scratch. Continue from the existing adapter with a targeted repair pass.

The repair notebook is:

```text
colab_repair_qwen35_9b.ipynb
```

It starts from:

```python
SOURCE_ADAPTER_REPO = "tanziro/bd-legal-qwen35-9b-lora"
SOURCE_ADAPTER_SUBFOLDER = "final-adapter"
```

It writes repaired weights to:

```text
repair-v1-final-adapter/
repair-v1-checkpoints/checkpoint-*/
```

This intentionally does not overwrite `final-adapter/`.

## Qwen3.5-9B Repair Runtime Failures

The repair notebook went through several Colab package failures. Future agents should preserve the current environment bootstrap unless they have tested an alternative in Colab.

Failure sequence:

1. Old `transformers` did not recognize the Qwen3.5 config/model type (`model_type: qwen3_5`).
2. `pip install --upgrade transformers` was not enough inside the live Colab kernel; the runtime had to restart before the new package was active.
3. `bitsandbytes` failed with `OSError: libnvJitLink.so.13: cannot open shared object file`.
4. Setting `load_in_4bit=False` did not fix it while `BitsAndBytesConfig` or `quantization_config` still existed in `model_kwargs`.
5. Removing `bitsandbytes` fixed the CUDA loader path, but PEFT then failed during `PeftModel.from_pretrained(...)` because Colab had `torchao 0.10.0` and the installed PEFT required `torchao > 0.16.0` if `torchao` is installed.

Current repair notebook defaults:

```python
USE_4BIT = False
```

The first cell now:

- uninstalls `bitsandbytes`;
- uninstalls `torchao`;
- upgrades `transformers`, `peft`, `accelerate`, and related dependencies;
- writes a new environment marker;
- kills the runtime once so the upgraded packages are actually imported on the next run.

The model load cell now:

- checks that Transformers can import Qwen3.5 support;
- fails early if `torchao` is still installed;
- uses `dtype=...` instead of deprecated `torch_dtype=...`;
- does not import `BitsAndBytesConfig` unless `USE_4BIT=True`;
- does not pass `quantization_config` unless `USE_4BIT=True`;
- freezes the base model manually for the no-4bit path and trains only the loaded PEFT adapter;
- uses `optim="adamw_torch"` when `USE_4BIT=False`.

Do not change the repair notebook back to `paged_adamw_8bit`, `BitsAndBytesConfig`, or `quantization_config` unless `USE_4BIT=True` and the Colab runtime has a known-good `bitsandbytes`/CUDA stack.

Because the no-4bit path uses more VRAM, the default real repair profiles expect A100/H100. Use `PERSISTENCE_SMOKE` only for setup testing.
