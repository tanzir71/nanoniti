# Colab Training Survival Guide

This guide is for a complete beginner running the legal assistant training notebooks in Google Colab.

It explains the three things that caused the painful runs:

1. Persistence: how to make sure trained weights actually leave Colab.
2. Resume settings: which setting to use after a disconnect.
3. Runtime overhead: why training can run for hours while doing very little actual training.

Use this guide with:

- `colab_train_v2.ipynb` for the Qwen2.5-3B run.
- `colab_train_qwen35_9b.ipynb` for the Qwen3.5-9B run.

## The One Rule

Colab `/content` is temporary. Treat it like scratch paper.

If the adapter is not in Google Drive or on Hugging Face, it is not safely saved. A completed training progress bar does not matter if the final adapter was never copied out of the runtime.

## Before You Start

You need three things before a real run:

1. A Hugging Face write token available as `HF_TOKEN`.
2. Google Drive mounted successfully.
3. A model repo on Hugging Face where adapter checkpoints can be uploaded.

Never paste a real Hugging Face token into a committed notebook or markdown file. Use the login cell or Colab secrets.

## What Preflight Means

Every real training attempt should print:

```text
PERSISTENCE PREFLIGHT PASSED
```

This only means:

- Colab can write a small file to Google Drive.
- Colab can upload a small file to Hugging Face.
- The Hugging Face token works.

It does not mean:

- training resumed from the previous step,
- the model is already saved,
- the final adapter exists,
- or the run is safe to abandon.

Preflight always runs before training. Seeing preflight again after a disconnect is normal.

## The Folders That Matter

There are three different kinds of saved output. They are not interchangeable.

### Final Adapter

This is the thing you ultimately want.

Hugging Face path:

```text
final-adapter/adapter_config.json
final-adapter/adapter_model.safetensors
```

Google Drive path:

```text
DRIVE_BACKUP_DIR/final-adapter/
```

If this exists and verifies, the run produced a usable final LoRA adapter.

### Adapter Checkpoint

This is the emergency-safe save made during training.

Hugging Face path:

```text
adapter-checkpoints/checkpoint-150/adapter_config.json
adapter-checkpoints/checkpoint-150/adapter_model.safetensors
```

Google Drive path:

```text
DRIVE_BACKUP_DIR/checkpoint-150/
```

This preserves learned LoRA weights at that step. It does not preserve the optimizer, scheduler, or exact Trainer step state.

Use this when Colab disconnected and you want to continue from the learned adapter weights.

### Full Trainer Checkpoint

This is a Hugging Face Trainer checkpoint.

Hugging Face path:

```text
checkpoint-150/
```

This can preserve the exact Trainer state, including optimizer and scheduler state, if the full checkpoint was pushed correctly.

The current safer default is not to rely on this path, because generic Trainer Hub pushes were slow and duplicated upload work. The notebooks prefer verified adapter checkpoints.

## What To Check Before Walking Away

Do not leave a long run unattended until all of these are true:

1. You saw:

   ```text
   PERSISTENCE PREFLIGHT PASSED
   ```

2. You saw a verified adapter checkpoint message like:

   ```text
   Verified adapter-checkpoints/checkpoint-25
   ```

3. In Hugging Face, the folder exists and contains:

   ```text
   adapter_config.json
   adapter_model.safetensors
   ```

4. In Google Drive, the matching checkpoint folder exists.

If only the preflight file exists, training is not protected yet.

## Resume Decision Tree

After a disconnect, first find the latest saved folder.

### Case 1: You Have `final-adapter/`

You are done training. Run evaluation or inference. Do not train again unless you want a better model.

### Case 2: You Have A Top-Level `checkpoint-150/`

Use exact Trainer resume:

```python
RESUME_FROM_HUB = True
START_FROM_ADAPTER_SUBFOLDER = None
```

You should see output like:

```text
found checkpoint on Hub: checkpoint-150 -> downloading to resume
will resume from: /content/legal-sft-out/checkpoint-150
```

If you do not see `will resume from`, it is not doing exact resume.

Only use this if the top-level checkpoint is known-good.

### Case 3: You Have `adapter-checkpoints/checkpoint-150/`

Use adapter continuation:

```python
RESUME_FROM_HUB = False
START_FROM_ADAPTER_SUBFOLDER = "adapter-checkpoints/checkpoint-150"
```

This keeps the learned LoRA weights from checkpoint 150 and continues training with a fresh optimizer.

The progress bar starts from step 0 again. That is expected. It does not mean the learned adapter weights were thrown away.

### Case 4: You Only Have `/content/...` And The Runtime Is Still Alive

Run the final upload or emergency save cell immediately.

Useful local paths:

```text
/content/legal-sft-out
/content/legal-sft-final-adapter
/content/legal-sft-qwen35-9b-out
/content/legal-sft-qwen35-9b-final-adapter
```

If the live Python variables still exist, save from `trainer.model` or `model`.

If the runtime restarted and no adapter exists in Drive or Hugging Face, the training work is not recoverable locally.

## What Not To Do After A Disconnect

Do not assume rerunning the notebook automatically resumes.

Do not set this blindly:

```python
RESUME_FROM_HUB = True
```

That only works for trusted top-level `checkpoint-*` folders. It does not resume from `adapter-checkpoints/checkpoint-*`.

Do not delete old checkpoint folders until the final adapter is verified on Hugging Face.

Do not rely on a screenshot or progress bar as proof of persistence.

## Why A Run Can Take Way Too Long

Training is not the only thing happening during the train cell.

The goal is to spend GPU time on training, not spend hours on eval/upload overhead.

The slow non-training work can include:

- evaluating the validation set,
- saving adapter files,
- uploading adapter files to Hugging Face,
- writing backups to Google Drive,
- verifying remote files,
- generic Trainer Hub pushes,
- loading or snapshotting Hub checkpoints.

The 3B notebook previously did too much of this:

```python
SAVE_STEPS = 25
eval_steps = SAVE_STEPS
push_to_hub = True
HUB_BACKUP_EVERY_SAVE = True
SUBSET_VAL = 1000
```

That meant it could evaluate 1,000 validation rows every 25 optimizer steps and also upload through two different paths. A run could spend a huge amount of wall-clock time evaluating and uploading instead of training.

The updated 3B notebook keeps frequent verified adapter saves but reduces waste:

```python
SAVE_STEPS = 25
EVAL_STEPS = 250
TRAINER_HUB_PUSH = False
HUB_BACKUP_EVERY_SAVE = True
```

Meaning:

- adapter backups still happen every 25 steps,
- full validation eval happens less often,
- generic Trainer Hub pushing is off,
- the safer `upload_adapter_copy(...)` path still verifies adapter files.

## How To Estimate Whether Training Should Finish

The rough optimizer-step count is:

```python
steps_per_epoch = train_rows / (BATCH_SIZE * GRAD_ACCUM)
```

For the 3B defaults:

```text
20,000 rows / (1 * 16) = about 1,250 optimizer steps
```

If the progress bar says:

```text
1100/1250
```

let it finish.

If it says:

```text
200/1250
```

after many hours, it is probably spending too much time on eval/upload overhead or running on a slow GPU.

## What To Do If A Run Is Still Going After Many Hours

Do not panic-stop immediately.

Use this order:

1. Check the progress bar: current step / total step.
2. Check the latest verified adapter checkpoint.
3. If it is close to the end, let it finish.
4. If it is far from the end, wait until the next verified checkpoint appears.
5. Stop only after a verified checkpoint lands.
6. Continue later from that adapter checkpoint.

Example:

```text
Verified adapter-checkpoints/checkpoint-300
```

Then continue with:

```python
RESUME_FROM_HUB = False
START_FROM_ADAPTER_SUBFOLDER = "adapter-checkpoints/checkpoint-300"
```

## 3B Notebook Settings

Use `colab_train_v2.ipynb`.

Important defaults:

```python
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
MAX_LEN = 512
BATCH_SIZE = 1
GRAD_ACCUM = 16
SAVE_STEPS = 25
EVAL_STEPS = 250
TRAINER_HUB_PUSH = False
HUB_BACKUP_EVERY_SAVE = True
RESUME_FROM_HUB = False
START_FROM_ADAPTER_SUBFOLDER = None
```

Expected total steps with the default 20k-row subset:

```text
about 1,250 optimizer steps
```

## Fast 3B Paid-GPU Notebook

Use `colab_train_qwen25_3b_fast_gpu.ipynb` when you are paying for A100/L4 and want the 3B adapter to finish quickly.

It writes to a separate repo and Drive folder:

```text
tanziro/bd-legal-qwen25-3b-fast-lora
/content/drive/MyDrive/legal-assistant-bd-legal-qwen25-3b-fast-lora
```

Profiles:

```text
PERSISTENCE_SMOKE   512 rows,   max_steps 10,  save_steps 5
A100_FAST_FINISH    20k rows,   max_steps 625, batch_size 4, grad_accum 8
L4_FAST_FINISH      16k rows,   max_steps 500, batch_size 2, grad_accum 16
A100_FULL_FAST      full train, no fixed max_steps, for after a first adapter exists
```

The fast notebook skips mid-training eval by default:

```python
EVAL_STEPS = 0
push_to_hub = False
```

It still saves verified adapter checkpoints during training. Do not walk away until you see:

```text
Verified adapter-checkpoints/checkpoint-25
```

## 9B Notebook Settings

Use `colab_train_qwen35_9b.ipynb`.

It has profiles:

```text
PERSISTENCE_SMOKE
A100_FINISH_TODAY
L4_FINISH_TODAY
A100_STRONG_HALF_DAY
```

Use `PERSISTENCE_SMOKE` first if you only want to prove persistence:

```python
TRAINING_PROFILE = "PERSISTENCE_SMOKE"
```

Use the A100 profile for the real paid-GPU run:

```python
TRAINING_PROFILE = "A100_FINISH_TODAY"
```

Use the L4 profile if Colab gives you L4:

```python
TRAINING_PROFILE = "L4_FINISH_TODAY"
```

Use the stronger A100 profile only after the finish-today run has produced a verified adapter:

```python
TRAINING_PROFILE = "A100_STRONG_HALF_DAY"
```

The 9B finish-today profiles use fixed `MAX_STEPS`, skip mid-training eval by default, and save verified adapter checkpoints every 10 optimizer steps. If the first steps predict that the run will exceed the time budget, the notebook requests a save and clean stop instead of silently burning paid runtime.

The 9B notebook refuses a real run on T4. That is intentional. T4 should only be used for smoke testing.

## Safe Stop Checklist

Before interrupting a run, confirm at least one of these exists:

- `final-adapter/` on Hugging Face,
- `adapter-checkpoints/checkpoint-<latest>/` on Hugging Face,
- `DRIVE_BACKUP_DIR/checkpoint-<latest>/` in Google Drive.

Then write down the exact folder name, for example:

```text
adapter-checkpoints/checkpoint-300
```

Use that exact string in:

```python
START_FROM_ADAPTER_SUBFOLDER = "adapter-checkpoints/checkpoint-300"
```

## Good Signs

These are good:

```text
PERSISTENCE PREFLIGHT PASSED
Drive adapter checkpoint step 25 saved
Verified adapter-checkpoints/checkpoint-25
Final clean adapter push verified on Hub at final-adapter
```

## Bad Signs

These need action:

```text
HF_TOKEN missing
No adapter files found
Hub preflight upload failed
Drive preflight write failed
remote adapter_config.json missing
```

If you see one of these, do not start a long run. Fix persistence first.

## Quick Recipes

### Start A Fresh 3B Run

```python
RESUME_FROM_HUB = False
START_FROM_ADAPTER_SUBFOLDER = None
TRAINER_HUB_PUSH = False
HUB_BACKUP_EVERY_SAVE = True
```

Wait for:

```text
PERSISTENCE PREFLIGHT PASSED
Verified adapter-checkpoints/checkpoint-25
```

### Continue A 3B Run From Adapter Checkpoint 150

```python
RESUME_FROM_HUB = False
START_FROM_ADAPTER_SUBFOLDER = "adapter-checkpoints/checkpoint-150"
```

The step counter starts over. That is okay.

### Exact Resume From A Full Trainer Checkpoint

Only if top-level `checkpoint-150/` exists and is trusted:

```python
RESUME_FROM_HUB = True
START_FROM_ADAPTER_SUBFOLDER = None
```

Look for:

```text
will resume from: /content/legal-sft-out/checkpoint-150
```

### Save A Finished Live Runtime

If training finished but the Hub does not show the final adapter, run the final upload or emergency save cell while the runtime is still alive.

The correct upload path verifies:

```text
adapter_config.json
adapter_model.safetensors
```

on Hugging Face before declaring success.

## Agent Handoff Rules

Future agents should not change these without explicit user approval:

- Keep Drive backup enabled for Colab runs.
- Keep verified Hub adapter checkpoints enabled.
- Keep `upload_adapter_copy(...)` for adapter uploads.
- Do not reintroduce bare `Trainer(tokenizer=...)`.
- Do not pass `warmup_ratio` directly to Colab `TrainingArguments`.
- Do not pass `group_by_length` or `length_column_name` to Colab `TrainingArguments` on the Transformers 5 runtime.
- Keep `model.enable_input_require_grads()`.
- Keep `gradient_checkpointing_kwargs={"use_reentrant": False}`.
- Do not claim a run resumed unless the logs show either `will resume from:` or `continuing from adapter:`.
