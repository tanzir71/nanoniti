---
license: apache-2.0
base_model: Qwen/Qwen3.5-9B
library_name: peft
pipeline_tag: text-generation
tags:
  - peft
  - lora
  - qlora
  - qwen
  - qwen3.5
  - bangladesh
  - legal
  - legal-research
  - citation
language:
  - en
  - bn
datasets:
  - tanziro/bd-legal-sft
---

# Bangladesh Legal Assistant Qwen3.5-9B LoRA

This repository contains a **LoRA/QLoRA adapter** for `Qwen/Qwen3.5-9B`, fine-tuned for Bangladesh legal research assistance.

It is **not** a standalone full model. Load it with the base model and PEFT.

## Intended Use

This adapter is intended for:

- Bangladesh legal research support
- cited summaries of statutes and sections
- plain-language explanations of legal provisions
- issue spotting from provided legal context
- refusal behavior when evidence is insufficient

This adapter is **not** a lawyer and does **not** provide legal advice. Users must verify every citation against the official Laws of Bangladesh portal and consult a qualified Bangladeshi advocate before acting.

## Base Model

- Base model: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B)
- Base model license: Apache 2.0
- Adapter type: LoRA / QLoRA
- Adapter repo: `tanziro/bd-legal-qwen35-9b-lora`

The base model is multimodal. This adapter was trained on text-only legal SFT examples.

## Training Data

Dataset:

- `tanziro/bd-legal-sft`

The dataset is built from Bangladesh legal sources and structured into supervised instruction rows. Non-refusal rows include citation metadata such as source title, source URL, act ID, section ID, and retrieval timestamp. Refusal rows are included for unsupported, predictive, or personal legal-advice style prompts.

The dataset is designed to teach the model to:

- answer only when source evidence is available,
- cite official legal source URLs,
- refuse unsupported requests,
- avoid pretending to be a lawyer.

## Training Recipe

Default 9B Colab profile used by the training notebook:

```text
Notebook: colab_train_qwen35_9b.ipynb
Profile: A100_FINISH_TODAY
Base model: Qwen/Qwen3.5-9B
Training method: QLoRA
Quantization: 4-bit NF4
Max sequence length: 384
Per-device batch size: 1
Gradient accumulation: 16
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.05
Optimizer: paged_adamw_8bit
Gradient checkpointing: enabled, non-reentrant
Max steps: 300
Save steps: 10
Mid-training eval: disabled
```

LoRA target modules:

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

The training notebook saves verified adapter checkpoints to both Google Drive and Hugging Face during training, then uploads a clean final adapter under:

```text
final-adapter/
```

### Repair Passes

The first benchmarked adapter was not considered production-ready. It learned some citation/disclaimer style, but still hallucinated fabricated provisions such as `section=9999`, failed exact citation rows, and had weak Bengali behavior.

The first targeted correction notebook is:

```text
colab_repair_qwen35_9b.ipynb
```

It continues from:

```text
final-adapter/
```

and saves repaired weights under:

```text
repair-v1-final-adapter/
repair-v1-checkpoints/checkpoint-*/
```

The second targeted correction notebook is:

```text
colab_repair_qwen35_9b_v2_citation_patch.ipynb
```

It continues from:

```text
repair-v1-final-adapter/
```

and saves the current best benchmarked adapter under:

```text
repair-v2-citation-final-adapter/
repair-v2-citation-checkpoints/checkpoint-*/
```

The repair notebooks do not overwrite `final-adapter/` or each other. To benchmark the current repaired adapter, run the benchmark with:

```python
ADAPTER_SUBFOLDER = "repair-v2-citation-final-adapter"
```

### Colab Repair Runtime Notes

The repair run hit several Colab environment failures. Future maintainers should not reintroduce the broken path:

- old `transformers` did not recognize `model_type: qwen3_5`;
- after upgrading `transformers`, Colab had to restart before the live kernel used the new version;
- `bitsandbytes` failed with `libnvJitLink.so.13`;
- disabling 4-bit was insufficient while `BitsAndBytesConfig` or `quantization_config` was still passed to the loader;
- PEFT then failed because Colab had `torchao 0.10.0`, while the installed PEFT expected `torchao > 0.16.0` if `torchao` was present.

The current repair notebook defaults to:

```text
USE_4BIT = False
Optimizer: adamw_torch
Runtime: A100/H100 recommended
Bootstrap: uninstall bitsandbytes and torchao, upgrade transformers/peft/accelerate, restart once
Model load: no BitsAndBytesConfig and no quantization_config unless USE_4BIT=True
```

This is deliberate. The notebook is slower and uses more VRAM than QLoRA, but it avoids the Colab CUDA/package conflict that blocked adapter loading.

## How To Load

Install dependencies:

```bash
pip install "transformers>=5.0.0" "peft>=0.12.0" "accelerate>=0.33.0" "bitsandbytes>=0.43.0"
```

Load the base model in 4-bit and attach the adapter:

```python
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model = "Qwen/Qwen3.5-9B"
adapter_repo = "tanziro/bd-legal-qwen35-9b-lora"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
)

try:
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
except Exception:
    base = AutoModelForImageTextToText.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )

model = PeftModel.from_pretrained(
    base,
    adapter_repo,
    subfolder="repair-v2-citation-final-adapter",
)
model.eval()
```

Example prompt format:

```python
SYSTEM_PROMPT = (
    "You are a Bangladesh legal research assistant. You are not a lawyer. "
    "Cite the official Laws of Bangladesh portal. "
    "Refuse if the retrieved evidence is insufficient."
)

def render_prompt(instruction, context=""):
    return (
        f"<SYSTEM>{SYSTEM_PROMPT}</SYSTEM>\n"
        f"<INSTRUCTION>{instruction}</INSTRUCTION>\n"
        f"<CONTEXT>{context}</CONTEXT>\n"
        f"<RESPONSE>"
    )

prompt = render_prompt(
    "What does section 302 of the Penal Code, 1860 provide? Quote the operative text and cite the source."
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=320,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Evaluation

Evaluation should be run with the repository benchmark before public release:

```bash
python benchmark.py \
  --base-model Qwen/Qwen3.5-9B \
  --adapter-repo tanziro/bd-legal-qwen35-9b-lora \
  --adapter-subfolder repair-v2-citation-final-adapter \
  --test-set benchmark/test_set.jsonl \
  --out-json benchmark/benchmark_report_qwen35_9b.json \
  --out-md benchmark/benchmark_report_qwen35_9b.md \
  --baseline
```

### Current Results

The following results came from the live debugging session. They are a benchmark history, not a release-quality claim.

#### Benchmark Log

| Report | Benchmark state | Adapter overall | Base overall | Notes |
|---|---|---:|---:|---|
| `benchmark_report_qwen35_9b.json` | adapter only, first run | `0.333` | n/a | Exposed weak citation, fake-section refusal, disclaimer, Bengali, and faithfulness behavior. |
| `benchmark_report_qwen35_9b (1).json` | baseline true, pre-patch benchmark | `0.333` | `0.083` | Adapter beat base, but benchmark scoring was flawed: Bengali prompt was corrupted as `????????`, predictive-refusal scoring was too literal, and one faithfulness row passed incorrectly. |
| `benchmark_report_qwen35_9b (2).json` | baseline true, patched benchmark | `0.333` | `0.250` | Cleaner signal: adapter slightly beat base but remained unsafe on fake provisions and weak on exact citations. |
| `benchmark_report_qwen35_9b (3).json` | `repair-v1-final-adapter`, baseline true | `0.583` | `0.250` | Improved disclaimer, refusal, Bengali, and faithfulness behavior, but most citation rows still omitted exact act-title tokens and fake-section refusal still leaked a fabricated URL. |
| `benchmark_report_qwen35_9b (4).json` | `repair-v2-citation-final-adapter`, baseline true | `0.917` | `0.250` | Citation, format, Bengali, and faithfulness rows passed; the remaining adapter failure is fabricated-section refusal for `section=9999`. |

#### Latest Benchmark

| Category | Adapter | Base | Delta |
|---|---:|---:|---:|
| citation_presence | `1.0` | `0.0` | `+1.0` |
| refusal_predictive | `1.0` | `1.0` | `+0.0` |
| refusal_personal_advice | `1.0` | `0.0` | `+1.0` |
| refusal_made_up_section | `0.0` | `1.0` | `-1.0` |
| format_disclaimer | `1.0` | `0.0` | `+1.0` |
| bilingual_robustness | `1.0` | `0.0` | `+1.0` |
| faithfulness | `1.0` | `0.5` | `+0.5` |
| overall | `0.917` | `0.250` | `+0.667` |

Interpretation:

- `repair-v2-citation-final-adapter/` fixed the prior exact-citation, Bengali, format, and context-faithfulness benchmark failures.
- It still failed the fabricated-section test by producing a rule and URL for `section=9999` of the Penal Code.
- Treat the adapter as a research prototype that requires retrieval, citation verification, and an unsupported-provision guardrail before any user-facing legal workflow.

## Limitations

- The adapter is trained for research assistance, not legal advice.
- It may produce incorrect, outdated, incomplete, or hallucinated legal statements.
- It should not be used without retrieval, citation checking, and human review.
- The model may fail on recent legal changes not present in the training data.
- It may not handle all Bengali legal phrasing reliably.
- It was trained on text-only examples even though the base model is multimodal.

## Safety Behavior

The training data includes refusals for:

- requests for personal legal advice,
- requests to predict court outcomes,
- unsupported or fabricated legal provisions,
- questions where evidence is insufficient.

Expected safe behavior:

```text
I do not have enough reliable source text to answer that. Please check the official Laws of Bangladesh portal or consult a qualified Bangladeshi advocate.
```

## Citation Policy

Substantive answers should cite the official Laws of Bangladesh portal when applicable.

Users should verify citations manually at:

```text
http://bdlaws.minlaw.gov.bd/
```

## Files

Preferred inference target:

```text
repair-v2-citation-final-adapter/adapter_config.json
repair-v2-citation-final-adapter/adapter_model.safetensors
repair-v2-citation-final-adapter/tokenizer.json
repair-v2-citation-final-adapter/tokenizer_config.json
```

Earlier adapters and training backups may also exist under:

```text
final-adapter/
repair-v1-final-adapter/
adapter-checkpoints/checkpoint-*/
repair-v1-checkpoints/checkpoint-*/
repair-v2-citation-checkpoints/checkpoint-*/
```

These are useful for recovery and comparison, but they are not the recommended inference target. Prefer `repair-v2-citation-final-adapter/` for the latest benchmarked behavior.

## License

This adapter follows the Apache 2.0 license metadata of the base model. Check the base model and dataset repositories for their own license and usage terms.

## Disclaimer

This model is for legal research and drafting support only. It is not a lawyer and does not replace professional legal advice. Verify all citations against official sources before relying on any output.
