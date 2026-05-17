from __future__ import annotations

import ast
import importlib.util
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_source(name: str) -> str:
    notebook = json.loads(read(name))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


class TrainingInfraTests(unittest.TestCase):
    def test_qlora_script_has_3060_safe_quantization_and_lora_settings(self) -> None:
        src = read("train_qlora.py")

        self.assertIn("BitsAndBytesConfig", src)
        self.assertIn("load_in_4bit=True", src)
        self.assertIn('bnb_4bit_quant_type="nf4"', src)
        self.assertIn("r=16", src)
        self.assertIn("lora_alpha=32", src)
        self.assertIn("paged_adamw_32bit", src)
        self.assertIn("gradient_checkpointing", src)

        for module in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ):
            self.assertIn(module, src)

    def test_qlora_gradient_checkpointing_uses_non_reentrant_inputs(self) -> None:
        src = read("train_qlora.py")
        training_args = src[src.index("TrainingArguments("):]

        self.assertIn("model.enable_input_require_grads()", src)
        self.assertIn('gradient_checkpointing_kwargs={"use_reentrant": False}', training_args)

    def test_colab_v2_uses_t4_safe_checkpointing_settings(self) -> None:
        src = notebook_source("colab_train_v2.ipynb")

        self.assertRegex(src, r"MAX_LEN\s*=\s*512\b")
        self.assertRegex(src, r"BATCH_SIZE\s*=\s*1\b")
        self.assertIn("model.enable_input_require_grads()", src)
        self.assertIn("gradient_checkpointing_kwargs={'use_reentrant': False}", src)
        self.assertIn("use_gradient_checkpointing=True", src)
        self.assertIn("gradient_checkpointing_kwargs={'use_reentrant': False}", src)
        self.assertRegex(src, r"EVAL_STEPS\s*=\s*250\b")
        self.assertRegex(src, r"TRAINER_HUB_PUSH\s*=\s*False\b")
        self.assertIn("push_to_hub=TRAINER_HUB_PUSH", src)
        self.assertRegex(src, r"RESUME_FROM_HUB\s*=\s*False\b")
        self.assertIn("if RESUME_FROM_HUB and ckpts:", src)

    def test_trainers_use_processing_class_compatibility(self) -> None:
        colab_src = notebook_source("colab_train_v2.ipynb")
        modal_src = read("modal_train.py")
        qlora_src = read("train_qlora.py")

        for src in (colab_src, modal_src, qlora_src):
            self.assertIn("processing_class", src)
            self.assertIn("Trainer.__init__", src)

    def test_colab_final_push_verifies_remote_adapter_files(self) -> None:
        src = notebook_source("colab_train_v2.ipynb")

        self.assertIn("FINAL_HUB_SUBFOLDER", src)
        self.assertIn("FINAL_ADAPTER_DIR", src)
        self.assertIn("trainer.model.save_pretrained(FINAL_ADAPTER_DIR", src)
        self.assertIn("upload_adapter_copy(FINAL_ADAPTER_DIR, FINAL_HUB_SUBFOLDER", src)
        self.assertIn("api.list_repo_files(HF_OUTPUT_REPO", src)
        self.assertIn("adapter_config.json", src)
        self.assertIn("adapter_model.safetensors", src)
        self.assertIn("remote adapter_config.json missing", src)

    def test_colab_saves_drive_backups_during_training(self) -> None:
        src = notebook_source("colab_train_v2.ipynb")

        self.assertIn("BACKUP_TO_DRIVE", src)
        self.assertRegex(src, r"SAVE_STEPS\s*=\s*25\b")
        self.assertIn("DRIVE_BACKUP_DIR", src)
        self.assertIn("drive.mount('/content/drive')", src)
        self.assertIn("DriveAdapterBackupCallback", src)
        self.assertIn("def on_save(", src)
        self.assertIn("save_adapter_copy(model, checkpoint_dir", src)
        self.assertIn("DRIVE_FINAL_ADAPTER_DIR", src)
        self.assertIn("verify_adapter_dir(DRIVE_FINAL_ADAPTER_DIR", src)

    def test_colab_saves_verified_hub_checkpoints_during_training(self) -> None:
        src = notebook_source("colab_train_v2.ipynb")

        self.assertIn("HUB_BACKUP_EVERY_SAVE", src)
        self.assertIn("HUB_CHECKPOINT_PREFIX", src)
        self.assertIn("upload_adapter_copy(", src)
        self.assertIn("create_repo(", src)
        self.assertIn("for attempt in range(1, 5):", src)
        self.assertIn("verify_remote_adapter(path_in_repo", src)
        self.assertIn("adapter-checkpoints/checkpoint-", src)

    def test_colab_runs_persistence_preflight_before_training(self) -> None:
        src = notebook_source("colab_train_v2.ipynb")

        self.assertIn("run_persistence_preflight()", src)
        self.assertIn("persistence-probes/latest.json", src)
        self.assertIn("drive-preflight.json", src)
        self.assertIn("PERSISTENCE PREFLIGHT PASSED", src)

    def test_colab_can_continue_from_saved_adapter(self) -> None:
        src = notebook_source("colab_train_v2.ipynb")

        self.assertIn("START_FROM_ADAPTER_REPO", src)
        self.assertIn("START_FROM_ADAPTER_SUBFOLDER", src)
        self.assertIn("PeftModel.from_pretrained", src)
        self.assertIn("continuing from adapter", src)

    def test_qwen35_9b_notebook_has_paid_gpu_profile_and_persistence_path(self) -> None:
        src = notebook_source("colab_train_qwen35_9b.ipynb")

        self.assertIn("Qwen/Qwen3.5-9B", src)
        self.assertIn("bd-legal-qwen35-9b-lora", src)
        self.assertIn("TRAINING_PROFILE = 'A100_FINISH_TODAY'", src)
        self.assertIn("'PERSISTENCE_SMOKE'", src)
        self.assertIn("'A100_FINISH_TODAY'", src)
        self.assertIn("'L4_FINISH_TODAY'", src)
        self.assertIn("'A100_STRONG_HALF_DAY'", src)
        self.assertIn("'MAX_LEN': 384", src)
        self.assertIn("'GRAD_ACCUM': 16", src)
        self.assertIn("'LORA_R': 8", src)
        self.assertIn("'SAVE_STEPS': 10", src)
        self.assertIn("'EVAL_STEPS': 0", src)
        self.assertIn("'MAX_STEPS': 300", src)
        self.assertIn("'SUBSET_TRAIN': 8000", src)
        self.assertIn("padding=False", src)
        self.assertIn("pad_to_multiple_of=8", src)
        self.assertNotIn("group_by_length", src)
        self.assertNotIn("length_column_name", src)
        self.assertIn("push_to_hub=False", src)
        self.assertIn("WallClockStopCallback", src)
        self.assertIn("ProjectedTimeGuardCallback", src)
        self.assertIn("ABORT_IF_PROJECTED_OVER_HOURS", src)
        self.assertIn("REQUIRE_FAST_GPU", src)
        self.assertIn("legal-assistant-bd-legal-qwen35-9b-lora", src)
        self.assertIn("PERSISTENCE PREFLIGHT PASSED", src)
        self.assertIn("upload_adapter_copy(FINAL_ADAPTER_DIR, FINAL_HUB_SUBFOLDER", src)
        self.assertIn("START_FROM_ADAPTER_SUBFOLDER", src)
        self.assertIn("AutoModelForImageTextToText", src)

    def test_qwen25_3b_fast_gpu_notebook_is_bounded_and_persistent(self) -> None:
        src = notebook_source("colab_train_qwen25_3b_fast_gpu.ipynb")

        self.assertIn("Qwen/Qwen2.5-3B-Instruct", src)
        self.assertIn("bd-legal-qwen25-3b-fast-lora", src)
        self.assertIn("TRAINING_PROFILE = 'A100_FAST_FINISH'", src)
        self.assertIn("'PERSISTENCE_SMOKE'", src)
        self.assertIn("'A100_FAST_FINISH'", src)
        self.assertIn("'L4_FAST_FINISH'", src)
        self.assertIn("'A100_FULL_FAST'", src)
        self.assertIn("'BATCH_SIZE': 4", src)
        self.assertIn("'GRAD_ACCUM': 8", src)
        self.assertIn("'SAVE_STEPS': 25", src)
        self.assertIn("'EVAL_STEPS': 0", src)
        self.assertIn("'MAX_STEPS': 625", src)
        self.assertIn("REQUIRE_FAST_GPU", src)
        self.assertIn("ProjectedTimeGuardCallback", src)
        self.assertIn("padding=False", src)
        self.assertIn("pad_to_multiple_of=8", src)
        self.assertNotIn("group_by_length", src)
        self.assertNotIn("length_column_name", src)
        self.assertIn("push_to_hub=False", src)
        self.assertIn("PERSISTENCE PREFLIGHT PASSED", src)
        self.assertIn("upload_adapter_copy(FINAL_ADAPTER_DIR, FINAL_HUB_SUBFOLDER", src)

    def test_qwen35_9b_huggingface_model_card_is_separate_and_complete(self) -> None:
        src = read("model_cards/bd-legal-qwen35-9b-lora/README.md")

        for phrase in (
            "base_model: Qwen/Qwen3.5-9B",
            "license: apache-2.0",
            "library_name: peft",
            "tanziro/bd-legal-qwen35-9b-lora",
            "tanziro/bd-legal-sft",
            "A100_FINISH_TODAY",
            "final-adapter/",
            "adapter-checkpoints/checkpoint-*/",
            "not legal advice",
            "python benchmark.py",
            "AutoModelForImageTextToText",
        ):
            self.assertIn(phrase, src)

    def test_benchmark_supports_hardened_adapter_subfolder_and_qwen35_loader(self) -> None:
        src = read("benchmark.py")

        self.assertIn("--adapter-subfolder", src)
        self.assertIn("adapter_subfolder", src)
        self.assertIn("AutoModelForImageTextToText", src)
        self.assertIn("subfolder=adapter_subfolder", src)

    def test_qwen35_9b_benchmark_notebook_is_self_contained_for_colab(self) -> None:
        src = notebook_source("colab_benchmark_qwen35_9b.ipynb")

        for phrase in (
            "Qwen/Qwen3.5-9B",
            "tanziro/bd-legal-qwen35-9b-lora",
            "ADAPTER_SUBFOLDER = 'final-adapter'",
            "AutoModelForImageTextToText",
            "BitsAndBytesConfig",
            "TEST_ROWS = [",
            "RUN_BASELINE = False",
            "benchmark_report_qwen35_9b.json",
            "benchmark_report_qwen35_9b.md",
            "files.download",
            "HF_TOKEN",
            "must_contain_any",
            "must_not_contain_regex",
            "cleanup_response",
            "বাংলাদেশের দণ্ডবিধি",
        ):
            self.assertIn(phrase, src)
        self.assertNotIn("????????", src)

    def test_qwen35_9b_repair_notebook_continues_existing_adapter_safely(self) -> None:
        src = notebook_source("colab_repair_qwen35_9b.ipynb")

        for phrase in (
            "Qwen/Qwen3.5-9B",
            "SOURCE_ADAPTER_SUBFOLDER = 'final-adapter'",
            "FINAL_HUB_SUBFOLDER = 'repair-v1-final-adapter'",
            "HUB_CHECKPOINT_PREFIX = 'repair-v1-checkpoints'",
            "USE_4BIT = False",
            "AUTO_RESUME_FROM_LATEST_REPAIR_HUB = True",
            "PERSISTENCE PREFLIGHT PASSED",
            "upload_adapter_copy",
            "AdapterPersistenceCallback",
            "'uninstall', '-y', 'bitsandbytes'",
            "'uninstall', '-y', 'torchao'",
            "torchao not installed, as intended",
            "Qwen3.5 support check passed",
            "prepare_model_for_kbit_training",
            "model.enable_input_require_grads()",
            "gradient_checkpointing_kwargs={'use_reentrant': False}",
            "optim='paged_adamw_8bit' if USE_4BIT else 'adamw_torch'",
            "push_to_hub=False",
            "must not invent",
            "section=9999",
            "does not mention cybercrime",
            "বাংলাদেশের দণ্ডবিধি",
            "not legal advice",
            "emergency save",
        ):
            self.assertIn(phrase, src)
        self.assertNotIn("group_by_length", src)
        self.assertNotIn("warmup_ratio", src)
        self.assertNotIn("????????", src)

    def test_colab_survival_guide_documents_persistence_resume_and_overhead(self) -> None:
        src = read("COLAB_TRAINING_SURVIVAL_GUIDE.md")

        for phrase in (
            "Colab `/content` is temporary",
            "PERSISTENCE PREFLIGHT PASSED",
            "adapter-checkpoints/checkpoint-",
            "RESUME_FROM_HUB = True",
            "START_FROM_ADAPTER_SUBFOLDER",
            "TRAINER_HUB_PUSH = False",
            "EVAL_STEPS = 250",
            "upload_adapter_copy(...)",
            "wait until the next verified checkpoint appears",
            "spend hours on eval/upload overhead",
        ):
            self.assertIn(phrase, src)

        self.assertIn("COLAB_TRAINING_SURVIVAL_GUIDE.md", read("README.md"))
        self.assertIn("COLAB_TRAINING_SURVIVAL_GUIDE.md", read("GUIDE.md"))

    def test_no_concrete_huggingface_tokens_are_committed(self) -> None:
        for path in (
            ROOT / ".gitignore",
            ROOT / "GUIDE.md",
            ROOT / "README.md",
            ROOT / "modal_train.py",
            ROOT / "TRAINING_FIX_NOTES.md",
            ROOT / "COLAB_TRAINING_SURVIVAL_GUIDE.md",
            ROOT / "model_cards" / "bd-legal-qwen35-9b-lora" / "README.md",
        ):
            src = path.read_text(encoding="utf-8")
            for match in re.finditer(r"hf_[A-Za-z0-9]{20,}", src):
                token = match.group(0)
                self.assertTrue(
                    set(token[3:]) == {"x"},
                    f"{path.relative_to(ROOT)} contains a concrete Hugging Face token",
                )

    def test_qlora_defaults_match_advocore_training_contract(self) -> None:
        tree = ast.parse(read("train_qlora.py"))
        constants = {
            node.targets[0].id: node.value.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }

        self.assertEqual(constants["PROJECT_NAME"], "Advocore")
        self.assertEqual(constants["DEFAULT_OUTPUT_DIR"], "models/advocore-adapters")
        self.assertEqual(constants["DEFAULT_MAX_SEQ_LENGTH"], 2048)
        self.assertEqual(constants["DEFAULT_BATCH_SIZE"], 1)
        self.assertEqual(constants["DEFAULT_GRAD_ACCUM"], 4)
        self.assertEqual(constants["DEFAULT_EPOCHS"], 5.0)
        self.assertEqual(constants["DEFAULT_WARMUP_RATIO"], 0.10)
        self.assertEqual(constants["DEFAULT_EVAL_STEPS"], 100)

    def test_merge_script_exports_fp16_merged_model(self) -> None:
        src = read("merge_and_export.py")

        self.assertIn("PeftModel.from_pretrained", src)
        self.assertIn("merge_and_unload", src)
        self.assertIn("torch_dtype=torch.float16", src)
        self.assertIn("models/advocore-adapters", src)
        self.assertIn("models/advocore-fp16", src)

    def test_dataset_generator_emits_three_dense_rows_and_complex_reasoning(self) -> None:
        build_dataset = load_module("build_dataset")
        doc = {
            "source_title": "The Penal Code, 1860 - section 300",
            "source_type": "section_page",
            "jurisdiction": "Bangladesh",
            "url": "http://example.test/penal#section=300",
            "act_id": "11",
            "section_id": "300",
            "retrieved_at": "2026-05-15T00:00:00+00:00",
            "body": (
                "Whoever causes death by doing an act with the intention of causing death "
                "commits culpable homicide amounting to murder in the circumstances described."
            ),
            "clauses": [{"marker": "1", "text": "The act must cause death."}],
        }
        sibling = {
            **doc,
            "source_title": "The Penal Code, 1860 - section 299",
            "url": "http://example.test/penal#section=299",
            "section_id": "299",
            "body": "Section 299 defines culpable homicide.",
        }

        rows = build_dataset.make_rows_for_doc(doc, siblings=[sibling])

        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["task_type"] for row in rows},
            {"plain_language_explanation", "legal_issue_spotting", "comparative_analysis"},
        )
        self.assertTrue(all(row["citations"] for row in rows))
        self.assertTrue(all(row.get("reasoning") for row in rows))


if __name__ == "__main__":
    unittest.main()
