"""
merge_and_export.py

Merge Advocore LoRA adapters into the base model and export an FP16 model for
later inference.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_ADAPTER_DIR = "models/advocore-adapters"
DEFAULT_OUTPUT_DIR = "models/advocore-fp16"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s merge_and_export %(message)s",
)
log = logging.getLogger("merge_and_export")


def parse_device_map(value: str):
    if value == "cpu":
        return {"": "cpu"}
    return value


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=os.environ.get("BASE_MODEL", DEFAULT_BASE_MODEL))
    p.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--device-map", default="cpu",
                   help="'cpu' is safest for a 12GB 3060; use 'auto' if you have more VRAM.")
    p.add_argument("--max-shard-size", default="4GB")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    log.info("loading base model in FP16: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map=parse_device_map(args.device_map),
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )

    log.info("loading Advocore adapters from %s", args.adapter_dir)
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        torch_dtype=torch.float16,
    )

    log.info("merging adapters")
    merged = model.merge_and_unload()
    os.makedirs(args.output_dir, exist_ok=True)
    merged.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )

    tokenizer_source = args.adapter_dir if os.path.exists(args.adapter_dir) else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(args.output_dir)
    log.info("saved merged FP16 model to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
