"""
push_to_hub.py

Upload data/train.jsonl and data/val.jsonl as a *private* Hugging Face dataset
that the Colab notebook will load with `datasets.load_dataset`.

Auth:
    export HF_TOKEN=hf_xxx
    # or:  huggingface-cli login

Usage:
    python push_to_hub.py --repo-id <user-or-org>/bd-legal-sft --private
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone

from huggingface_hub import HfApi, create_repo, upload_folder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s push_to_hub %(message)s",
)
log = logging.getLogger("push_to_hub")


DATASET_README = """\
# Bangladesh Legal Assistant — SFT Dataset (private)

Supervised fine-tuning data for a Bangladesh legal research assistant, built
from the official Laws of Bangladesh portal plus the
`sakhadib/Bangladesh-Legal-Acts-Dataset` baseline.

Each row follows the schema:

```
instruction, context, reasoning, response, citations,
source_title, source_url, source_type, jurisdiction,
topic, task_type, confidence, refusal_reason
```

Every non-refusal row carries citations tied to source metadata. This dataset
is intended for **legal research and drafting support only** — not legal
advice.

Splits:
- `train` — `train.jsonl`
- `validation` — `val.jsonl`

Generated on {date} from the Bangladesh legal-assistant pipeline.
"""


def line_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", required=True,
                   help="e.g. tanzir/bd-legal-sft")
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--val", default="data/val.jsonl")
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--public", dest="private", action="store_false")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = p.parse_args()

    n_train = line_count(args.train)
    n_val = line_count(args.val)
    if n_train == 0 or n_val == 0:
        log.error("train/val JSONL is empty (train=%d, val=%d). "
                  "Run build_dataset.py + split_data.py first.",
                  n_train, n_val)
        return 2
    log.info("train rows=%d val rows=%d", n_train, n_val)

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    log.info("repo ready: %s (private=%s)", args.repo_id, args.private)

    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "data"), exist_ok=True)
        # Copy splits into a flat layout the Hub auto-detects.
        import shutil
        shutil.copy(args.train, os.path.join(tmp, "train.jsonl"))
        shutil.copy(args.val, os.path.join(tmp, "val.jsonl"))
        # README + dataset card with split mapping.
        with open(os.path.join(tmp, "README.md"), "w", encoding="utf-8") as f:
            f.write("---\n")
            f.write("license: other\n")
            f.write("language:\n- en\n- bn\n")
            f.write("tags:\n- legal\n- bangladesh\n- sft\n")
            f.write("configs:\n")
            f.write("- config_name: default\n")
            f.write("  data_files:\n")
            f.write("  - split: train\n    path: train.jsonl\n")
            f.write("  - split: validation\n    path: val.jsonl\n")
            f.write("---\n\n")
            f.write(DATASET_README.format(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d")
            ))
        upload_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=tmp,
            token=args.token,
            commit_message=f"Push SFT splits (train={n_train}, val={n_val})",
        )
    log.info("uploaded. view: https://huggingface.co/datasets/%s", args.repo_id)
    print(f"OK https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
