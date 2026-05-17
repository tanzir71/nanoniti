"""
split_data.py

Stratified train/validation split of data/dataset.jsonl, stratified by
task_type so each split contains every task class. Writes data/train.jsonl
and data/val.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s split_data %(message)s",
)
log = logging.getLogger("split_data")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="inp", default="data/dataset.jsonl")
    p.add_argument("--train", default="data/train.jsonl")
    p.add_argument("--val", default="data/val.jsonl")
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not os.path.exists(args.inp):
        log.error("input not found: %s", args.inp)
        return 1

    rows = []
    with open(args.inp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        log.error("no rows to split")
        return 1

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r.get("task_type", "other")].append(r)

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for task, items in by_task.items():
        rng.shuffle(items)
        cut = max(1, int(len(items) * args.val_frac))
        val_rows.extend(items[:cut])
        train_rows.extend(items[cut:])
        log.info("task=%s total=%d val=%d train=%d", task, len(items), cut, len(items) - cut)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    os.makedirs(os.path.dirname(args.train) or ".", exist_ok=True)
    with open(args.train, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.val, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("wrote train=%d val=%d", len(train_rows), len(val_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
