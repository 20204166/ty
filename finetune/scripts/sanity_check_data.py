from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from finetune.data import load_training_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, default="app/models/data/text/training_data.json")
    ap.add_argument("--print_examples", type=int, default=3)
    args = ap.parse_args()

    loaded = load_training_json(args.data_path)
    print(f"✅ Loaded {len(loaded.raw)} examples from {args.data_path}")
    print("Task counts:")
    for k, v in sorted(loaded.task_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")

    n = min(args.print_examples, len(loaded.raw))
    print(f"\nShowing {n} examples:")
    for i in range(n):
        ex: Dict[str, Any] = loaded.raw[i]
        print("=" * 80)
        print(f"task: {ex['task']}")
        print("- source:")
        print(ex["source"][:800])
        print("- target:")
        print(ex["target"][:800])

    # Optional: write a small schema summary next to the file
    schema_path = Path(args.data_path).with_suffix(".schema.json")
    schema = {
        "required_keys": ["source", "target", "task"],
        "example_count": len(loaded.raw),
        "task_counts": loaded.task_counts,
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"\nWrote schema summary to: {schema_path}")


if __name__ == "__main__":
    main()
