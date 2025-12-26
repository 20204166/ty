from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from finetune.data import (
    load_training_json,
    task_counts,
    to_trl_prompt_completion_dataset,
    train_val_split,
)
from finetune.models import build_model_bundle, make_lora_config
from finetune.utils import RunInfo, brief_system_info, ensure_dir, now_iso, set_seed, write_json


DEFAULT_DATA_PATH = "app/models/data/text/training_data.json"
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Single-GPU QLoRA SFT (mixed-task) training on generator JSON.")

    ap.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--output_dir", type=str, default="outputs/qwen2.5-7b-instruct-lora-4bit")

    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)

    # Sequence length
    ap.add_argument("--max_seq_length", type=int, default=2048)

    # Training hyperparams (RunPod-friendly defaults)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_total_limit", type=int, default=3)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="",
        help="Comma-separated module names. If empty, auto-select for Qwen/Llama.",
    )

    # Optional: filter tasks (for later)
    ap.add_argument(
        "--only_tasks",
        type=str,
        default="",
        help="Comma-separated task labels to keep (e.g. summarization,code_cpp,math). Empty keeps all.",
    )

    # Optional perf toggles
    ap.add_argument("--use_flash_attn_2", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)

    ap.add_argument("--resume_from_checkpoint", type=str, default="")

    return ap.parse_args()


def filter_by_tasks(data: List[Dict[str, Any]], only_tasks_csv: str) -> List[Dict[str, Any]]:
    if not only_tasks_csv.strip():
        return data
    allowed = {t.strip() for t in only_tasks_csv.split(",") if t.strip()}
    if not allowed:
        return data
    filtered = [ex for ex in data if str(ex.get("task", "")).strip() in allowed]
    if len(filtered) == 0:
        raise ValueError(f"--only_tasks filtered out all data. Allowed={sorted(allowed)}")
    return filtered


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this training pipeline (single GPU). "
            "Your environment reports torch.cuda.is_available() == False."
        )

    ensure_dir(args.output_dir)
    set_seed(args.seed)

    # Load + validate JSON
    loaded = load_training_json(args.data_path)
    raw = filter_by_tasks(loaded.raw, args.only_tasks)

    train_raw, val_raw = train_val_split(raw, val_ratio=args.val_ratio, seed=args.seed)

    # Log counts per task
    counts_all = task_counts(raw)
    counts_train = task_counts(train_raw)
    counts_val = task_counts(val_raw)

    print("\n=== Dataset summary ===")
    print(f"Data path: {args.data_path}")
    print(f"Total: {len(raw)} | Train: {len(train_raw)} | Val: {len(val_raw)}")
    print("Task counts (total):")
    for k, v in sorted(counts_all.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    print("Task counts (train):")
    for k, v in sorted(counts_train.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    print("Task counts (val):")
    for k, v in sorted(counts_val.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")

    # Convert to TRL dataset format (prompt-completion conversational)
    train_ds = to_trl_prompt_completion_dataset(train_raw)
    val_ds = to_trl_prompt_completion_dataset(val_raw)

    # Load model/tokenizer (4-bit) + LoRA config
    bundle = build_model_bundle(model_id=args.model_id, use_flash_attn_2=args.use_flash_attn_2)

    target_modules: Optional[List[str]] = None
    if args.lora_target_modules.strip():
        target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]

    peft_cfg: LoraConfig = make_lora_config(
        model=bundle.model,
        model_id=args.model_id,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    # TRL SFT config
    # Note: for prompt-completion datasets, loss is computed on completion by default.
    # This preserves instruction-following behavior and reduces "prompt overfitting".
    bf16 = torch.cuda.is_bf16_supported()
    fp16 = not bf16

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=bf16,
        fp16=fp16,
        report_to=["tensorboard"],
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        dataloader_num_workers=2,
    )

    # Metadata logs
    run_info = RunInfo(
        started_at_unix=int(os.path.getmtime(args.data_path)) if Path(args.data_path).exists() else 0,
        started_at_iso=now_iso(),
        seed=args.seed,
        model_id=args.model_id,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )

    write_json(
        Path(args.output_dir) / "run_info.json",
        {
            "run_info": run_info.__dict__,
            "system": brief_system_info(),
            "counts": {
                "total": counts_all,
                "train": counts_train,
                "val": counts_val,
            },
            "args": vars(args),
        },
    )

    # Trainer
    trainer = SFTTrainer(
        model=bundle.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=bundle.tokenizer,
        peft_config=peft_cfg,
    )

    resume = args.resume_from_checkpoint.strip() or None
    if resume is not None and not Path(resume).exists():
        raise FileNotFoundError(f"--resume_from_checkpoint path does not exist: {resume}")

    trainer.train(resume_from_checkpoint=resume)

    # Save adapters + tokenizer
    trainer.model.save_pretrained(args.output_dir)
    bundle.tokenizer.save_pretrained(args.output_dir)

    # Save final metrics (if present)
    metrics = getattr(trainer, "state", None)
    write_json(
        Path(args.output_dir) / "trainer_state_summary.json",
        {
            "global_step": getattr(metrics, "global_step", None),
            "epoch": getattr(metrics, "epoch", None),
            "best_metric": getattr(metrics, "best_metric", None),
            "best_model_checkpoint": getattr(metrics, "best_model_checkpoint", None),
        },
    )

    print(f"\n✅ Training complete. Saved LoRA adapters + tokenizer to: {args.output_dir}")
    print("Next:")
    print(f"  - Merge adapters (optional): python scripts/merge_lora.py --base_model {args.model_id} --lora_dir {args.output_dir} --out_dir outputs/merged-model")
    print(f"  - CLI inference: python scripts/infer_cli.py --model_dir {args.output_dir} --base_model {args.model_id} --use_adapters")


if __name__ == "__main__":
    main()
