from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.utils import ensure_dir, pick_compute_dtype, write_json


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge LoRA adapters into a base model for inference.")
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--lora_dir", type=str, required=True, help="Directory containing adapter_model.safetensors + adapter_config.json")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for merged model")
    ap.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    args = ap.parse_args()

    lora_dir = Path(args.lora_dir)
    if not lora_dir.exists():
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

    ensure_dir(args.out_dir)

    if args.torch_dtype == "auto":
        dtype = pick_compute_dtype(prefer_bf16=True)
    elif args.torch_dtype == "bf16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading adapters from: {lora_dir}")
    peft_model = PeftModel.from_pretrained(base, str(lora_dir))

    print("Merging adapters...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to: {args.out_dir}")
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    tok.save_pretrained(args.out_dir)

    write_json(
        Path(args.out_dir) / "merge_meta.json",
        {
            "base_model": args.base_model,
            "lora_dir": str(lora_dir),
            "dtype": str(dtype),
        },
    )

    print("✅ Done.")


if __name__ == "__main__":
    main()
