from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.prompts import apply_chat_template, build_messages


def load_model_with_optional_adapters(
    *,
    model_dir: str,
    base_model: Optional[str],
    use_adapters: bool,
    load_4bit: bool,
) -> tuple[Any, Any]:
    if use_adapters:
        if not base_model:
            raise ValueError("--base_model is required when --use_adapters is set.")

        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if load_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=quant,
                device_map="auto",
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                device_map="auto",
            )

        model = PeftModel.from_pretrained(base, model_dir)
        model.eval()
        return model, tok

    # merged model dir
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tok


@torch.inference_mode()
def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="CLI inference for merged model or base+LoRA adapters.")
    ap.add_argument("--model_dir", type=str, required=True, help="Merged model dir OR LoRA adapter dir")
    ap.add_argument("--base_model", type=str, default="", help="Required if using adapters")
    ap.add_argument("--use_adapters", action="store_true", help="Load base_model + adapters from model_dir")
    ap.add_argument("--load_4bit", action="store_true", help="Only applies when --use_adapters is set")

    ap.add_argument("--mode", type=str, default="summarize", choices=["summarize", "generate"])
    ap.add_argument("--text", type=str, default="")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)

    ap.add_argument("--system", type=str, default="")

    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this CLI inference script.")

    model, tok = load_model_with_optional_adapters(
        model_dir=args.model_dir,
        base_model=args.base_model.strip() or None,
        use_adapters=args.use_adapters,
        load_4bit=args.load_4bit,
    )

    if args.mode == "summarize":
        if not args.text.strip():
            raise ValueError("--text is required for summarize mode.")
        user = (
            "Summarize the following text clearly and with expressive phrasing when appropriate, "
            "while staying faithful to the source.\n\n"
            f"{args.text}"
        )
        messages = build_messages(user_text=user, system_text=args.system.strip() or None)
        prompt = apply_chat_template(tok, messages, add_generation_prompt=True)
        full = generate_text(
            model=model,
            tokenizer=tok,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(full)
        return

    if args.mode == "generate":
        if not args.prompt.strip():
            raise ValueError("--prompt is required for generate mode.")
        messages = build_messages(user_text=args.prompt, system_text=args.system.strip() or None)
        prompt = apply_chat_template(tok, messages, add_generation_prompt=True)
        full = generate_text(
            model=model,
            tokenizer=tok,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(full)
        return


if __name__ == "__main__":
    main()
