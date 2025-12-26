from __future__ import annotations

import os
from typing import Any, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from finetune.prompts import apply_chat_template, build_messages


app = FastAPI(title="RunPod Instruction Model Server", version="1.0.0")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    system: str = ""
    max_new_tokens: int = 256
    temperature: float = 0.5
    top_p: float = 0.9


class GenerateResponse(BaseModel):
    output: str


_model: Optional[Any] = None
_tokenizer: Optional[Any] = None


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if not v:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_model() -> tuple[Any, Any]:
    """
    Supports two modes:

    1) MERGED model directory:
        MODEL_DIR=/path/to/merged_model

    2) Base + adapters:
        BASE_MODEL=Qwen/Qwen2.5-7B-Instruct
        ADAPTER_DIR=/path/to/lora_adapters
        LOAD_4BIT=true|false  (optional; true is good for smaller GPUs)

    Priority:
      - if MODEL_DIR is set -> merged mode
      - else base+adapters mode
    """
    model_dir = os.environ.get("MODEL_DIR", "").strip()
    base_model = os.environ.get("BASE_MODEL", "").strip()
    adapter_dir = os.environ.get("ADAPTER_DIR", "").strip()
    load_4bit = _env_bool("LOAD_4BIT", default=False)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this server.")

    if model_dir:
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

    if not base_model or not adapter_dir:
        raise RuntimeError(
            "Server is not configured.\n"
            "Set either:\n"
            "  MODEL_DIR=/path/to/merged_model\n"
            "OR:\n"
            "  BASE_MODEL=<hf model id>\n"
            "  ADAPTER_DIR=/path/to/lora_adapters\n"
            "Optionally:\n"
            "  LOAD_4BIT=true\n"
        )

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
        base = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quant, device_map="auto")
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok


@app.on_event("startup")
def startup_event() -> None:
    global _model, _tokenizer
    _model, _tokenizer = _load_model()


@torch.inference_mode()
def _generate(prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    assert _model is not None and _tokenizer is not None
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    out = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.pad_token_id,
    )
    return _tokenizer.decode(out[0], skip_special_tokens=True)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    messages = build_messages(user_text=req.prompt, system_text=req.system.strip() or None)
    prompt = apply_chat_template(_tokenizer, messages, add_generation_prompt=True)

    output = _generate(
        prompt=prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return GenerateResponse(output=output)


@app.post("/summarize", response_model=GenerateResponse)
def summarize(req: SummarizeRequest) -> GenerateResponse:
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    user = (
        "Summarize the following text clearly and with expressive phrasing when appropriate, "
        "while staying faithful to the source.\n\n"
        f"{req.text}"
    )
    messages = build_messages(user_text=user, system_text=req.system.strip() or None)
    prompt = apply_chat_template(_tokenizer, messages, add_generation_prompt=True)

    output = _generate(
        prompt=prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return GenerateResponse(output=output)
