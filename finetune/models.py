from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from finetune.utils import pick_compute_dtype


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    tokenizer: any


def default_lora_target_modules(model_id: str) -> List[str]:
    mid = model_id.lower()
    # Qwen + Llama families generally use these projection names.
    # We will filter them to only those present in the model at runtime.
    if "qwen" in mid or "llama" in mid:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def filter_existing_target_modules(model: torch.nn.Module, candidates: List[str]) -> List[str]:
    names = set()
    for n, _m in model.named_modules():
        names.add(n.split(".")[-1])
    filtered = [c for c in candidates if c in names]
    # If nothing matches (rare), fall back to the original list to surface a clear PEFT error later.
    return filtered if filtered else candidates


def load_tokenizer(model_id: str) -> any:
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Ensure pad token exists for batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_qlora_model(
    model_id: str,
    compute_dtype: Optional[torch.dtype] = None,
    use_flash_attn_2: bool = False,
) -> torch.nn.Module:
    if compute_dtype is None:
        compute_dtype = pick_compute_dtype(prefer_bf16=True)

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    attn_impl = "flash_attention_2" if use_flash_attn_2 else None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=compute_dtype,
        attn_implementation=attn_impl,
    )
    model.config.use_cache = False
    return model


def make_lora_config(
    model: torch.nn.Module,
    model_id: str,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Optional[List[str]] = None,
) -> LoraConfig:
    if target_modules is None:
        target_modules = filter_existing_target_modules(model, default_lora_target_modules(model_id))

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
    )


def prepare_for_qlora(model: torch.nn.Module) -> torch.nn.Module:
    # Enables gradients for input embeddings where needed + stabilizes LayerNorms in k-bit training.
    return prepare_model_for_kbit_training(model)


def build_model_bundle(
    model_id: str,
    use_flash_attn_2: bool,
) -> ModelBundle:
    tokenizer = load_tokenizer(model_id)
    model = load_qlora_model(model_id=model_id, use_flash_attn_2=use_flash_attn_2)
    model = prepare_for_qlora(model)
    return ModelBundle(model=model, tokenizer=tokenizer)
