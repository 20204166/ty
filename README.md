# RunPod QLoRA Mixed-Task Instruction Fine-tuning (Qwen2.5 / Llama3.1)

This repo trains/fine-tunes an open-source instruction model on a single GPU RunPod instance using your existing dataset generator output:

- Input JSON: `app/models/data/text/training_data.json` (default)
- Each item must be a dict with keys:
  - `source` (instruction prompt)
  - `target` (desired output)
  - `task` (label like summarization|code_cpp|math, etc.)

Training uses:
- TRL `SFTTrainer` + `SFTConfig`
- 4-bit quantization (bitsandbytes) + LoRA adapters (PEFT)
- Mixed-task training is supported by default (no task filtering unless you pass `--only_tasks`)

---

## 0) Recommended RunPod hardware

A single GPU instance (A100 40/80GB, L40S, etc.) is ideal for 7B-8B QLoRA.
Smaller GPUs can work with smaller batch and seq length.

---

## 1) Install deps (no Docker)

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
