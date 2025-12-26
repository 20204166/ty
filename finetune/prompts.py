from __future__ import annotations

from typing import Any, Dict, List


def build_messages(user_text: str, system_text: str | None = None) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system_text and system_text.strip():
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": user_text})
    return msgs


def apply_chat_template(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> str:
    """
    Uses the model's native chat template when available.
    Falls back to a simple <|user|>/<|assistant|> style if template is absent.
    """
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Fallback (generic)
    if len(messages) == 2 and messages[0]["role"] == "system":
        system = messages[0]["content"]
        user = messages[1]["content"]
        return f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n"
    user = messages[-1]["content"]
    return f"<|user|>\n{user}\n<|assistant|>\n"
