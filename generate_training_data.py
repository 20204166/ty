import json
import os
import random
import re
import sys
from typing import List, Dict, Optional

from datasets import load_dataset


# -----------------------------
# Basic text helpers (NL only)
# -----------------------------


def clean_nl(text: str) -> str:
    """
    Light cleaning for natural language ONLY.
    No weird character stripping – just collapse whitespace.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def truncate_summary_complete(text: str, max_words: int) -> str:
    """
    Truncate a summary to max_words, but try to end on ., ?, or !
    so the model sees complete sentences.
    """
    valid_endings = {".", "?", "!"}
    text = clean_nl(text)
    words = text.split()
    if len(words) <= max_words:
        return text if text and text[-1] in valid_endings else (text + ".").strip()

    truncated = " ".join(words[:max_words])
    if truncated and truncated[-1] in valid_endings:
        return truncated
    last = max(truncated.rfind(p) for p in valid_endings)
    if last != -1:
        return truncated[: last + 1].strip()
    return (truncated + ".").strip()


# -----------------------------------
# Helpers to build instruction tasks
# -----------------------------------


def make_summarization_example(
    document: str,
    summary: str,
    max_src_words: int = 512,
    max_tgt_words: int = 128,
) -> Optional[Dict]:
    doc = clean_nl(document)
    summ = clean_nl(summary)

    if not doc or not summ:
        return None

    doc = truncate_words(doc, max_src_words)
    summ = truncate_summary_complete(summ, max_tgt_words)

    source = (
        "[SUMMARIZATION]\n"
        "Summarise the following text in clear English:\n\n"
        f"{doc}\n\nSummary:"
    )

    return {
        "source": source,
        "target": summ,
        "task": "summarization",
    }


# -----------------------------
# 1) Summarisation datasets
# -----------------------------


def process_cnn_dailymail(
    max_examples: int = 60_000,
    max_src_words: int = 512,
    max_tgt_words: int = 128,
) -> List[Dict]:
    """
    CNN/DailyMail 3.0.0: fields 'article' and 'highlights'.
    """
    try:
        ds = load_dataset("cnn_dailymail", "3.0.0", split="train")
    except Exception as e:
        print(f"⚠️ CNN/DailyMail not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        ex = make_summarization_example(
            document=s.get("article", ""),
            summary=s.get("highlights", ""),
            max_src_words=max_src_words,
            max_tgt_words=max_tgt_words,
        )
        if ex:
            out.append(ex)

    if len(out) > max_examples:
        out = random.sample(out, max_examples)
    return out


def process_reddit_tifu(
    max_examples: int = 40_000,
    config: str = "long",  # "long" uses tldr as summary, "short" uses title as summary
    max_src_words: int = 512,
    max_tgt_words: int = 64,
) -> List[Dict]:
    """
    Reddit TIFU via ctr4si/reddit_tifu:
      - fields: 'documents', 'tldr', 'title'
      - 'long' config: tldr is the summary
      - 'short' config: title is the summary
    NOTE: This currently requires a dataset script (trust_remote_code),
    so on Kaggle / latest datasets it is likely to be skipped.
    """
    try:
        ds = load_dataset(
            "ctr4si/reddit_tifu",
            config,
            split="train",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"⚠️ Reddit TIFU not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        doc = s.get("documents", "")
        if config == "long":
            summ = s.get("tldr", "") or s.get("title", "")
        else:  # short config
            summ = s.get("title", "") or s.get("tldr", "")

        ex = make_summarization_example(
            document=doc,
            summary=summ,
            max_src_words=max_src_words,
            max_tgt_words=max_tgt_words,
        )
        if ex:
            out.append(ex)

    if len(out) > max_examples:
        out = random.sample(out, max_examples)
    return out


def process_billsum(
    max_examples: int = 50_000,
    max_src_words: int = 512,
    max_tgt_words: int = 128,
) -> List[Dict]:
    """
    BillSum: fields 'text' and 'summary'.
    """
    try:
        ds = load_dataset("billsum", split="train")
    except Exception as e:
        print(f"⚠️ BillSum not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        ex = make_summarization_example(
            document=s.get("text", ""),
            summary=s.get("summary", ""),
            max_src_words=max_src_words,
            max_tgt_words=max_tgt_words,
        )
        if ex:
            out.append(ex)

    if len(out) > max_examples:
        out = random.sample(out, max_examples)
    return out


def process_xsum(
    max_examples: int = 50_000,
    max_src_words: int = 512,
    max_tgt_words: int = 64,
) -> List[Dict]:
    """
    XSum: fields 'document' and 'summary'.
    NOTE: On Kaggle with latest `datasets`, this may be skipped because it
    still uses a dataset script.
    """
    try:
        ds = load_dataset("xsum", split="train")
    except Exception as e:
        print(f"⚠️ XSum not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        ex = make_summarization_example(
            document=s.get("document", ""),
            summary=s.get("summary", ""),
            max_src_words=max_src_words,
            max_tgt_words=max_tgt_words,
        )
        if ex:
            out.append(ex)

    if len(out) > max_examples:
        out = random.sample(out, max_examples)
    return out


# -----------------------------
# 2) C++ code generation
# -----------------------------


def process_cpp_vault(
    max_examples: int = 200_000,
    max_desc_words: int = 128,
    max_code_chars: int = 4000,
) -> List[Dict]:
    """
    C++ code generation from Fsoft-AIC/the-vault-function:
      - we filter language="C++"
      - we use docstring as description, code as target
      - streaming=True because the dataset is very large

    NOTE: This dataset currently relies on a Python loading script and may
    be skipped on Kaggle / datasets>=4 with "dataset scripts not supported".
    """
    try:
        data = load_dataset(
            "Fsoft-AIC/the-vault-function",
            split_set=["train"],
            languages=["C++"],
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"⚠️ the-vault-function (C++) not found – skipping. ({e})", file=sys.stderr)
        return []

    stream = data.get("train", None)
    if stream is None:
        stream = next(iter(data.values()))

    out: List[Dict] = []
    for sample in stream:
        docstring = (
            sample.get("docstring")
            or sample.get("short_docstring")
            or sample.get("original_docstring")
            or ""
        )
        code = sample.get("code") or sample.get("original_string") or ""

        if not docstring or not code:
            continue

        desc = truncate_words(clean_nl(docstring), max_desc_words)
        code_str = str(code).strip()
        if len(code_str) > max_code_chars:
            code_str = code_str[:max_code_chars]

        source = (
            "[CODE_CPP]\n"
            "You are an assistant that writes clean, well-formatted C++ code.\n"
            "Write a C++ function that matches this description:\n\n"
            f"{desc}\n\n"
            "C++ code:"
        )

        out.append(
            {
                "source": source,
                "target": code_str,
                "task": "code_cpp",
            }
        )

        if len(out) >= max_examples:
            break

    return out


# -----------------------------
# 3) Math QA
# -----------------------------


def process_simple_math(
    max_examples: int = 100_000,
    max_q_words: int = 128,
    max_a_words: int = 64,
) -> List[Dict]:
    """
    Simple maths from PingVortex/simple-math:
      - fields: 'question', 'answer'
    """
    try:
        ds = load_dataset("PingVortex/simple-math", split="train")
    except Exception as e:
        print(f"⚠️ simple-math not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        q = clean_nl(s.get("question", ""))
        a = clean_nl(s.get("answer", ""))
        if not q or not a:
            continue

        q = truncate_words(q, max_q_words)
        a = truncate_words(a, max_a_words)

        source = (
            "[MATH]\n"
            "Solve the following math problem step by step and give the final answer:\n\n"
            f"{q}\n\nAnswer:"
        )

        out.append(
            {
                "source": source,
                "target": a,
                "task": "math",
            }
        )

        if len(out) >= max_examples:
            break

    return out


def process_gsm8k_fix(
    max_examples: int = 8_000,
    max_q_words: int = 128,
    max_reason_words: int = 256,
) -> List[Dict]:
    """
    Grade-school math word problems from hkust-nlp/gsm8k-fix.
    Fields (parquet): 'query', 'resp' (reasoning), 'ans' (final answer). :contentReference[oaicite:2]{index=2}
    We train the model to produce reasoning + final numeric answer.
    """
    try:
        ds = load_dataset("hkust-nlp/gsm8k-fix", split="train")
    except Exception as e:
        print(f"⚠️ gsm8k-fix not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        q = clean_nl(s.get("query", ""))
        reasoning = clean_nl(s.get("resp", ""))
        final_ans = clean_nl(s.get("ans", "") or s.get("gt_ans", ""))

        if not q or not reasoning or not final_ans:
            continue

        q = truncate_words(q, max_q_words)
        reasoning = truncate_words(reasoning, max_reason_words)

        source = (
            "[MATH]\n"
            "Solve the following grade-school math word problem step by step. "
            "End your solution with a line 'Final answer: <number>'.\n\n"
            f"{q}\n\nSolution:"
        )
        target = f"{reasoning}\n\nFinal answer: {final_ans}"

        out.append(
            {
                "source": source,
                "target": target,
                "task": "math",
            }
        )

        if len(out) >= max_examples:
            break

    return out


def process_math_qa(
    max_examples: int = 30_000,
    max_q_words: int = 128,
    max_a_words: int = 256,
) -> List[Dict]:
    """
    Multiple-choice math problems from regisss/math_qa. :contentReference[oaicite:3]{index=3}
    Fields: 'Problem', 'Rationale', 'options', 'correct'.
    We use the rationale as the answer text (it usually already includes 'answer: X').
    """
    try:
        ds = load_dataset("regisss/math_qa", split="train")
    except Exception as e:
        print(f"⚠️ math_qa not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        problem = clean_nl(s.get("Problem", ""))
        rationale = clean_nl(s.get("Rationale", ""))
        options = clean_nl(s.get("options", ""))

        if not problem or not rationale:
            continue

        problem = truncate_words(problem, max_q_words)
        rationale = truncate_words(rationale, max_a_words)

        source = (
            "[MATH]\n"
            "You are a helpful math tutor. Solve the problem step by step and choose the correct option.\n\n"
            f"Problem: {problem}\n\nOptions: {options}\n\nAnswer:"
        )

        out.append(
            {
                "source": source,
                "target": rationale,
                "task": "math",
            }
        )

        if len(out) >= max_examples:
            break

    return out


# -----------------------------
# 4) Optional custom JSONL
# -----------------------------


def load_custom_jsonl(path: str, max_examples: int = 100_000) -> List[Dict]:
    """
    Load custom data from a JSONL file with lines like:
      {"source": "...", "target": "...", "task": "custom"}
    If 'task' is missing, it defaults to 'custom'.
    """
    if not path or not os.path.exists(path):
        return []

    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = obj.get("source")
            tgt = obj.get("target")
            if not src or not tgt:
                continue
            task = obj.get("task", "custom")
            out.append(
                {
                    "source": str(src),
                    "target": str(tgt),
                    "task": str(task),
                }
            )
            if len(out) >= max_examples:
                break
    return out


# -----------------------------
# 5) Orchestration / saving
# -----------------------------


def save_combined_data(
    output_file: str,
    max_per_summarization: int = 100_000,
    max_cpp: int = 200_000,
    max_math: int = 100_000,
    custom_jsonl: Optional[str] = None,
):
    random.seed(42)
    parts: List[Dict] = []

    # ---- Summarisation sources ----
    try:
        cnn_data = process_cnn_dailymail(max_examples=max_per_summarization)
        parts.extend(cnn_data)
        print(f"Processed CNN/DailyMail: {len(cnn_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing CNN/DailyMail: {e}", file=sys.stderr)

    try:
        xsum_data = process_xsum(max_examples=max_per_summarization)
        parts.extend(xsum_data)
        print(f"Processed XSum: {len(xsum_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing XSum: {e}", file=sys.stderr)

    try:
        reddit_data = process_reddit_tifu(max_examples=max_per_summarization)
        parts.extend(reddit_data)
        print(f"Processed Reddit TIFU: {len(reddit_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing Reddit TIFU: {e}", file=sys.stderr)

    try:
        billsum_data = process_billsum(max_examples=max_per_summarization)
        parts.extend(billsum_data)
        print(f"Processed BillSum: {len(billsum_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing BillSum: {e}", file=sys.stderr)

    # ---- C++ code ----
    try:
        cpp_data = process_cpp_vault(max_examples=max_cpp)
        parts.extend(cpp_data)
        print(f"Processed C++ code (the-vault-function): {len(cpp_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing C++ dataset: {e}", file=sys.stderr)

    # ---- Math (multiple sources) ----
    try:
        simple_math_data = process_simple_math(max_examples=max_math)
        parts.extend(simple_math_data)
        print(f"Processed simple-math: {len(simple_math_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing simple-math: {e}", file=sys.stderr)

    try:
        gsm8k_data = process_gsm8k_fix(max_examples=min(8_000, max_math))
        parts.extend(gsm8k_data)
        print(f"Processed gsm8k-fix: {len(gsm8k_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing gsm8k-fix: {e}", file=sys.stderr)

    try:
        math_qa_data = process_math_qa(max_examples=min(30_000, max_math))
        parts.extend(math_qa_data)
        print(f"Processed math_qa: {len(math_qa_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing math_qa: {e}", file=sys.stderr)

    # ---- Custom ----
    if custom_jsonl:
        custom_data = load_custom_jsonl(custom_jsonl)
        parts.extend(custom_data)
        print(f"Loaded custom JSONL: {len(custom_data)} examples")

    if not parts:
        raise RuntimeError("No data processed—check your datasets.")

    random.shuffle(parts)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(parts, f, ensure_ascii=False, indent=2)
    print(f"\nCombined training data written to {output_file} with {len(parts)} examples")


if __name__ == "__main__":
    """
    Usage:
      python generate_training_data.py [output_path] [custom_jsonl]

    Examples:
      python generate_training_data.py app/models/data/text/training_data.json
      python generate_training_data.py app/models/data/text/training_data.json custom.jsonl
    """
    out_path = sys.argv[1] if len(sys.argv) > 1 else "app/models/data/text/training_data.json"
    custom_path = sys.argv[2] if len(sys.argv) > 2 else None

    save_combined_data(
        output_file=out_path,
        max_per_summarization=100_000,  # tune for GPU budget
        max_cpp=200_000,
        max_math=100_000,
        custom_jsonl=custom_path,
    )
