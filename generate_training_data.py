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

def process_natural_questions_clean(
    max_examples: int = 50_000,
    max_ctx_words: int = 512,
    max_q_words: int = 64,
    max_ans_words: int = 64,
) -> List[Dict]:
    """
    Natural Questions (clean version).
    Works with variants where the fields are like:
      - document/context (passage)
      - question/query
      - answer (string, list or dict)
    """
    try:
        # your original ID – this is what succeeded, just our field mapping was wrong
        ds = load_dataset("lighteval/natural_questions_clean", split="train")
    except Exception:
        # fall back to another common variant if the above ever fails
        try:
            ds = load_dataset("rojagtap/natural_questions_clean", split="train")
        except Exception as e:
            print(f"⚠️ Natural Questions (clean) not found – skipping. ({e})", file=sys.stderr)
            return []

    out: List[Dict] = []

    for s in ds:
        # context / passage
        context = (
            s.get("context")
            or s.get("document")
            or s.get("passage")
            or s.get("ctx")
            or ""
        )

        # question
        question = s.get("question") or s.get("query") or ""

        # answer can be str / list / dict depending on version
        answer = s.get("answer", "")

        if isinstance(answer, dict):
            answer = (
                answer.get("text")
                or answer.get("short_answers")
                or answer.get("long_answer")
                or ""
            )
        elif isinstance(answer, list):
            if not answer:
                answer = ""
            else:
                a0 = answer[0]
                if isinstance(a0, dict):
                    answer = (
                        a0.get("text")
                        or a0.get("short_answers")
                        or a0.get("long_answer")
                        or ""
                    )
                else:
                    answer = a0

        context = clean_nl(context)
        question = clean_nl(question)
        answer = clean_nl(answer)

        if not context or not question or not answer:
            continue

        context = truncate_words(context, max_ctx_words)
        question = truncate_words(question, max_q_words)
        answer = truncate_summary_complete(answer, max_ans_words)

        source = (
            "[READING_COMPREHENSION]\n"
            "Read the passage and answer the question in a short, direct sentence.\n\n"
            f"Passage: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        out.append(
            {
                "source": source,
                "target": answer,
                "task": "summarization",
            }
        )

        if len(out) >= max_examples:
            break

    return out

def process_hotpot_qa(
    max_examples: int = 50_000,
    max_ctx_words: int = 512,
    max_q_words: int = 64,
    max_ans_words: int = 64,
) -> List[Dict]:
    """
    HotpotQA multi-hop QA:
      dataset: 'hotpotqa/hotpot_qa', config 'distractor'
      key fields: 'question', 'answer', 'context'

    'context' can be:
      - list of [title, [sent1, sent2, ...]] (older format), OR
      - list of {"title": ..., "sentences": [...]} (newer format)
    """
    try:
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    except Exception as e:
        print(f"⚠️ HotpotQA not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        question = clean_nl(s.get("question", ""))
        answer = clean_nl(s.get("answer", ""))

        ctx_list = s.get("context", []) or s.get("documents", []) or []
        passages = []

        for item in ctx_list:
            title = ""
            sentences = None

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # old-style [title, [sent1, sent2, ...]]
                title, sentences = item[0], item[1]
            elif isinstance(item, dict):
                # new-style {"title": ..., "sentences": [...]}
                title = item.get("title", "")
                sentences = (
                    item.get("sentences")
                    or item.get("sentence")
                    or item.get("text")
                )
            else:
                # bare string or something weird – treat as text only
                sentences = item

            title = clean_nl(title)

            if isinstance(sentences, (list, tuple)):
                text = clean_nl(" ".join(str(s) for s in sentences))
            else:
                text = clean_nl(sentences)

            if not text:
                continue

            if title:
                passages.append(f"{title}: {text}")
            else:
                passages.append(text)

        if not question or not answer or not passages:
            continue

        context = truncate_words(" ".join(passages), max_ctx_words)
        question = truncate_words(question, max_q_words)
        answer = truncate_summary_complete(answer, max_ans_words)

        source = (
            "[MULTIHOP_QA]\n"
            "Read the documents and answer the question using information from the text. "
            "Give a short, direct answer (one or two sentences).\n\n"
            f"Documents:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        out.append(
            {
                "source": source,
                "target": answer,
                "task": "summarization",
            }
        )

        if len(out) >= max_examples:
            break

    return out

def process_squad(
    max_examples: int = 50_000,
    max_ctx_words: int = 400,
    max_q_words: int = 64,
    max_ans_words: int = 64,
) -> List[Dict]:
    """
    SQuAD-style reading comprehension:
      dataset: 'squad'
      fields: 'context', 'question', 'answers'['text']

    We turn it into an instruction:
      - input: passage + question
      - target: short answer text (1-2 sentences)
    This is not "summarisation" strictly, but very similar behaviour.
    """
    try:
        ds = load_dataset("squad", split="train")
    except Exception as e:
        print(f"⚠️ SQuAD not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        context = clean_nl(s.get("context", ""))
        question = clean_nl(s.get("question", ""))
        answers = s.get("answers", {}) or {}
        texts = answers.get("text", []) or []

        if not context or not question or not texts:
            continue

        # use the first annotated answer
        answer = clean_nl(texts[0])
        if not answer:
            continue

        context = truncate_words(context, max_ctx_words)
        question = truncate_words(question, max_q_words)
        answer = truncate_summary_complete(answer, max_ans_words)

        source = (
            "[READING_COMPREHENSION]\n"
            "Read the passage and answer the question in one or two sentences.\n\n"
            f"Passage: {context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        out.append(
            {
                "source": source,
                "target": answer,
                # 📌 treat as 'summarization-like' for task sampling
                "task": "summarization",
            }
        )

        if len(out) >= max_examples:
            break

    return out


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
def process_stackoverflow_linux(
    max_examples: int = 30_000,
    max_q_words: int = 256,
    max_a_words: int = 256,
) -> List[Dict]:
    """
    Linux / shell Q&A from KonradSzafer/stackoverflow_linux.
    Fields: 'title', 'question', 'answer', 'url'.
    We turn it into instruction-style shell tasks.
    """
    try:
        ds = load_dataset("KonradSzafer/stackoverflow_linux", split="train")
    except Exception as e:
        print(f"⚠️ stackoverflow_linux not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        question = clean_nl(s.get("question", "")) or clean_nl(s.get("title", ""))
        answer = clean_nl(s.get("answer", ""))

        if not question or not answer:
            continue

        question = truncate_words(question, max_q_words)
        answer = truncate_words(answer, max_a_words)

        source = (
            "[SHELL]\n"
            "You are a Linux terminal expert. Answer the user's question with "
            "the exact shell commands and a short explanation:\n\n"
            f"{question}\n\nAnswer:"
        )

        out.append(
            {
                "source": source,
                "target": answer,
                # keep it in the code bucket so TASK_RATIOS still work
                "task": "code_cpp",
            }
        )

        if len(out) >= max_examples:
            break

    return out

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
    
def process_codeparrot_general(
    max_examples: int = 50_000,
    max_code_chars: int = 4000,
) -> List[Dict]:
    """
    General code completion from codeparrot/codeparrot-clean-valid.
    Fields: 'content' contains the code snippet.
    We create a prefix->full-completion style task.

    This still uses task='code_cpp' so your existing TASK_RATIOS
    do not need to change.
    """
    try:
        ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train")
    except Exception as e:
        print(f"⚠️ codeparrot/codeparrot-clean-valid not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        code = s.get("content", None)
        if not code:
            continue

        code_str = str(code).strip()
        if not code_str:
            continue

        # Full snippet target
        if len(code_str) > max_code_chars:
            code_str = code_str[:max_code_chars]

        # Use the first part as a "prefix" context for completion
        prefix_chars = max_code_chars // 4
        prefix = code_str[:prefix_chars]

        source = (
            "[CODE_GENERAL]\n"
            "You are an assistant that completes and improves code.\n"
            "Continue and complete the following code snippet in the same language:\n\n"
            f"{prefix}\n\n"
            "Completed code:"
        )

        out.append(
            {
                "source": source,
                "target": code_str,
                "task": "code_cpp",  # keep same task label so sampling logic still works
            }
        )

        if len(out) >= max_examples:
            break

    return out

def process_stack_smol_python(
    max_examples: int = 30_000,
    max_code_chars: int = 4000,
) -> List[Dict]:
    """
    Python code from ml6team/the-stack-smol-python.
    Fields include 'content' (code), 'lang', etc.
    We again do prefix->completion style tasks.

    Still tagged as 'code_cpp' to avoid changing training code.
    """
    try:
        ds = load_dataset("ml6team/the-stack-smol-python", split="train")
    except Exception as e:
        print(f"⚠️ the-stack-smol-python not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        code = s.get("content", None)
        if not code:
            continue

        code_str = str(code).strip()
        if not code_str:
            continue

        if len(code_str) > max_code_chars:
            code_str = code_str[:max_code_chars]

        prefix_chars = max_code_chars // 4
        prefix = code_str[:prefix_chars]

        source = (
            "[CODE_PYTHON]\n"
            "You are an assistant that writes clean, idiomatic Python.\n"
            "Continue and complete the following Python code:\n\n"
            f"{prefix}\n\n"
            "Completed Python code:"
        )

        out.append(
            {
                "source": source,
                "target": code_str,
                "task": "code_cpp",  # same bucket as C++ for now
            }
        )

        if len(out) >= max_examples:
            break

    return out

def process_linux_commands_dataset(
    max_examples: int = 100_000,
    max_input_words: int = 64,
    max_cmd_chars: int = 256,
):
    """
    Shell / Linux one-liners from hrsvrn/linux-commands-dataset.
    Columns: 'input' (NL description), 'output' (command).
    """
    try:
        ds = load_dataset("hrsvrn/linux-commands-dataset", split="train")
    except Exception as e:
        print(f"⚠️ linux-commands-dataset not found – skipping. ({e})", file=sys.stderr)
        return []

    out = []
    for row in ds:
        nl = clean_nl(row.get("input", ""))
        cmd = row.get("output", "")

        if not nl or not cmd:
            continue

        nl = truncate_words(nl, max_input_words)
        cmd = str(cmd).strip()
        if not cmd:
            continue
        if len(cmd) > max_cmd_chars:
            cmd = cmd[:max_cmd_chars]

        source = (
            "[SHELL]\n"
            "Write a single Linux shell command that correctly performs this task:\n\n"
            f"{nl}\n\nCommand:"
        )

        out.append(
            {
                "source": source,
                "target": cmd,
                # treat shell as 'code' so it falls into your existing TASK_RATIOS
                "task": "code_cpp",
            }
        )

        if len(out) >= max_examples:
            break

    return out

def process_leetcode_dataset(
    max_examples: int = 5_000,
    max_problem_words: int = 256,
    max_answer_words: int = 512,
) -> List[Dict]:
    """
    LeetCode-style coding problems from newfacade/LeetCodeDataset.
    Fields (see dataset card):
      - 'query' / 'problem_description': prompt/problem
      - 'response' / 'completion': solution or reasoning + code

    We turn them into instruction-style code tasks.
    Still use task='code_cpp' so you don't have to change TASK_RATIOS.
    """
    try:
        ds = load_dataset("newfacade/LeetCodeDataset", split="train")
    except Exception as e:
        print(f"⚠️ LeetCodeDataset not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        # Prefer the ready-made LLM-style fields if they exist
        problem = s.get("query") or s.get("problem_description") or ""
        answer = s.get("response") or s.get("completion") or ""

        problem = clean_nl(problem)
        answer = clean_nl(answer)

        if not problem or not answer:
            continue

        problem = truncate_words(problem, max_problem_words)
        answer = truncate_words(answer, max_answer_words)

        source = (
            "[LEETCODE]\n"
            "You are an expert competitive programming assistant.\n"
            "Solve the following coding problem. Provide a clear explanation "
            "and a complete, correct solution in code:\n\n"
            f"{problem}\n\n"
            "Solution:"
        )

        out.append(
            {
                "source": source,
                "target": answer,
                "task": "code_cpp",  # keep it in the 'code' bucket
            }
        )

        if len(out) >= max_examples:
            break

    return out

# -----------------------------
# 3) Math QA
# -----------------------------

def process_stackoverflow_linux(
    max_examples: int = 30_000,
    max_q_words: int = 256,
    max_a_words: int = 256,
) -> List[Dict]:
    """
    Linux / shell Q&A from KonradSzafer/stackoverflow_linux.
    Fields: 'title', 'question', 'answer', 'url'.
    We turn it into instruction-style shell tasks.
    """
    try:
        ds = load_dataset("KonradSzafer/stackoverflow_linux", split="train")
    except Exception as e:
        print(f"⚠️ stackoverflow_linux not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        question = clean_nl(s.get("question", "")) or clean_nl(s.get("title", ""))
        answer = clean_nl(s.get("answer", ""))

        if not question or not answer:
            continue

        question = truncate_words(question, max_q_words)
        answer = truncate_words(answer, max_a_words)

        source = (
            "[SHELL]\n"
            "You are a Linux terminal expert. Answer the user's question with "
            "the exact shell commands and a short explanation:\n\n"
            f"{question}\n\nAnswer:"
        )

        out.append(
            {
                "source": source,
                "target": answer,
                # keep it in the code bucket so TASK_RATIOS still work
                "task": "code_cpp",
            }
        )

        if len(out) >= max_examples:
            break

    return out
    
def process_hendrycks_math(
    max_examples: int = 20_000,
    max_q_words: int = 256,
    max_a_words: int = 512,
) -> List[Dict]:
    try:
        ds = load_dataset("HuggingFaceTB/MATH", "all", split="train")
    except Exception as e:
        print(f"⚠️ hendrycks_math/MATH not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        problem = clean_nl(s.get("problem", ""))
        solution = clean_nl(s.get("solution", ""))

        if not problem or not solution:
            continue

        problem = truncate_words(problem, max_q_words)
        solution = truncate_words(solution, max_a_words)

        source = (
            "[MATH]\n"
            "You are an expert university-level math tutor.\n"
            "Solve the following problem step by step and give a final answer at the end:\n\n"
            f"{problem}\n\nSolution:"
        )

        out.append(
            {
                "source": source,
                "target": solution,
                "task": "math",
            }
        )

        if len(out) >= max_examples:
            break

    return out

    
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
def process_arc_ai2(
    max_examples: int = 20_000,
    max_q_words: int = 128,
) -> List[Dict]:
    """
    Multi-choice reasoning from ai2_arc (ARC-Easy + ARC-Challenge).
    Fields: 'question', 'choices'['text','label'], 'answerKey'.
    We train the model to output just the correct letter.
    """
    try:
        ds_easy = load_dataset("ai2_arc", "ARC-Easy", split="train")
        ds_challenge = load_dataset("ai2_arc", "ARC-Challenge", split="train")
    except Exception as e:
        print(f"⚠️ ai2_arc not found – skipping. ({e})", file=sys.stderr)
        return []

    combined = list(ds_easy) + list(ds_challenge)
    random.shuffle(combined)

    out: List[Dict] = []
    for s in combined:
        q = clean_nl(s.get("question", ""))
        choices = s.get("choices", {}) or {}
        texts = choices.get("text", []) or []
        labels = choices.get("label", []) or []
        answer_key = (s.get("answerKey", "") or "").strip()

        if not q or not texts or not labels or not answer_key:
            continue

        q = truncate_words(q, max_q_words)

        option_lines = []
        for lab, txt in zip(labels, texts):
            option_lines.append(f"{lab}. {clean_nl(txt)}")
        options_str = "\n".join(option_lines)

        source = (
            "[MATH]\n"
            "You are a reasoning assistant. Read the question and choose the correct option.\n\n"
            f"Question: {q}\n\n"
            f"Options:\n{options_str}\n\n"
            "Answer with the letter of the correct option.\n\n"
            "Answer:"
        )

        target = answer_key  # e.g. "A"
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


def process_openbookqa(
    max_examples: int = 10_000,
    max_q_words: int = 128,
) -> List[Dict]:
    """
    OpenBookQA main split.
    Fields: 'question_stem', 'choices'['text','label'], 'answerKey'.
    Again, we train to output just the correct letter.
    """
    try:
        ds = load_dataset("openbookqa", "main", split="train")
    except Exception as e:
        print(f"⚠️ openbookqa not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        stem = clean_nl(s.get("question_stem", ""))
        choices = s.get("choices", {}) or {}
        texts = choices.get("text", []) or []
        labels = choices.get("label", []) or []
        answer_key = (s.get("answerKey", "") or "").strip()

        if not stem or not texts or not labels or not answer_key:
            continue

        stem = truncate_words(stem, max_q_words)

        option_lines = []
        for lab, txt in zip(labels, texts):
            option_lines.append(f"{lab}. {clean_nl(txt)}")
        options_str = "\n".join(option_lines)

        source = (
            "[MATH]\n"
            "You are a reasoning assistant. Answer the question using common-sense and science.\n\n"
            f"Question: {stem}\n\n"
            f"Options:\n{options_str}\n\n"
            "Answer with the letter of the correct option.\n\n"
            "Answer:"
        )
        target = answer_key

        out.append(
            {
                "source": source,
                "target": target,
                "task": "math",   # stays in the math/reasoning bucket
            }
        )

        if len(out) >= max_examples:
            break

    return out


def process_svamp(
    max_examples: int = 15_000,
    max_q_words: int = 128,
) -> List[Dict]:
    """
    SVAMP: math word problems.
    Typical fields: 'Body', 'Question', 'Answer'.
    We join Body + Question and train to output the numeric answer.
    """
    try:
        ds = load_dataset("ChilleD/SVAMP", split="train")
    except Exception:
        # fallback to default id if alias changes
        try:
            ds = load_dataset("svamp", split="train")
        except Exception as e:
            print(f"⚠️ svamp not found – skipping. ({e})", file=sys.stderr)
            return []

    out: List[Dict] = []
    for s in ds:
        body = clean_nl(s.get("Body", "") or s.get("body", ""))
        question = clean_nl(s.get("Question", "") or s.get("question", ""))
        ans = clean_nl(str(s.get("Answer", "") or s.get("answer", "")))

        if not question or not ans:
            continue

        full_q = (body + " " + question).strip()
        full_q = truncate_words(full_q, max_q_words)

        source = (
            "[MATH]\n"
            "Solve the following arithmetic word problem step by step. "
            "End with 'Final answer: <number>'.\n\n"
            f"{full_q}\n\nSolution:"
        )
        target = f"Final answer: {ans}"

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


def process_boolq(
    max_examples: int = 20_000,
    max_passage_words: int = 160,
    max_q_words: int = 64,
) -> List[Dict]:
    """
    BoolQ: reading comprehension yes/no questions.
    Fields: 'passage', 'question', 'answer' (bool).
    We train the model to output 'yes' or 'no'.
    """
    try:
        ds = load_dataset("boolq", split="train")
    except Exception as e:
        print(f"⚠️ boolq not found – skipping. ({e})", file=sys.stderr)
        return []

    out: List[Dict] = []
    for s in ds:
        passage = clean_nl(s.get("passage", ""))
        question = clean_nl(s.get("question", ""))
        ans_bool = s.get("answer", None)

        if not passage or not question or ans_bool is None:
            continue

        passage = truncate_words(passage, max_passage_words)
        question = truncate_words(question, max_q_words)

        # bool -> "yes"/"no"
        ans_str = "yes" if ans_bool else "no"

        source = (
            "[MATH]\n"
            "Read the passage and answer the question with 'yes' or 'no'. "
            "You may reason briefly, but finish with a single word answer.\n\n"
            f"Passage: {passage}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        target = ans_str

        out.append(
            {
                "source": source,
                "target": target,
                "task": "math",  # we still treat this as reasoning/maths bucket
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
    
    # NEW: SQuAD-style RC
    try:
        squad_data = process_squad(max_examples=max_per_summarization)
        parts.extend(squad_data)
        print(f"Processed SQuAD: {len(squad_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing SQuAD: {e}", file=sys.stderr)

    # NEW: Natural Questions (clean, long-passages)
    try:
        nq_data = process_natural_questions_clean(max_examples=max_per_summarization)
        parts.extend(nq_data)
        print(f"Processed Natural Questions (clean): {len(nq_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing Natural Questions (clean): {e}", file=sys.stderr)

    # NEW: HotpotQA multi-hop
    try:
        hotpot_data = process_hotpot_qa(max_examples=max_per_summarization)
        parts.extend(hotpot_data)
        print(f"Processed HotpotQA: {len(hotpot_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing HotpotQA: {e}", file=sys.stderr)

    # ---- C++ code ----
    try:
        cpp_data = process_cpp_vault(max_examples=max_cpp)
        parts.extend(cpp_data)
        print(f"Processed C++ code (the-vault-function): {len(cpp_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing C++ dataset: {e}", file=sys.stderr)
    
    # ---- LeetCode-style problems ----
    try:
        leet_data = process_leetcode_dataset(max_examples=3_000)
        parts.extend(leet_data)
        print(f"Processed LeetCodeDataset: {len(leet_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing LeetCodeDataset: {e}", file=sys.stderr)


    # ---- Extra general code: codeparrot ----
    try:
        codeparrot_data = process_codeparrot_general(max_examples=50_000)
        parts.extend(codeparrot_data)
        print(f"Processed general code (codeparrot-clean-valid): {len(codeparrot_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing codeparrot dataset: {e}", file=sys.stderr)

    # ---- Shell / Linux Q&A ----
    try:
        so_linux_data = process_stackoverflow_linux(max_examples=30_000)
        parts.extend(so_linux_data)
        print(f"Processed StackOverflow Linux: {len(so_linux_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing StackOverflow Linux: {e}", file=sys.stderr)

    # ---- Simple command mapping ----
    try:
        linux_data = process_linux_commands_dataset(max_examples=100_000)
        parts.extend(linux_data)
        print(f"Processed linux-commands-dataset: {len(linux_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing linux-commands-dataset: {e}", file=sys.stderr)


    # ---- Extra Python code: the-stack-smol-python ----
    try:
        py_smol_data = process_stack_smol_python(max_examples=30_000)
        parts.extend(py_smol_data)
        print(f"Processed Python code (the-stack-smol-python): {len(py_smol_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing the-stack-smol-python: {e}", file=sys.stderr)

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

    # ---- Advanced math (Hendrycks) ----
        # ---- Advanced math (Hendrycks) ----
    try:
        hendrycks_data = process_hendrycks_math(max_examples=min(20_000, max_math))
        parts.extend(hendrycks_data)
        print(f"Processed hendrycks_math: {len(hendrycks_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing hendrycks_math: {e}", file=sys.stderr)

    # ---- ARC-style reasoning (ai2_arc) ----
    try:
        arc_data = process_arc_ai2(max_examples=min(20_000, max_math))
        parts.extend(arc_data)
        print(f"Processed ai2_arc: {len(arc_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing ai2_arc: {e}", file=sys.stderr)

    # ---- OpenBookQA reasoning ----
    try:
        obqa_data = process_openbookqa(max_examples=min(10_000, max_math))
        parts.extend(obqa_data)
        print(f"Processed openbookqa: {len(obqa_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing openbookqa: {e}", file=sys.stderr)

    # ---- SVAMP arithmetic word problems ----
    try:
        svamp_data = process_svamp(max_examples=min(15_000, max_math))
        parts.extend(svamp_data)
        print(f"Processed svamp: {len(svamp_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing svamp: {e}", file=sys.stderr)

    # ---- BoolQ yes/no reasoning ----
    try:
        boolq_data = process_boolq(max_examples=min(20_000, max_math))
        parts.extend(boolq_data)
        print(f"Processed boolq: {len(boolq_data)} examples")
    except Exception as e:
        print(f"⚠️ Error processing boolq: {e}", file=sys.stderr)

    


    # ---- Custom ----
    
    if custom_jsonl:
        custom_data = load_custom_jsonl(custom_jsonl)
        parts.extend(custom_data)
        print(f"Loaded custom JSONL: {len(custom_data)} examples")

    if not parts:
        raise RuntimeError("No data processed—check your datasets.")

    #  Final safety pass: drop any examples with empty / whitespace-only source/target
    cleaned_parts = []
    for ex in parts:
        src = ex.get("source", "")
        tgt = ex.get("target", "")
        if src is None or tgt is None:
            continue
        if not str(src).strip() or not str(tgt).strip():
            continue
        cleaned_parts.append(ex)

    if not cleaned_parts:
        raise RuntimeError("All examples filtered out as empty source/target – check data generation.")

    parts = cleaned_parts

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
        max_per_summarization=170_000,  # tune for GPU budget
        max_cpp=170_000,
        max_math=220_000,
        custom_jsonl=custom_path,
    )
