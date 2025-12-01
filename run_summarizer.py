import json
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Optional: if you import pure keras you can also enable this globally
try:
    import keras
    keras.config.enable_unsafe_deserialization()
except Exception:
    # Not critical, load_model(safe_mode=False) below is usually enough
    pass

# ===== MUST MATCH TRAINING =====
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 128
MODEL_PATH = "app/models/saved_model/summarization_model.keras"
TOK_IN_PATH = "app/models/saved_model/tokenizer_input.json"
TOK_TGT_PATH = "app/models/saved_model/tokenizer_target.json"


# -------------------------------------------------
# 1) Load tokenizers
# -------------------------------------------------
with open(TOK_IN_PATH, "r", encoding="utf-8") as f:
    tok_in = tokenizer_from_json(f.read())

with open(TOK_TGT_PATH, "r", encoding="utf-8") as f:
    tok_tgt = tokenizer_from_json(f.read())

oov_id = tok_tgt.word_index.get(tok_tgt.oov_token)

# Try original "<start>", then "start", then fall back to OOV
start_id = tok_tgt.word_index.get("<start>")
if start_id is None:
    start_id = tok_tgt.word_index.get("start", oov_id)

# Try original "<end>", then "end"; -1 means "no special end"
end_id = tok_tgt.word_index.get("<end>")
if end_id is None:
    end_id = tok_tgt.word_index.get("end", -1)

print("oov_id:", oov_id, "start_id:", start_id, "end_id:", end_id)


# -------------------------------------------------
# 2) Load model (unsafe deserialization allowed)
# -------------------------------------------------
print("Loading model from:", MODEL_PATH)
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,      # we don't train here
    safe_mode=False,    # allow Lambda layers with python lambdas
)
print("Model loaded.")


# -------------------------------------------------
# 3) Utilities to build / encode inputs
# -------------------------------------------------
def build_source_summarization(text: str) -> str:
    """
    Recreate the *exact* format used in generate_training_data.py for summarization:
      source = "[SUMMARIZATION]\\nSummarise ... {doc} ... \\n\\nSummary:"
      and then training added:
      src_with_task = f"<task_summarization> {source}"
    """
    base = (
        "[SUMMARIZATION]\n"
        "Summarise the following text in clear English:\n\n"
        f"{text}\n\n"
        "Summary:"
    )
    return f"<task_summarization> {base}"


def encode_input_from_source(source: str):
    """
    Encode a *ready* source string (already includes task token etc.).
    """
    seq = tok_in.texts_to_sequences([source])
    enc = pad_sequences(seq, maxlen=MAX_INPUT_LEN, padding="post", truncating="post")
    return enc


def encode_input_for_summarization(text: str):
    """
    Convenience wrapper for summarization.
    """
    src = build_source_summarization(text)
    return encode_input_from_source(src)


# -------------------------------------------------
# 4) Greedy decoding core
# -------------------------------------------------
def _decode(enc_tensor) -> str:
    """
    Greedy decode a single example.
    enc_tensor: shape (1, MAX_INPUT_LEN)
    Returns decoded text.
    """
    # start token
    dec_in = tf.fill([1, 1], start_id)
    collected = []

    for _ in range(MAX_TARGET_LEN):
        pad_len = MAX_TARGET_LEN - dec_in.shape[1]
        dec_padded = tf.pad(dec_in, [[0, 0], [0, pad_len]])

        # logits: (1, T, vocab)
        logits = model([enc_tensor, dec_padded], training=False)

        last_step = dec_in.shape[1] - 1
        next_tok = tf.argmax(logits[:, last_step, :], axis=-1)
        next_tok = int(next_tok.numpy()[0])

        # stop if end token reached
        if end_id != -1 and next_tok == end_id:
            break

        collected.append(next_tok)

        # grow decoder input
        next_tok_tensor = tf.convert_to_tensor([[next_tok]], dtype=tf.int32)
        dec_in = tf.concat([dec_in, next_tok_tensor], axis=1)

    # ids → words
    words = [
        tok_tgt.index_word.get(int(t), "<OOV>")
        for t in collected
        if t not in (0, start_id, end_id)
    ]
    return " ".join(words)


# -------------------------------------------------
# 5) High-level helpers
# -------------------------------------------------
def summarize(text: str) -> str:
    """
    High-level summarization helper.
    """
    enc = encode_input_for_summarization(text)
    enc = tf.convert_to_tensor(enc, dtype=tf.int32)
    return _decode(enc)


def generate_raw(source: str) -> str:
    """
    Low-level helper: you pass full 'source' string yourself.

    Example:
        source = "<task_math> Solve step by step: 7 * (5 + 2)"
        print(generate_raw(source))

    This lets you play with any task token you trained with.
    """
    enc = encode_input_from_source(source)
    enc = tf.convert_to_tensor(enc, dtype=tf.int32)
    return _decode(enc)


# -------------------------------------------------
# 6) Simple CLI interface
# -------------------------------------------------
def interactive_loop():
    print("\n=== Interactive interface ===")
    print("Modes:")
    print("  1) summarization   – you paste a document, model returns a summary")
    print("  2) raw             – you type full source incl. <task_...> token")
    print("  q) quit\n")

    while True:
        mode = input("Choose mode [1/2/q]: ").strip().lower()
        if mode == "q":
            print("Bye.")
            break

        if mode == "1":
            print("\nPaste text to summarise (end with a blank line):")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            doc = "\n".join(lines).strip()
            if not doc:
                print("No text provided.\n")
                continue
            print("\n--- SUMMARY ---")
            print(summarize(doc))
            print("---------------\n")

        elif mode == "2":
            print(
                "\nType full source including task token.\n"
                "Example: <task_math> Solve step by step: 7 * (5 + 2)\n"
                "End with a blank line."
            )
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            src = "\n".join(lines).strip()
            if not src:
                print("No source provided.\n")
                continue
            print("\n--- MODEL OUTPUT ---")
            print(generate_raw(src))
            print("--------------------\n")

        else:
            print("Unknown option, please choose 1, 2, or q.\n")


# -------------------------------------------------
# 7) Quick test + optional CLI
# -------------------------------------------------
if __name__ == "__main__":
    # If a text file path is passed, summarise its contents once.
    # Otherwise, start interactive loop.
    if len(sys.argv) > 1:
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        print("=== Single-run summary for file:", path, "===")
        print(summarize(content))
    else:
        interactive_loop()
