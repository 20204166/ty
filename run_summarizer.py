import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ==== must match training ====
MAX_INPUT_LEN = 256
MAX_TARGET_LEN = 128

# -------------------------------------------------
# 1) Load tokenizers
# -------------------------------------------------
with open("app/models/saved_model/tokenizer_input.json", "r", encoding="utf-8") as f:
    tok_in = tokenizer_from_json(f.read())

with open("app/models/saved_model/tokenizer_target.json", "r", encoding="utf-8") as f:
    tok_tgt = tokenizer_from_json(f.read())

start_id = tok_tgt.word_index.get("<start>")
end_id   = tok_tgt.word_index.get("<end>")

print("start_id:", start_id, "end_id:", end_id)

# -------------------------------------------------
# 2) Load model (no compile needed)
# -------------------------------------------------
model = tf.keras.models.load_model(
    "app/models/saved_model/summarization_model.keras",
    compile=False,   # we don't train, just forward pass
)
print("Model loaded.")


# -------------------------------------------------
# 3) Build source EXACTLY like training
#    (this is how generate_training_data.py created it)
# -------------------------------------------------
def build_source(text: str) -> str:
    """
    Training pipeline:
      source = "[SUMMARIZATION]\\nSummarise ... {doc} ... \\n\\nSummary:"
      then load_training_data added:
      src_with_task = f"<task_summarization> {source}"
    We must replicate that.
    """
    base = (
        "[SUMMARIZATION]\n"
        "Summarise the following text in clear English:\n\n"
        f"{text}\n\n"
        "Summary:"
    )
    return f"<task_summarization> {base}"


def encode_input(text: str):
    src = build_source(text)
    seq = tok_in.texts_to_sequences([src])
    enc = pad_sequences(seq, maxlen=MAX_INPUT_LEN, padding="post", truncating="post")
    return enc


# -------------------------------------------------
# 4) Greedy decoding
# -------------------------------------------------
def summarize(text: str) -> str:
    enc = encode_input(text)
    enc = tf.convert_to_tensor(enc, dtype=tf.int32)

    # start token
    dec_in = tf.fill([1, 1], start_id)
    tokens = []

    for _ in range(MAX_TARGET_LEN):
        pad_len = MAX_TARGET_LEN - dec_in.shape[1]
        dec_padded = tf.pad(dec_in, [[0, 0], [0, pad_len]])

        # logits shape: (1, T, vocab)
        logits = model([enc, dec_padded], training=False)

        # pick last time-step
        last_step = dec_in.shape[1] - 1
        next_tok = tf.argmax(logits[:, last_step, :], axis=-1)
        next_tok = int(next_tok.numpy()[0])

        if next_tok == end_id:
            break

        tokens.append(next_tok)
        # append to decoder input
        next_tok_tensor = tf.convert_to_tensor([[next_tok]], dtype=tf.int32)
        dec_in = tf.concat([dec_in, next_tok_tensor], axis=1)

    # ids → words
    words = [
        tok_tgt.index_word.get(int(t), "<OOV>")
        for t in tokens
        if t not in (0, start_id, end_id)
    ]
    return " ".join(words)


# -------------------------------------------------
# 5) Quick manual test
# -------------------------------------------------
if __name__ == "__main__":
    test_text = """
    Artificial intelligence is transforming many industries, including healthcare,
    finance, and education. However, it also raises questions about ethics,
    jobs, and regulation.
    """

    print("INPUT TEXT:\n", test_text)
    print("\nGENERATED SUMMARY:\n", summarize(test_text))
