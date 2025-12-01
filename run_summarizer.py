import os
import sys
import json
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from tensorflow.keras.layers import (
    Attention,
    Concatenate,
    Dense,
    Embedding,
    Input,
    LSTMCell,
    Dropout,
    LayerNormalization,
    Add,
    MultiHeadAttention,
    Lambda,
)
from tensorflow.keras.models import Model

# ===== MUST MATCH TRAINING =====
MAX_VOCAB = 50_000
max_length_input = 256
max_length_target = 128
EMB_DIM = 50          # <- change if you trained with different emb_dim

MODEL_DIR = "app/models/saved_model"
TOK_IN_PATH = os.path.join(MODEL_DIR, "tokenizer_input.json")
TOK_TGT_PATH = os.path.join(MODEL_DIR, "tokenizer_target.json")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "summarization_model.weights.h5")


# -------------------------------------------------
# 0) (Optional) GPU memory-growth like in training
# -------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus, "GPU")
    except Exception:
        pass


# -------------------------------------------------
# 1) Architecture – must match training build_seq2seq_model
# -------------------------------------------------
def build_seq2seq_model(
    vocab_in,
    vocab_tgt,
    emb_dim,
    max_in,
    max_tgt,
    enc_units=128,
    dec_units=128,
    dropout_rate=0.3,
):
    enc_inputs = Input(shape=(max_in,), name="enc_inputs")
    enc_emb = Embedding(vocab_in, emb_dim, name="enc_emb")(enc_inputs)
    enc_emb = Dropout(dropout_rate, name="enc_emb_dropout")(enc_emb)

    # Encoder: 2-layer LSTM
    enc_cell1 = LSTMCell(enc_units, name="enc_cell1")
    enc_rnn1 = tf.keras.layers.RNN(
        enc_cell1,
        return_sequences=True,
        return_state=True,
        name="enc_rnn1",
    )
    out1, h1, c1 = enc_rnn1(enc_emb)

    enc_cell2 = LSTMCell(enc_units, name="enc_cell2")
    enc_rnn2 = tf.keras.layers.RNN(
        enc_cell2,
        return_sequences=True,
        return_state=True,
        name="enc_rnn2",
    )
    enc_outs, h2, c2 = enc_rnn2(out1)
    enc_states = [h2, c2]

    # Self-attention on encoder
    enc_self_attn = Attention(name="enc_self_attn")([enc_outs, enc_outs])

    # Fuse raw encoder outputs + self-attention
    enc_context_mix = Concatenate(name="enc_context_mix")(
        [enc_outs, enc_self_attn]
    )

    # Project back to enc_units
    enc_outs = Dense(
        enc_units,
        activation="tanh",
        name="enc_context_proj",
    )(enc_context_mix)

    enc_norm = LayerNormalization(name="enc_ln")(enc_outs)
    enc_ffn = Dense(enc_units * 4, activation="relu", name="enc_ffn1")(enc_norm)
    enc_ffn = Dense(enc_units, name="enc_ffn2")(enc_ffn)
    enc_outs = Add(name="enc_ffn_res")([enc_outs, enc_ffn])

    # ===== GLOBAL ENCODER BLOCK =====
    genc_ln1 = LayerNormalization(name="genc_ln1")(enc_outs)

    # Pool over time to get global summary
    genc_pool = Lambda(
        lambda x: tf.reduce_mean(x, axis=1),
        name="genc_pool_mean",
    )(genc_ln1)

    genc_query = Dense(
        enc_units,
        activation="tanh",
        name="genc_query",
    )(genc_pool)
    genc_query = Lambda(
        lambda x: tf.expand_dims(x, axis=1),
        name="genc_query_expand",
    )(genc_query)

    genc_attn = MultiHeadAttention(
        num_heads=4,
        key_dim=enc_units // 4,
        name="genc_mha",
    )(query=genc_query, value=genc_ln1, key=genc_ln1)  # (B, 1, enc_units)

    # Broadcast global token over time
    genc_broadcast = Lambda(
        lambda pair: tf.tile(
            pair[0], [1, tf.shape(pair[1])[1], 1]
        ),
        name="genc_broadcast",
    )([genc_attn, enc_outs])

    genc_res1 = Add(name="genc_res1")([enc_outs, genc_broadcast])

    genc_ln2 = LayerNormalization(name="genc_ln2")(genc_res1)
    genc_ffn1 = Dense(enc_units * 4, activation="relu", name="genc_ffn1")(genc_ln2)
    genc_ffn2 = Dense(enc_units, name="genc_ffn2")(genc_ffn1)

    enc_outs = Add(name="genc_ffn_res")([genc_ln2, genc_ffn2])
    # enc_outs: (B, T_enc, enc_units), globally enriched

    # ===== DECODER =====
    dec_inputs = Input(shape=(max_tgt,), name="dec_inputs")
    dec_emb = Embedding(vocab_tgt, emb_dim, name="dec_emb")(dec_inputs)
    dec_emb = Dropout(dropout_rate, name="dec_emb_dropout")(dec_emb)

    dec_cell1 = LSTMCell(dec_units, name="dec_cell1")
    dec_rnn1 = tf.keras.layers.RNN(
        dec_cell1,
        return_sequences=True,
        return_state=True,
        name="dec_rnn1",
    )
    dec_out1, _, _ = dec_rnn1(dec_emb, initial_state=enc_states)

    dec_cell2 = LSTMCell(dec_units, name="dec_cell2")
    dec_rnn2 = tf.keras.layers.RNN(
        dec_cell2,
        return_sequences=True,
        return_state=True,
        name="dec_rnn2",
    )
    dec_out2, _, _ = dec_rnn2(dec_out1)

    # Cross-attention: decoder → encoder
    cross_attn = Attention(name="cross_attn")([dec_out2, enc_outs])

    # Self-attention on decoder
    self_attn = Attention(name="self_attn")([dec_out2, dec_out2])

    fused = Add(name="decoder_fused")([dec_out2, cross_attn, self_attn])

    dec_norm = LayerNormalization(name="dec_ln")(fused)
    dec_ffn = Dense(dec_units * 4, activation="relu", name="dec_ffn1")(dec_norm)
    dec_ffn = Dense(dec_units, name="dec_ffn2")(dec_ffn)
    dec_context_res = Add(name="dec_ffn_res")([fused, dec_ffn])
    dec_context = LayerNormalization(name="dec_ln2")(dec_context_res)

    # ===== GLOBAL DECODER BLOCK =====
    gdec_ln1 = LayerNormalization(name="gdec_ln1")(dec_context)

    gdec_pool = Lambda(
        lambda x: tf.reduce_mean(x, axis=1),
        name="gdec_pool_mean",
    )(gdec_ln1)

    gdec_query = Dense(
        dec_units,
        activation="tanh",
        name="gdec_query",
    )(gdec_pool)
    gdec_query = Lambda(
        lambda x: tf.expand_dims(x, axis=1),
        name="gdec_query_expand",
    )(gdec_query)

    gdec_attn = MultiHeadAttention(
        num_heads=4,
        key_dim=dec_units // 4,
        name="gdec_mha",
    )(query=gdec_query, value=gdec_ln1, key=gdec_ln1)

    gdec_broadcast = Lambda(
        lambda pair: tf.tile(
            pair[0], [1, tf.shape(pair[1])[1], 1]
        ),
        name="gdec_broadcast",
    )([gdec_attn, dec_context])

    gdec_res1 = Add(name="gdec_res1")([dec_context, gdec_broadcast])

    gdec_ln2 = LayerNormalization(name="gdec_ln2")(gdec_res1)
    gdec_ffn1 = Dense(dec_units * 4, activation="relu", name="gdec_ffn1")(gdec_ln2)
    gdec_ffn2 = Dense(dec_units, name="gdec_ffn2")(gdec_ffn1)

    dec_context = Add(name="gdec_ffn_res")([gdec_ln2, gdec_ffn2])

    # ===== REFINE BLOCK =====
    refine_attn_norm = LayerNormalization(name="refine_attn_ln")(dec_context)
    refine_attn = MultiHeadAttention(
        num_heads=4,
        key_dim=dec_units // 4,
        name="refine_self_attn",
    )(refine_attn_norm, refine_attn_norm)
    refine_attn_res = Add(name="refine_attn_res")([dec_context, refine_attn])

    refine_ffn_norm = LayerNormalization(name="refine_ffn_ln")(refine_attn_res)
    refine_ffn = Dense(dec_units * 4, activation="relu", name="refine_ffn1")(refine_ffn_norm)
    refine_ffn = Dense(dec_units, name="refine_ffn2")(refine_ffn)
    dec_final = Add(name="refine_ffn_res")([refine_attn_res, refine_ffn])

    outputs = Dense(
        vocab_tgt,
        activation=None,
        name="decoder_dense",
        dtype="float32",
    )(dec_final)

    model = Model([enc_inputs, dec_inputs], outputs)
    return model


# -------------------------------------------------
# 2) Load tokenizers
# -------------------------------------------------
with open(TOK_IN_PATH, "r", encoding="utf-8") as f:
    tok_in = tokenizer_from_json(f.read())

with open(TOK_TGT_PATH, "r", encoding="utf-8") as f:
    tok_tgt = tokenizer_from_json(f.read())

oov_id = tok_tgt.word_index.get(tok_tgt.oov_token)
start_id = tok_tgt.word_index.get("<start>", tok_tgt.word_index.get("start", oov_id))
end_id = tok_tgt.word_index.get("<end>", tok_tgt.word_index.get("end", -1))

print("oov_id:", oov_id, "start_id:", start_id, "end_id:", end_id)

vs_in = min(len(tok_in.word_index) + 1, MAX_VOCAB + 1)
vs_tgt = min(len(tok_tgt.word_index) + 1, MAX_VOCAB + 1)

print("Vocab sizes – input:", vs_in, "target:", vs_tgt)


# -------------------------------------------------
# 3) Build model & load weights (.h5)
# -------------------------------------------------
print("Building model...")
model = build_seq2seq_model(
    vocab_in=vs_in,
    vocab_tgt=vs_tgt,
    emb_dim=EMB_DIM,
    max_in=max_length_input,
    max_tgt=max_length_target,
)

print("Loading weights from:", WEIGHTS_PATH)
model.load_weights(WEIGHTS_PATH, skip_mismatch=True)
print("Weights loaded.")


# -------------------------------------------------
# 4) Encode helpers (must mirror training)
# -------------------------------------------------
def build_source_summarization(text: str) -> str:
    """
    Training pipeline for summarization:
      source = "[SUMMARIZATION]\\nSummarise ... {doc} ... \\n\\nSummary:"
      then load_training_data added:
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
    seq = tok_in.texts_to_sequences([source])
    enc = pad_sequences(
        seq,
        maxlen=max_length_input,
        padding="post",
        truncating="post",
    )
    return enc


def encode_input_for_summarization(text: str):
    src = build_source_summarization(text)
    return encode_input_from_source(src)


# -------------------------------------------------
# 5) Core greedy decoder
# -------------------------------------------------
def _decode(enc_tensor) -> str:
    """
    Greedy decode a single example.
    enc_tensor: shape (1, max_length_input)
    """
    dec_in = tf.fill([1, 1], start_id)
    tokens = []

    for _ in range(max_length_target):
        pad_len = max_length_target - dec_in.shape[1]
        dec_padded = tf.pad(dec_in, [[0, 0], [0, pad_len]])

        logits = model([enc_tensor, dec_padded], training=False)
        last_step = dec_in.shape[1] - 1
        next_tok = tf.argmax(logits[:, last_step, :], axis=-1)
        next_tok = int(next_tok.numpy()[0])

        if end_id != -1 and next_tok == end_id:
            break

        tokens.append(next_tok)
        next_tok_tensor = tf.convert_to_tensor([[next_tok]], dtype=tf.int32)
        dec_in = tf.concat([dec_in, next_tok_tensor], axis=1)

    words = [
        tok_tgt.index_word.get(int(t), "<OOV>")
        for t in tokens
        if t not in (0, start_id, end_id)
    ]
    return " ".join(words)


# -------------------------------------------------
# 6) High-level helpers
# -------------------------------------------------
def summarize(text: str) -> str:
    enc = encode_input_for_summarization(text)
    enc = tf.convert_to_tensor(enc, dtype=tf.int32)
    return _decode(enc)


def generate_raw(source: str) -> str:
    """
    Low-level API: pass full source string (with <task_...> token etc.).
    Example:
        generate_raw("<task_math> 7 * (5 + 2) = ?")
    """
    enc = encode_input_from_source(source)
    enc = tf.convert_to_tensor(enc, dtype=tf.int32)
    return _decode(enc)


# -------------------------------------------------
# 7) Interactive CLI
# -------------------------------------------------
def interactive_loop():
    print("\n=== Interactive interface ===")
    print("Modes:")
    print("  1) summarization   – paste a document, get a summary")
    print("  2) raw             – send full source incl. <task_...>")
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
# 8) Entry point
# -------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Summarise file
        path = sys.argv[1]
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        print("=== Summary for file:", path, "===")
        print(summarize(content))
    else:
        interactive_loop()
