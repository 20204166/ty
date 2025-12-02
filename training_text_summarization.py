import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"          # disable MKL/oneDNN fused ops
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_XLA_ENABLE_XLA_DEVICES"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import json
import subprocess
import random

import matplotlib.pyplot as plt
import numpy as np
import psutil
from rouge_score import rouge_scorer

import tensorflow as tf

from tensorflow.keras.callbacks import Callback, EarlyStopping, TerminateOnNaN

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
    Multiply, 
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

USE_MULTI_GPU = False # set False for 1 GPU, True for both




max_length_input = 256
max_length_target = 128
# Desired task ratios for multi-task training (only used if the data has "task")
TASK_RATIOS = {
    "summarization": 0.3,
    "code_cpp": 0.3,
    "math": 0.4,
}
# Optional cap on total number of examples after rebalancing
TASK_MAX_TOTAL = None  # e.g. 500_000 or None for "whatever the data allows"
ENABLE_TASK_SAMPLING = True


def sample_by_task_ratios(data, ratios, max_total=None, seed=42):
    """
    Re-sample a mixed-task dataset to roughly match the given ratios.

    Args:
        data: list of dicts (each may have 'task' key).
        ratios: dict like {"summarization": 0.4, "code_cpp": 0.4, "math": 0.2}.
        max_total: optional cap on total number of examples (int) or None.
        seed: random seed.

    Returns:
        A new list of dicts sampled according to the ratios, shuffled.
    """
    random.seed(seed)

    # Group by task (default "generic" if not present)
    groups = {}
    for item in data:
        task = str(item.get("task", "generic")).strip().lower().replace(" ", "_")
        groups.setdefault(task, []).append(item)

    # Only consider tasks that appear both in data and in ratios
    available_tasks = [t for t in ratios.keys() if t in groups and ratios[t] > 0]
    if not available_tasks:
        # Nothing to rebalance, just return original data
        return data

    # Find the maximum total we can sample without oversampling any task
    base_totals = []
    for t in available_tasks:
        r = ratios[t]
        if r <= 0:
            continue
        # To keep ratio r for task t, max total is len(group[t]) / r
        base_totals.append(len(groups[t]) / r)

    if not base_totals:
        return data

    base_total = min(base_totals)
    if max_total is not None:
        base_total = min(base_total, max_total)

    # Sample per task
    result = []
    for t in available_tasks:
        r = ratios[t]
        if r <= 0:
            continue
        n_t = int(round(base_total * r))
        n_t = min(n_t, len(groups[t]))
        result.extend(random.sample(groups[t], n_t))

    random.shuffle(result)
    return result

def load_training_data(data_path: str, input_key: str = None, target_key: str = None):
    """
    General loader for:
      - New multi-task format: {"source": ..., "target": ..., "task": "..."}
      - Old summarisation formats: {"text": ..., "summary": ...} or {"article": ..., "highlights": ...}

    Returns:
        inputs:  list of input strings
        targets: list of output strings with <start> ... <end>
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        cleaned = []
        for item in data:
            src = item.get("source") or item.get("text") or item.get("article")
            tgt = item.get("target") or item.get("summary") or item.get("highlights")
            if src is None or tgt is None:
                cleaned.append(item)  # handled by branch logic below
                continue
            if str(src).strip() and str(tgt).strip():
                cleaned.append(item)
        data = cleaned


    if not data or not isinstance(data[0], dict):
        raise ValueError("Training data must be a non-empty list of objects.")

    # If explicit keys are provided and present, use them
    if input_key is not None and target_key is not None and input_key in data[0] and target_key in data[0]:
        inputs = []
        targets = []
        for item in data:
            src = str(item[input_key])
            tgt = str(item[target_key])
            inputs.append(src)
            targets.append(f"<start> {tgt} <end>")
        return inputs, targets

    # 1) NEW multi-task format: source / target / task
    if "source" in data[0] and "target" in data[0]:
        # Optional per-task rebalancing
        if ENABLE_TASK_SAMPLING:
            data = sample_by_task_ratios(data, TASK_RATIOS, max_total=TASK_MAX_TOTAL)

        inputs = []
        targets = []
        for item in data:
            src = str(item["source"])
            tgt = str(item["target"])
            task = str(item.get("task", "generic")).strip().lower().replace(" ", "_")

            # Prefix task token so one model can do summarization, C++, math, etc.
            src_with_task = f"<task_{task}> {src}"
            inputs.append(src_with_task)
            targets.append(f"<start> {tgt} <end>")

        return inputs, targets

    # 2) Backwards-compatible summarisation formats

    # CNN/DailyMail-style
    if "article" in data[0] and "highlights" in data[0]:
        inputs = [str(item["article"]) for item in data]
        targets = [f"<start> {item['highlights']} <end>" for item in data]
        return inputs, targets

    # Generic text/summary
    if "text" in data[0] and "summary" in data[0]:
        inputs = [str(item["text"]) for item in data]
        targets = [f"<start> {item['summary']} <end>" for item in data]
        return inputs, targets

    raise ValueError(
        "Unsupported data format. Expected one of:\n"
        "  - {'source', 'target', 'task'}\n"
        "  - {'text', 'summary'}\n"
        "  - {'article', 'highlights'}\n"
        "  Or pass explicit input_key/target_key."
    )



MAX_VOCAB = 50_000


def create_tokenizer(texts, oov_token="<OOV>", max_words=MAX_VOCAB):
    """
    Simple tokenizer: no extra <start>/<end> here.
    Those are already added inside load_training_data for targets.
    """
    tok = Tokenizer(num_words=max_words, oov_token=oov_token)
    tok.fit_on_texts(texts)
    return tok



def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())


def preprocess_texts(texts, tokenizer, max_length, max_vocab):
    sequences = tokenizer.texts_to_sequences(texts)
    arr = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    arr = np.where(arr >= max_vocab, 1, arr)
    return arr


def prepare_decoder_sequences(sequences):
    dec_in = sequences[:, :-1]
    dec_tgt = sequences[:, 1:]
    pad_width = max_length_target - dec_in.shape[1]
    if pad_width > 0:
        dec_in = np.pad(dec_in, ((0, 0), (0, pad_width)), mode="constant")
        dec_tgt = np.pad(dec_tgt, ((0, 0), (0, pad_width)), mode="constant")
    return dec_in, dec_tgt
def masked_sparse_ce(y_true, y_pred):
    """
    Stable sparse cross-entropy with padding mask.
    - y_true: (batch, T) integer labels (0 = pad)
    - y_pred: (batch, T, vocab) logits
    """
    # --- dtypes ---
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)

    # --- vocab & label range ---
    vocab_size = tf.shape(y_pred)[-1]
    # Clamp labels into [0, vocab_size-1] just in case
    y_true_clipped = tf.clip_by_value(y_true, 0, vocab_size - 1)

    # --- numerically stable log-softmax ---
    y_pred = tf.clip_by_value(y_pred, -30.0, 30.0)
    log_probs = tf.nn.log_softmax(y_pred, axis=-1)  # (B, T, V)

    # --- gather log p(true_class) for each token ---
    batch_size = tf.shape(y_true_clipped)[0]
    seq_len = tf.shape(y_true_clipped)[1]

    batch_idx = tf.range(batch_size)[:, tf.newaxis]          # (B, 1)
    time_idx = tf.range(seq_len)[tf.newaxis, :]              # (1, T)
    batch_idx = tf.tile(batch_idx, [1, seq_len])             # (B, T)
    time_idx = tf.tile(time_idx, [batch_size, 1])            # (B, T)

    indices = tf.stack([batch_idx, time_idx, y_true_clipped], axis=-1)
    # indices: (B, T, 3)

    true_log_probs = tf.gather_nd(log_probs, indices)        # (B, T)

    # cross-entropy per token = -log p(true_class)
    ce = -true_log_probs                                     # (B, T)

    # Replace any crazy values with a big but finite constant
    ce = tf.where(tf.math.is_finite(ce), ce, tf.zeros_like(ce) + 50.0)

    # --- mask padding (id=0) ---
    mask = tf.cast(tf.not_equal(y_true_clipped, 0), tf.float32)  # (B, T)
    ce = ce * mask

    denom = tf.reduce_sum(mask) + 1e-8
    return tf.reduce_sum(ce) / denom




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

    enc_self_attn = Attention(name="enc_self_attn")([enc_outs, enc_outs])

    # Fuse raw encoder outputs + self-attention into a single sequence
    enc_context_mix = Concatenate(name="enc_context_mix")([enc_outs, enc_self_attn])

    # Project back to enc_units so enc_outs keeps the SAME SHAPE as before
    enc_outs = Dense(
        enc_units,
        activation="tanh",
        name="enc_context_proj",
    )(enc_context_mix)
    
    # enc_outs is now a richer encoder representation, but same shape
    enc_norm = LayerNormalization(name="enc_ln")(enc_outs)
    enc_ffn = Dense(enc_units * 4, activation="relu", name="enc_ffn1")(enc_norm)
    enc_ffn = Dense(enc_units, name="enc_ffn2")(enc_ffn)
    enc_outs = Add(name="enc_ffn_res")([enc_outs, enc_ffn])

    enc_local = enc_outs

    # ===== GLOBAL ENCODER BLOCK (hierarchical on encoder side) =====
    

    genc_ln1 = LayerNormalization(name="genc_ln1")(enc_outs)

    # Pool over time to get a global summary
    genc_pool = Lambda(
        lambda x: tf.reduce_mean(x, axis=1),
        name="genc_pool_mean",
    )(genc_ln1)
  
    # Turn pooled vector into a "global query"
    genc_query = Dense(
        enc_units,
        activation="tanh",
        name="genc_query",
    )(genc_pool)                   # (B, enc_units)
    genc_query = Lambda(
        lambda x: tf.expand_dims(x, axis=1),
        name="genc_query_expand",
    )(genc_query)  

    # Multi-head attention: global token attends over full encoder sequence
    genc_attn = MultiHeadAttention(
        num_heads=4,
        key_dim=enc_units // 4,
        name="genc_mha",
    )(query=genc_query, value=genc_ln1, key=genc_ln1)  # (B, 1, enc_units)
 
    # ---- encoder global broadcast ----
    genc_broadcast = Lambda(
        lambda pair: tf.tile(
            pair[0], [1, tf.shape(pair[1])[1], 1]
        ),
        name="genc_broadcast",
    )([genc_attn, enc_outs])


    # Residual: add global context onto encoder outputs
    genc_res1 = Add(name="genc_res1")([enc_outs, genc_broadcast])

    genc_ln2 = LayerNormalization(name="genc_ln2")(genc_res1)
    genc_ffn1 = Dense(enc_units * 4, activation="relu", name="genc_ffn1")(genc_ln2)
    genc_ffn2 = Dense(enc_units, name="genc_ffn2")(genc_ffn1)

    enc_outs = Add(name="genc_ffn_res")([genc_ln2, genc_ffn2])
    # enc_outs stays shape (B, T_enc, enc_units) but is now globally enriched

    enc_tf_proj = Dense(
        512,
        activation="relu",
        name="enc_tf_proj",
    )(enc_outs) 

    enc_tf_ln1 = LayerNormalization(name="enc_tf_ln1")(enc_tf_proj)

    enc_tf_mha = MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        name="enc_tf_mha",
    )(enc_tf_ln1, enc_tf_ln1)      # (B, T_enc, 512)

    enc_tf_res1 = Add(name="enc_tf_res1")([enc_tf_proj, enc_tf_mha])

    enc_tf_ln2 = LayerNormalization(name="enc_tf_ln2")(enc_tf_res1)
    enc_tf_ffn1 = Dense(2048, activation="relu", name="enc_tf_ffn1")(enc_tf_ln2)
    enc_tf_ffn2 = Dense(512, name="enc_tf_ffn2")(enc_tf_ffn1)
    enc_tf_res2 = Add(name="enc_tf_res2")([enc_tf_res1, enc_tf_ffn2])

    enc_tf_back = Dense(
        enc_units,
        name="enc_tf_backproj",
    )(enc_tf_res2)                 

    # Final encoder representation: LSTM + global + transformer
    enc_outs = Add(name="enc_tf_out_res")([enc_outs, enc_tf_back])
    


    # Decoder: 2-layer LSTM
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

    cross_attn_local = Attention(name="cross_attn_local")([dec_out2, enc_local])


    # Self-attention on the decoder outputs: decoder → decoder
    self_attn = Attention(name="self_attn")([dec_out2, dec_out2])
    
    # but KEEP the last dim = dec_units (no shape change)
    fused = Add(name="decoder_fused")(
        [dec_out2, cross_attn, cross_attn_local, self_attn]
    )



    dec_norm = LayerNormalization(name="dec_ln")(fused)

    # FFN on normalized fused representation
    dec_ffn = Dense(dec_units * 4, activation="relu", name="dec_ffn1")(dec_norm)
    dec_ffn = Dense(dec_units, name="dec_ffn2")(dec_ffn)

    # Residual connection back to fused
    dec_context_res = Add(name="dec_ffn_res")([fused, dec_ffn])

    # 2nd LayerNorm on the residual output
    dec_context = LayerNormalization(name="dec_ln2")(dec_context_res)

    # STREAM 2: shallow decoder-only linear stream (extra linear reasoning path)
    dec_linear = Dense(
        dec_units,
        activation="tanh",
        name="dec_linear_stream",
    )(dec_context)

    lin_tf_proj = Dense(
        256,
        activation="relu",
        name="lin_tf_proj",
    )(dec_linear)   # (B, T_dec, 256)

    lin_tf_ln1 = LayerNormalization(name="lin_tf_ln1")(lin_tf_proj)

    lin_tf_mha = MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        name="lin_tf_mha",
    )(lin_tf_ln1, lin_tf_ln1)  # (B, T_dec, 256)

    lin_tf_res1 = Add(name="lin_tf_res1")([lin_tf_proj, lin_tf_mha])

    lin_tf_ln2 = LayerNormalization(name="lin_tf_ln2")(lin_tf_res1)
    lin_tf_ffn1 = Dense(1024, activation="relu", name="lin_tf_ffn1")(lin_tf_ln2)
    lin_tf_ffn2 = Dense(256, name="lin_tf_ffn2")(lin_tf_ffn1)
    lin_tf_res2 = Add(name="lin_tf_res2")([lin_tf_res1, lin_tf_ffn2])

    lin_tf_back = Dense(dec_units, name="lin_tf_backproj")(lin_tf_res2)
    dec_linear_enh = Add(name="lin_tf_out_res")([dec_linear, lin_tf_back])

    # ===== GLOBAL DECODER BLOCK (hierarchical on decoder side) =====
    

    gdec_ln1 = LayerNormalization(name="gdec_ln1")(dec_context)

    # Pool over time (global token)
    gdec_pool = Lambda(
        lambda x: tf.reduce_mean(x, axis=1),
        name="gdec_pool_mean",
    )(gdec_ln1)

    gdec_query = Dense(
        dec_units,
        activation="tanh",
        name="gdec_query",
    )(gdec_pool)                    # (B, dec_units)
    gdec_query = Lambda(
        lambda x: tf.expand_dims(x, axis=1),
        name="gdec_query_expand",
    )(gdec_query)

    gdec_attn = MultiHeadAttention(
        num_heads=4,
        key_dim=dec_units // 4,
        name="gdec_mha",
    )(query=gdec_query, value=gdec_ln1, key=gdec_ln1)  # (B, 1, dec_units)

    # ---- decoder global broadcast ----
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
    # dec_context now = GLOBAL + local, same shape as before

    
    ### === TRANSFORMER BLOCK 1 (TF1, 512-dim) AFTER GLOBAL DECODER ===
    # Use encoder summary + decoder context together

    tf1_enc_pool = Lambda(
        lambda x: tf.reduce_mean(x, axis=1),
        name="tf1_enc_pool",
    )(enc_outs)  # (B, 128)

    tf1_enc_proj = Dense(
        dec_units,
        activation="tanh",
        name="tf1_enc_proj",
    )(tf1_enc_pool)  # (B, 128)

    tf1_enc_expand = Lambda(
        lambda x: tf.expand_dims(x, axis=1),
        name="tf1_enc_expand",
    )(tf1_enc_proj)  # (B, 1, 128)

    tf1_enc_broadcast = Lambda(
        lambda pair: tf.tile(pair[0], [1, tf.shape(pair[1])[1], 1]),
        name="tf1_enc_broadcast",
    )([tf1_enc_expand, dec_context])  # (B, T_dec, 128)

    tf1_in = Concatenate(name="tf1_in_concat")(
        [dec_context, tf1_enc_broadcast]
    )  # (B, T_dec, 256)

    tf1_proj = Dense(
        512,
        activation="relu",
        name="tf1_proj",
    )(tf1_in)  # (B, T_dec, 512)

    tf1_ln1 = LayerNormalization(name="tf1_ln1")(tf1_proj)

    tf1_mha = MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        name="tf1_mha",
    )(tf1_ln1, tf1_ln1)  # (B, T_dec, 512)

    tf1_res1 = Add(name="tf1_res1")([tf1_proj, tf1_mha])

    tf1_ln2 = LayerNormalization(name="tf1_ln2")(tf1_res1)
    tf1_ffn1 = Dense(2048, activation="relu", name="tf1_ffn1")(tf1_ln2)
    tf1_ffn2 = Dense(512, name="tf1_ffn2")(tf1_ffn1)
    tf1_res2 = Add(name="tf1_res2")([tf1_res1, tf1_ffn2])

    tf1_back = Dense(dec_units, name="tf1_backproj")(tf1_res2)
    dec_context = Add(name="tf1_out_res")([dec_context, tf1_back])
    
    #NEW STREAM 3: dual-side synapse path (encoder <-> decoder)
    # Project encoder + decoder into a shared space and let them attend to each other.
    syn_enc = Dense(
        dec_units,
        activation="tanh",
        name="syn_enc_proj",
    )(enc_outs)
    syn_dec = Dense(
        dec_units,
        activation="tanh",
        name="syn_dec_proj",
    )(dec_context)  # (B, T_dec, dec_units)

    syn_attn = Attention(name="syn_cross_attn")([syn_dec, syn_enc])  
    
    # ===== REFINE BLOCK (also 2× LayerNorm) =====

    # LN before refine self-attention
    refine_attn_norm = LayerNormalization(
        name="refine_attn_ln"
    )(dec_context)

    refine_attn = MultiHeadAttention(
        num_heads=4,
        key_dim=dec_units // 4,
        name="refine_self_attn",
    )(refine_attn_norm, refine_attn_norm)

    # Residual: attention + original dec_context
    refine_attn_res = Add(name="refine_attn_res")([dec_context, refine_attn])

    # LN before refine FFN
    refine_ffn_norm = LayerNormalization(
        name="refine_ffn_ln"
    )(refine_attn_res)

    refine_ffn = Dense(
        dec_units * 4,
        activation="relu",
        name="refine_ffn1",
    )(refine_ffn_norm)

    refine_ffn = Dense(
        dec_units,
        name="refine_ffn2",
    )(refine_ffn)
    
    # 2 small "synapse" gates that modulate the two extra streams
   # gate sees the *enriched* linear stream
    syn_gate1 = Dense(
        dec_units,
        activation="sigmoid",
        name="syn_gate1",
    )(dec_linear_enh)   # (B, T, 128)

    # 1 - gate (how much to keep of the raw linear stream)
    syn_gate1_inv = Lambda(
        lambda g: 1.0 - g,
        name="syn_gate1_inv",
    )(syn_gate1)

    # raw part: dec_linear * (1 - gate)
    syn_part_raw = Multiply(name="syn_part_raw")([dec_linear, syn_gate1_inv])

    # enriched part: dec_linear_enh * gate
    syn_part_enh = Multiply(name="syn_part_enh")([dec_linear_enh, syn_gate1])

    # mixed: per-token mix of raw vs enriched
    gated_syn1 = Add(name="syn_gated1")([syn_part_raw, syn_part_enh])

    syn_gate2 = Dense(
        dec_units,
        activation="sigmoid",
        name="syn_gate2",
    )(syn_attn)
    
    gated_syn2 = Multiply(name="syn_gated2")([syn_attn, syn_gate2])

    # Residual again
    dec_final = Add(name="refine_ffn_res")(
        [refine_attn_res, refine_ffn, gated_syn1, gated_syn2]
    )

    ### === TRANSFORMER BLOCK 2 (TF2, 512-dim) BEFORE LOGITS ===
    
    tf2_proj = Dense(
        512,
        activation="relu",
        name="tf2_proj",
    )(dec_final)  # (B, T_dec, 512)

    tf2_ln1 = LayerNormalization(name="tf2_ln1")(tf2_proj)

    tf2_mha = MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        name="tf2_mha",
    )(tf2_ln1, tf2_ln1)  # (B, T_dec, 512)

    tf2_res1 = Add(name="tf2_res1")([tf2_proj, tf2_mha])

    tf2_ln2 = LayerNormalization(name="tf2_ln2")(tf2_res1)
    tf2_ffn1 = Dense(2048, activation="relu", name="tf2_ffn1")(tf2_ln2)
    tf2_ffn2 = Dense(512, name="tf2_ffn2")(tf2_ffn1)
    tf2_res2 = Add(name="tf2_res2")([tf2_res1, tf2_ffn2])

    tf2_back = Dense(dec_units, name="tf2_backproj")(tf2_res2)
    dec_final = Add(name="tf2_out_res")([dec_final, tf2_back])
    
    # Final logits from refined decoder representation
    outputs = Dense(
        vocab_tgt,
        activation=None,
        name="decoder_dense",
        dtype="float32",  # keep float32 with mixed precision
    )(dec_final)

    model = Model([enc_inputs, dec_inputs], outputs)
    return model



def plot_history(history, save_dir):
    h = history.history
    keys = h.keys()
    epochs = range(1, len(h.get("loss", [])) + 1)

    # 1) Loss
    plt.figure()
    plt.plot(epochs, h["loss"], label="Train loss")
    plt.plot(epochs, h["val_loss"], label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_full.png"))
    plt.close()

    # 2) Accuracy
    acc_key = "token_accuracy" if "token_accuracy" in keys else "sparse_categorical_accuracy"
    val_acc_key = "val_" + acc_key

    plt.figure()
    plt.plot(epochs, h[acc_key], label="Train acc")
    plt.plot(epochs, h[val_acc_key], label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs. Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "acc_full.png"))
    plt.close()


class SnapshotCallback(Callback):
    def __init__(self, save_dir, interval_epochs=10):
        super().__init__()
        self.save_dir = save_dir
        self.interval = interval_epochs
        os.makedirs(self.save_dir, exist_ok=True)
        self.cpu_usage = []
        self.ram_usage = []
        self.gpu_usage = {}
        # keep track of it own metrics history 
        self.history = {}

    def _get_gpu_utils(self):
        try:
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            return [float(line) for line in raw.decode().splitlines() if line.strip()]
        except Exception:
            return []

    def _plot_metrics(self, upto, suffix):
        h = self.history
        if "loss" not in h:
            return
        
        upto = min(upto, len(h["loss"]))
        epochs = range(1, upto + 1)
        # 1) Loss
        plt.figure()
        plt.plot(epochs, h["loss"][:upto], label="Train loss")
        if "val_loss" in h:
            plt.plot(epochs, h["val_loss"][:upto], label="Val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Loss (1–{upto})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"loss_1to{upto}{suffix}.png"))
        plt.close()
        
        # 2) Token‐accuracy (your real metric)
        if "token_accuracy" in h:
            plt.figure()
            plt.plot(epochs, h["token_accuracy"][:upto], label="Train token-acc")
            if "val_token_accuracy" in h:
                plt.plot(epochs, h["val_token_accuracy"][:upto], label="Val token-acc")
            plt.xlabel("Epoch")
            plt.ylabel("Token Accuracy")
            plt.title(f"Token Accuracy (1–{upto})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"token_acc_1to{upto}{suffix}.png"))
            plt.close()


        # 3) ROUGE
        if "val_rouge1" in h:
            plt.figure()
            plt.plot(epochs, h["val_rouge1"][:upto], label="ROUGE-1")
            if "val_rouge2" in h:
                plt.plot(epochs, h["val_rouge2"][:upto], label="ROUGE-2")
            if "val_rougeL" in h:
                plt.plot(epochs, h["val_rougeL"][:upto], label="ROUGE-L")
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.title(f"ROUGE (1–{upto})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"rouge_1to{upto}{suffix}.png"))
            plt.close() 

    def _plot_resources(self, upto, suffix):
        epochs = range(1, upto + 1)

        # CPU & RAM
        plt.figure()
        plt.plot(epochs, self.cpu_usage[:upto], label="CPU %")
        plt.plot(epochs, self.ram_usage[:upto], label="RAM %")
        plt.xlabel("Epoch")
        plt.ylabel("Percent")
        plt.title(f"CPU & RAM (1–{upto})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"cpu_ram_1to{upto}{suffix}.png"))
        plt.close()

        # Per-GPU
        for i, util in self.gpu_usage.items():
            plt.figure()
            plt.plot(epochs, util[:upto], label=f"GPU{i} %")
            plt.xlabel("Epoch")
            plt.ylabel("GPU Util %")
            plt.title(f"GPU {i} (1–{upto})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, f"gpu{i}_1to{upto}{suffix}.png"))
            plt.close()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 1) Track resource usage
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        self.cpu_usage.append(cpu)
        self.ram_usage.append(ram)

        gpu_utils = self._get_gpu_utils()
        if gpu_utils:
            for i, util in enumerate(gpu_utils):
                if i not in self.gpu_usage:
                    self.gpu_usage[i] = []
                self.gpu_usage[i].append(util)

        # 2) Track metric values from logs
        for k, v in logs.items():
            # only store simple numeric metrics
            try:
                val = float(v)
            except (TypeError, ValueError):
                continue
            self.history.setdefault(k, []).append(val)

        # 3) Periodic plotting
        if (epoch + 1) % self.interval == 0:
            suffix = f"_ep{epoch+1}"
            self._plot_metrics(epoch + 1, suffix)
            self._plot_resources(epoch + 1, suffix)


    def on_train_end(self, logs=None):
        total = len(self.history.get("loss",[]))
        if total > 0:
            self._plot_metrics(total, "_final")
            self._plot_resources(total, "_final")


class SamplePrediction(Callback):
    def __init__(self, val_ds, tokenizer, max_len, samples=1, save_path="sample_pred.png"):
        super().__init__()

        self.val_ds = val_ds.take(1).unbatch().batch(samples)
        self.tokenizer = tokenizer
        self.start_id = tokenizer.word_index.get("<start>", tokenizer.word_index[tokenizer.oov_token])
        self.end_id = tokenizer.word_index.get("<end>", tokenizer.word_index[tokenizer.oov_token])
        self.max_length = max_len
        self.save_path = save_path

    def on_train_end(self, logs=None):

        for (enc, _), _ in self.val_ds:

            dec_in = tf.fill([enc.shape[0], 1], self.start_id)
            result = []

            for _ in range(self.max_length):
                pad_len = self.max_length - dec_in.shape[1]
                logits = self.model(
                    [enc, tf.pad(dec_in, [[0, 0], [0, pad_len]])],
                    training=False,
                )
                next_tok = tf.argmax(logits[:, -1, :], axis=-1, output_type=tf.int32)
                dec_in = tf.concat([dec_in, next_tok[:, None]], axis=1)
                result.append(next_tok)
            preds = tf.stack(result, axis=1).numpy()

            # only take the first sample for display
            pred_seq = preds[0]
            # cut off at the first <end> token
            if self.end_id in pred_seq:
                cut = list(pred_seq).index(self.end_id)
                pred_seq = pred_seq[:cut]

            words = [self.tokenizer.index_word.get(int(w), "<OOV>") for w in pred_seq if w not in (0, self.start_id)]
            text = " ".join(words)

            plt.figure(figsize=(8, 1.5))
            plt.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=12)
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(self.save_path, dpi=150)
            plt.close()

            print(f" Saved sample prediction to {self.save_path}")
            break


class RougeCallback(Callback):
    def __init__(self, val_ds, tgt_tokenizer, max_length_target, n_samples):
        super().__init__()
        self.val_ds = val_ds
        self.tokenizer = tgt_tokenizer
        self.start_id = tgt_tokenizer.word_index.get("<start>", tgt_tokenizer.word_index[tgt_tokenizer.oov_token])
        self.end_id = tgt_tokenizer.word_index.get("<end>", tgt_tokenizer.word_index[tgt_tokenizer.oov_token])
        self.max_length = max_length_target
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        self.n_samples = n_samples

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        (enc_batch, _), dec_tgt_batch = next(iter(self.val_ds))
        total = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        batch = tf.shape(enc_batch)[0]

        dec_input = tf.fill([batch, 1], self.start_id)
        result = tf.zeros([batch, 0], dtype=tf.int32)
        for _ in range(self.max_length):
            pad_amt = self.max_length - tf.shape(dec_input)[1]
            dec_padded = tf.pad(dec_input, [[0, 0], [0, pad_amt]])
            logits = self.model([enc_batch, dec_padded], training=False)
        
            next_token = tf.cast(tf.argmax(logits[:, tf.shape(dec_input)[1]-1, :], axis=-1), tf.int32)
            
            dec_input = tf.concat([dec_input, next_token[:, None]], axis=1)
            result = tf.concat([result, next_token[:, None]], axis=1)


        count = 0
        for ref_seq, pred_seq in zip(dec_tgt_batch.numpy(), result.numpy()):
            def seq_to_text(seq):
                words = []
                for w in seq:
                    if w in (0, self.start_id):
                        continue
                    if w == self.end_id:
                        break
                    words.append(self.tokenizer.index_word.get(w, ""))
                return " ".join(words)
                
            ref_txt  = seq_to_text(ref_seq)
            pred_txt = seq_to_text(pred_seq)
            
            sc = self.scorer.score(ref_txt, pred_txt)
            total["rouge1"] += sc["rouge1"].fmeasure
            total["rouge2"] += sc["rouge2"].fmeasure
            total["rougeL"] += sc["rougeL"].fmeasure
            count += 1
        avg = {k: (total[k] / count if count else 0.0) for k in total}
        logs.update({f"val_{k}": v for k, v in avg.items()})


class SaveOnAnyImprovement(tf.keras.callbacks.Callback):
    def __init__(
        self,
        model_path,
        weights_path,
        monitor="val_token_accuracy",  # primary metric to "watch"
        mode="max",                    # "max" for acc/rouge, "min" for loss
        min_acc_gain=0.001,            # extra condition for token_acc
        max_loss_increase=0.10,        # allow up to +10% worse loss when saving on acc
    ):
        super().__init__()
        self.model_path = model_path
        self.weights_path = weights_path
        self.monitor = monitor
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

        self.min_acc_gain = float(min_acc_gain)
        self.max_loss_increase = float(max_loss_increase)

        # where we persist best metrics across runs
        self.best_file = weights_path + ".best.json"

        # defaults if nothing on disk yet
        if self.mode == "min":
            self.best_primary = float("inf")
        else:
            self.best_primary = float("-inf")

        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.best_rouges = {
            "val_rouge1": 0.0,
            "val_rouge2": 0.0,
            "val_rougeL": 0.0,
        }

        # try to load previous best values
        if os.path.exists(self.best_file):
            try:
                with open(self.best_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self.monitor in data:
                    self.best_primary = float(data[self.monitor])
                if "val_loss" in data:
                    self.best_loss = float(data["val_loss"])
                if "val_token_accuracy" in data:
                    self.best_acc = float(data["val_token_accuracy"])
                for k in self.best_rouges.keys():
                    if k in data:
                        self.best_rouges[k] = float(data[k])

                print(
                    f"[SaveOnAnyImprovement] Loaded previous bests: "
                    f"{self.monitor}={self.best_primary:.6f}, "
                    f"val_loss={self.best_loss:.6f}, "
                    f"val_token_accuracy={self.best_acc:.6f}, "
                    f"ROUGE1={self.best_rouges['val_rouge1']:.4f}, "
                    f"ROUGEL={self.best_rouges['val_rougeL']:.4f}"
                )
            except Exception as e:
                print(
                    f"[SaveOnAnyImprovement] Could not read {self.best_file} "
                    f"({e}), starting fresh."
                )

    def _primary_improved(self, current):
        if self.mode == "min":
            return current < self.best_primary
        else:
            return current > self.best_primary

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # pull all metrics we might use
        current_primary = logs.get(self.monitor)
        new_loss = logs.get("val_loss")
        new_acc = logs.get("val_token_accuracy")
        new_r1 = logs.get("val_rouge1")
        new_r2 = logs.get("val_rouge2")
        new_rL = logs.get("val_rougeL")

        should_save = False
        reasons = []

        # 1) "monitor" logic like ModelCheckpoint (primary metric)
        if current_primary is not None and self._primary_improved(current_primary):
            should_save = True
            reasons.append(
                f"{self.monitor} improved {self.best_primary:.6f} → {float(current_primary):.6f}"
            )

        # 2) ROUGE-based triggers (like your old version)
        if new_r1 is not None and new_r1 > self.best_rouges["val_rouge1"]:
            should_save = True
            reasons.append(f"val_rouge1 ↑ {self.best_rouges['val_rouge1']:.4f} → {new_r1:.4f}")

        if new_rL is not None and new_rL > self.best_rouges["val_rougeL"]:
            should_save = True
            reasons.append(f"val_rougeL ↑ {self.best_rouges['val_rougeL']:.4f} → {new_rL:.4f}")

        # 3) token_accuracy with loss guard
        if new_acc is not None and new_loss is not None:
            acc_improved = new_acc > self.best_acc + self.min_acc_gain
            loss_not_too_bad = new_loss < self.best_loss * (1.0 + self.max_loss_increase)
            if acc_improved and loss_not_too_bad:
                should_save = True
                reasons.append(
                    f"val_token_accuracy ↑ {self.best_acc:.4f} → {new_acc:.4f} "
                    f"with val_loss {new_loss:.4f} (best {self.best_loss:.4f})"
                )

        # if we never saved before but have some metrics, save at least once
        if not os.path.exists(self.weights_path) and current_primary is not None:
            should_save = True
            reasons.append(f"first time saving {self.monitor}={float(current_primary):.6f}")

        # --- actually save if ANY of the above fired ---
        if should_save:
            # update bests
            if current_primary is not None:
                self.best_primary = float(current_primary)

            if new_loss is not None:
                self.best_loss = min(self.best_loss, float(new_loss))

            if new_acc is not None:
                self.best_acc = max(self.best_acc, float(new_acc))

            if new_r1 is not None:
                self.best_rouges["val_rouge1"] = max(
                    self.best_rouges["val_rouge1"], float(new_r1)
                )
            if new_r2 is not None:
                self.best_rouges["val_rouge2"] = max(
                    self.best_rouges["val_rouge2"], float(new_r2)
                )
            if new_rL is not None:
                self.best_rouges["val_rougeL"] = max(
                    self.best_rouges["val_rougeL"], float(new_rL)
                )

            # write all bests to json so resume works
            best_dict = {
                self.monitor: self.best_primary,
                "val_loss": self.best_loss,
                "val_token_accuracy": self.best_acc,
                **self.best_rouges,
            }
            try:
                with open(self.best_file, "w", encoding="utf-8") as f:
                    json.dump(best_dict, f)
            except Exception as e:
                print(f"[SaveOnAnyImprovement] Warning: could not write best file: {e}")

            # save model + weights
            self.model.save(self.model_path, overwrite=True)
            self.model.save_weights(self.weights_path, overwrite=True)

            print(
                f"✔️ Saved model at epoch {epoch+1} because: "
                + " | ".join(reasons)
            )
        else:
            if current_primary is not None:
                print(
                    f"[SaveOnAnyImprovement] No improvement this epoch. "
                    f"{self.monitor}={float(current_primary):.6f}, "
                    f"best={self.best_primary:.6f}"
                )
            else:
                print(
                    f"[SaveOnAnyImprovement] {self.monitor} not in logs at epoch "
                    f"{epoch+1}, nothing to compare."
                )

class DebugLoss(Callback):
    def on_train_batch_end(self, batch, logs=None):
        loss = logs.get("loss")
        if loss is not None and not np.isfinite(loss):
            print(f"[DebugLoss] Non-finite loss at batch {batch}: {loss}")
        elif batch % 50 == 0:
            print(f"[DebugLoss] batch {batch}, loss={loss:.4f}")

class CustomEval(Callback):
    def __init__(self, val_ds, strategy):
        super().__init__()
        self.val_ds = val_ds
        self.strategy = strategy

    @tf.function
    def _eval_step(self, enc, dec_in, dec_tgt):
        preds = self.model([enc, dec_in], training=False)
        mask = tf.cast(tf.not_equal(dec_tgt, 0), tf.float32)
        acc = tf.keras.metrics.sparse_categorical_accuracy(dec_tgt, preds)
        return tf.reduce_sum(acc * mask), tf.reduce_sum(mask)

    def on_epoch_end(self, epoch, logs=None):
        total_correct = 0.0
        total_count = 0.0
        for (enc, dec_in), dec_tgt in self.val_ds:
            c, n = self._eval_step(enc, dec_in, dec_tgt)
            total_correct += c
            total_count += n
        token_acc = total_correct / total_count
        logs = logs or {}
        logs["val_token_accuracy"] = token_acc
        print(f"Validation token accuracy: {token_acc:.4f}")

def configure_trainable_for_phase(model, phase: str):
    """
    Set layer.trainable flags based on a simple phase name.
    This lets you freeze encoder / decoder / head / refine_* in stages.
    """
    phase = (phase or "all").lower()
    print(f">>> Configuring trainable layers for phase = {phase!r}")

    # default everything to frozen
    for layer in model.layers:
        layer.trainable = False

    if phase == "all":
        # everything can move
        for layer in model.layers:
            layer.trainable = True

    elif phase == "encoder_frozen":
        # freeze encoder (enc_*), train decoder / attn / refine_* / head
        for layer in model.layers:
            name = layer.name
            if name.startswith("enc_"):
                layer.trainable = False
            else:
                layer.trainable = True

    elif phase == "decoder_frozen":
        # train encoder only – keep decoder, refine_* & head frozen
        for layer in model.layers:
            name = layer.name
            if (
                name.startswith("dec_") or
                name.startswith("refine_") or
                name == "decoder_dense"
            ):
                layer.trainable = False
            else:
                layer.trainable = True

    elif phase == "head_only":
        # only train final head + refine block
        for layer in model.layers:
            name = layer.name
            if name == "decoder_dense" or name.startswith("refine_"):
                layer.trainable = True
            else:
                layer.trainable = False
                
    elif phase == "head_plus_synapses":
        trainable_names = [
            "decoder_dense",
            "refine_ffn1", "refine_ffn2",
            "dec_linear_stream",
            "syn_gate1", "syn_gate2",
        ]
        for layer in model.layers:
            layer.trainable = (layer.name in trainable_names)

    elif phase == "global_only":
        # only global encoder/decoder blocks trainable
        for layer in model.layers:
            name = layer.name
            if name.startswith("genc_") or name.startswith("gdec_"):
                layer.trainable = True
                
    elif phase == "syn_linear_only":
        for layer in model.layers:
            name = layer.name
            if name in ["dec_linear_stream", "syn_gate1", "syn_gated1"]:
                layer.trainable = True
            else:
                layer.trainable = False
                
    elif phase == "enc_tf_plus_head_synapses":
        trainable_names = [
            "genc_mha", "genc_ffn1", "genc_ffn2",
            "gdec_mha", "gdec_ffn1", "gdec_ffn2",
            "refine_self_attn",
            
            # new encoder TF
            "enc_tf_proj", "enc_tf_ln1", "enc_tf_mha",
            "enc_tf_res1", "enc_tf_ln2",
            "enc_tf_ffn1", "enc_tf_ffn2",
            "enc_tf_res2", "enc_tf_backproj", "enc_tf_out_res",
            
            # your existing decision bits
            "dec_linear_stream", "dec_linear_enh",
            "lin_tf_proj", "lin_tf_ln1", "lin_tf_mha",
            "lin_tf_res1", "lin_tf_ln2", "lin_tf_ffn1",
            "lin_tf_ffn2", "lin_tf_res2", "lin_tf_backproj", "lin_tf_out_res",
            
            "syn_enc_proj", "syn_dec_proj", "syn_cross_attn",
            "syn_gate1", "syn_gate2",
            "refine_ffn1", "refine_ffn2",
            
            "tf1_proj", "tf1_ln1", "tf1_mha",
            "tf1_res1", "tf1_ln2", "tf1_ffn1", "tf1_ffn2",
            "tf1_res2", "tf1_backproj", "tf1_out_res",
            
            "tf2_proj", "tf2_ln1", "tf2_mha",
            "tf2_res1", "tf2_ln2", "tf2_ffn1", "tf2_ffn2",
            "tf2_res2", "tf2_backproj", "tf2_out_res",
            
            "decoder_dense",
        ]
        for layer in model.layers:
            layer.trainable = (layer.name in trainable_names)

    elif phase == "syn_cross_only":
        for layer in model.layers:
            name = layer.name
            if name in [
                "syn_enc_proj", 
                "syn_dec_proj", 
                "syn_cross_attn",
                "syn_gate2", 
                "syn_gated2",    
                "decoder_dense",      # VERY important
                "refine_ffn1",        # optional but recommended (shape changed)
                "refine_ffn2", 
            ]:
                layer.trainable = True
            else:
                layer.trainable = False

    elif phase == "enc_tf_only":
        trainable_names = [
            "enc_tf_proj", "enc_tf_ln1", "enc_tf_mha",
            "enc_tf_res1", "enc_tf_ln2",
            "enc_tf_ffn1", "enc_tf_ffn2",
            "enc_tf_res2", "enc_tf_backproj", "enc_tf_out_res",
        ]
        for layer in model.layers:
            layer.trainable = (layer.name in trainable_names)

    elif phase == "decoder_plus_syn":
        for layer in model.layers:
            ame = layer.name
            if name.startswith("enc_"):
                layer.trainable = False
            else:
                layer.trainable = True
    
    elif phase == "new_streams_warmup":
        new_layer_names = [
            # new local cross-attn stream
            "cross_attn_local",
            "dec_linear_stream",
            
            # synapse projections and cross-attn
            "syn_enc_proj",
            "syn_dec_proj",
            "syn_cross_attn",
            
            # gates
            "syn_gate1",
            "syn_gate2",
            
            # refine FFN that changed shape
            "refine_ffn1",
            "refine_ffn2",
            
            # final head
            "decoder_dense",
        ]
        for layer in model.layers:
            layer.trainable = (layer.name in new_layer_names)

    else:
        print(f"⚠️ Unknown phase {phase!r}, defaulting to all trainable")
        for layer in model.layers:
            layer.trainable = True

    trainable_count = sum(int(l.trainable) for l in model.layers)
    print(f">>> Layers trainable this phase: {trainable_count} / {len(model.layers)}")
    
def warm_start_from_old_model(model, old_model_path):
    """
    Load an old full .keras model and copy weights layer-by-layer
    into the new model wherever names and shapes match.
    New layers (dec_ln2, refine_*, etc.) will just stay random.
    """
    if not os.path.exists(old_model_path):
        print(f"⚠️ No previous .keras model at {old_model_path}, skipping warm-start.")
        return

    try:
        print(f"Loading old .keras model from {old_model_path} for warm-start...")
        old_model = tf.keras.models.load_model(
            old_model_path, 
            compile=False, 
            safe_mode=False,
        )
    except Exception as e:
        print("⚠️ Could not load old .keras model, skipping warm-start. Reason:", e)
        return

    copied, skipped = 0, 0
    for layer in model.layers:
        try:
            old_layer = old_model.get_layer(layer.name)
        except ValueError:
            # layer with this name didn't exist before
            skipped += 1
            continue

        old_weights = old_layer.get_weights()
        if not old_weights:
            skipped += 1
            continue

        try:
            layer.set_weights(old_weights)
            copied += 1
        except Exception:
            skipped += 1
            continue

    print(f"✅ Warm-start finished: copied weights for {copied} layers, skipped {skipped}.")

def train_model(data_path, epochs=15, batch_size=64, emb_dim=50, train_from_scratch=False, phase="enc_tf_plus_head_synapses"):
    inputs, targets = load_training_data(data_path)
    split = int(0.9 * len(inputs))
    save_dir = "app/models/saved_model"
    tok_in_path = f"{save_dir}/tokenizer_input.json"
    tok_tgt_path = f"{save_dir}/tokenizer_target.json"
    model_path = f"{save_dir}/summarization_model.keras"
    weights_path = f"{save_dir}/summarization_model.weights.h5"

    os.makedirs(save_dir, exist_ok=True)

    # ---------------- Tokenizers ----------------
    train_in, train_tgt = inputs[:split], targets[:split]
    val_in, val_tgt = inputs[split:], targets[split:]
    if os.path.exists(tok_in_path) and os.path.exists(tok_tgt_path):
        tok_in = load_tokenizer(tok_in_path)
        tok_tgt = load_tokenizer(tok_tgt_path)
    else:
        tok_in = create_tokenizer(inputs, max_words=MAX_VOCAB)
        tok_tgt = create_tokenizer(targets, max_words=MAX_VOCAB)

    vs_in = min(len(tok_in.word_index) + 1, MAX_VOCAB + 1)
    vs_tgt = min(len(tok_tgt.word_index) + 1, MAX_VOCAB + 1)

    # ---------------- Preprocess ----------------
    train_enc = preprocess_texts(train_in, tok_in, max_length_input, vs_in)
    train_dec = preprocess_texts(train_tgt, tok_tgt, max_length_target, vs_tgt)
    train_dec_in, train_dec_tgt = prepare_decoder_sequences(train_dec)
    
    print("Train_dec_tgt min / max:", train_dec_tgt.min(), train_dec_tgt.max())
    print("Vocab size (vs_tgt):", vs_tgt)


    val_enc = preprocess_texts(val_in, tok_in, max_length_input, vs_in)
    val_dec = preprocess_texts(val_tgt, tok_tgt, max_length_target, vs_tgt)
    val_dec_in, val_dec_tgt = prepare_decoder_sequences(val_dec)
    print("Val_dec_tgt min / max:", val_dec_tgt.min(), val_dec_tgt.max())

    num_train = len(train_enc)

    #  cap steps/epoch so Kaggle doesn't take 3h
    MAX_STEPS_PER_EPOCH = 1500  # you can drop to 1000 if still too slow
    steps_per_epoch = min(
        MAX_STEPS_PER_EPOCH,
        max(1, num_train // batch_size),
    )

    train_ds = (
        tf.data.Dataset.from_tensor_slices(((train_enc, train_dec_in), train_dec_tgt))
        .shuffle(buffer_size=num_train, seed=42)
        .batch(batch_size, drop_remainder=False)
        .repeat()
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(((val_enc, val_dec_in), val_dec_tgt))
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_steps = max(
        1,
        len(val_enc) // batch_size + (1 if len(val_enc) % batch_size else 0),
    )

    n_rouge = 8
    rouge_ds = (
        tf.data.Dataset.from_tensor_slices(((val_enc, val_dec_in), val_dec_tgt))
        .shuffle(len(val_enc))
        .take(n_rouge)
        .cache()
        .batch(n_rouge, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    if USE_MULTI_GPU:
        strategy = tf.distribute.MirroredStrategy()
        print(">>> Using MirroredStrategy with",
              strategy.num_replicas_in_sync, "replicas")
    else:
        strategy = tf.distribute.get_strategy()
        print(">>> Using default (single GPU / CPU) strategy")
        
    with strategy.scope():
        # -------- Build model first --------
        model = build_seq2seq_model(
            vs_in,
            vs_tgt,
            emb_dim,
            max_length_input,
            max_length_target,
        )

        # -------- Optional warm-start from weights --------
        if not train_from_scratch:
            try:
                model.load_weights(weights_path, skip_mismatch=True)
                print("Weights loaded (with skip_mismatch=True).")
                print(f"Loaded weights from {weights_path}")
            except Exception as e:
                print("Direct load_weights failed, trying warm-start from .keras:", e)
                warm_start_from_old_model(model, model_path)
        else:
            warm_start_from_old_model(model, model_path)
            print("train_from_scratch=True → starting from random init (warm-start if possible).")

        # -------- Optimizer: smaller LR + gradient clipping --------
        configure_trainable_for_phase(model, phase)

        base_opt = Adam(
            learning_rate=1e-3,
            global_clipnorm=1.0,  # gradient clipping
        )
        opt = base_opt

        model.compile(
            optimizer=opt,
            loss=masked_sparse_ce,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
        )

        # quick sanity check
        (enc_batch, dec_in_batch), dec_tgt_batch = next(iter(train_ds))
        logits = model([enc_batch, dec_in_batch], training=False)
        print(">>> Using loss object:", model.loss)
        print(
            "Output logits stats:",
            tf.reduce_min(logits).numpy(),
            tf.reduce_max(logits).numpy(),
            "any NaN?",
            tf.reduce_any(tf.math.is_nan(logits)).numpy(),
        )
        print(">>> Global policy:", tf.keras.mixed_precision.global_policy().name)
        print(">>> Optimizer class:", type(model.optimizer).__name__)
        debug_cb = DebugLoss()

        snap_cb = SnapshotCallback(save_dir="app/models/saved_model/plots", interval_epochs=10)
        rouge_cb = RougeCallback(
            val_ds=rouge_ds,
            tgt_tokenizer=tok_tgt,
            max_length_target=max_length_target,
            n_samples=n_rouge,
        )

        save_cb = SaveOnAnyImprovement(
            model_path=model_path,
            weights_path=weights_path,
            monitor="val_token_accuracy",
            mode="max",          # accuracy should increase
            min_acc_gain=0.0005,  # how much better acc must be
            max_loss_increase=0.10,
        )


        callbacks = [
            rouge_cb,
            EarlyStopping(
                monitor="val_token_accuracy",
                mode="max",
                patience=8,
                restore_best_weights=True,
            ),
            save_cb,
            TerminateOnNaN(),
            
        ]

        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
            initial_epoch=0, 
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=val_steps,
        )

    # -------- Save tokenizers + plots --------
    with open(tok_in_path, "w", encoding="utf-8") as f:
        f.write(tok_in.to_json())
        f.flush()
        os.fsync(f.fileno())
    with open(tok_tgt_path, "w", encoding="utf-8") as f:
        f.write(tok_tgt.to_json())
        f.flush()
        os.fsync(f.fileno())

    plot_history(history, os.path.dirname(model_path))
    return model



if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.set_memory_growth(gpu, True)
            except AttributeError:
                tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus, "GPU")

    print("Physical GPUs:", gpus)
    print("Logical GPUs:", tf.config.list_logical_devices("GPU"))

    print("TensorFlow version:", tf.__version__)
    with tf.device("/GPU:0" if gpus else "/CPU:0"):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print("Operation result shape:", c.shape)
    os.system("nvidia-smi")

    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "app/models/data/text/training_data.json"
    model = train_model(data_path)
    print("Training complete.")
    print("Model saved to:", "app/models/saved_model/summarization_model.keras")
    print("Input tokenizer saved to:", "app/models/saved_model/tokenizer_input.json")
