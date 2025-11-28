import json
import os
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
)

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



max_length_input = 256
max_length_target = 128
# Desired task ratios for multi-task training (only used if the data has "task")
TASK_RATIOS = {
    "summarization": 0.5,
    "code_cpp": 0.2,
    "math": 0.3,
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

    # Self-attention on the decoder outputs: decoder → decoder
    self_attn = Attention(name="self_attn")([dec_out2, dec_out2])
    
    # but KEEP the last dim = dec_units (no shape change)
    fused = tf.keras.layers.Add(name="decoder_fused")(
        [dec_out2, cross_attn, self_attn]
    )

    dec_norm = LayerNormalization(name="dec_ln")(fused)
    dec_ffn = Dense(dec_units * 4, activation="relu", name="dec_ffn1")(dec_norm)
    dec_ffn = Dense(dec_units, name="dec_ffn2")(dec_ffn)
    dec_context = Add(name="dec_ffn_res")([fused, dec_ffn])
    
    outputs = Dense(
        vocab_tgt,
        activation=None,
        name="decoder_dense",
        dtype="float32",  # keep float32 with mixed precision
    )(dec_context)

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
        for t in range(self.max_length):
            pad_amt = self.max_length - tf.shape(dec_input)[1]
            dec_padded = tf.pad(dec_input, [[0, 0], [0, pad_amt]])
            logits = self.model([enc_batch, dec_padded], training=False)
            next_token = tf.cast(tf.argmax(logits[:, t, :], axis=-1), tf.int32)
            dec_input = tf.concat([dec_input, next_token[:, None]], axis=1)
            result = tf.concat([result, next_token[:, None]], axis=1)

        count = 0
        for ref_seq, pred_seq in zip(dec_tgt_batch.numpy(), result.numpy()):
            ref_txt = " ".join(
                self.tokenizer.index_word[w] for w in ref_seq if w not in (0, self.start_id, self.end_id)
            )
            pred_txt = " ".join(
                self.tokenizer.index_word.get(w, "") for w in pred_seq if w not in (0, self.start_id, self.end_id)
            )
            sc = self.scorer.score(ref_txt, pred_txt)
            total["rouge1"] += sc["rouge1"].fmeasure
            total["rouge2"] += sc["rouge2"].fmeasure
            total["rougeL"] += sc["rougeL"].fmeasure
            count += 1
        avg = {k: (total[k] / count if count else 0.0) for k in total}
        logs.update({f"val_{k}": v for k, v in avg.items()})


class SaveOnAnyImprovement(tf.keras.callbacks.Callback):
    def __init__(self, model_path, weights_path, monitor="val_rouge1"):
        super().__init__()
        self.model_path = model_path       # full .keras model
        self.weights_path = weights_path   # weights .h5
        self.monitor = monitor
        self.best_loss = float("inf")
        self.best_acc = 0.0
        self.best_rouges = [0.0, 0.0, 0.0]  # [rouge1, rouge2, rougeL]

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        new_loss = logs.get("val_loss")
        new_acc = logs.get("val_token_accuracy")
        new_rouges = [logs.get(f"val_rouge{i}") for i in [1, 2, "L"]]

        should_save = False
        save_reasons = []

        # Primary: ROUGE
        if new_rouges[0] is not None and new_rouges[0] > self.best_rouges[0]:
            should_save = True
            save_reasons.append(f"val_rouge1 ↑ {new_rouges[0]:.4f}")
        if new_rouges[2] is not None and new_rouges[2] > self.best_rouges[2]:
            should_save = True
            save_reasons.append(f"val_rougeL ↑ {new_rouges[2]:.4f}")

        # Secondary: token accuracy with loss constraint
        if (
            new_acc is not None
            and new_loss is not None
            and new_acc > self.best_acc + 0.001
            and new_loss < self.best_loss * 1.1
        ):
            should_save = True
            save_reasons.append(f"val_token_accuracy ↑ {new_acc:.4f}")

        if should_save or not os.path.exists(self.weights_path):
            # Save full model + weights
            self.model.save(self.model_path, overwrite=True)
            self.model.save_weights(self.weights_path, overwrite=True)
            print(f"✔️ Saved model at epoch {epoch + 1} because {', '.join(save_reasons)}")

        # Update best metrics
        if new_loss is not None:
            self.best_loss = min(self.best_loss, new_loss)
        if new_acc is not None:
            self.best_acc = max(self.best_acc, new_acc)
        self.best_rouges = [
            max(nr if nr is not None else 0.0, br)
            for nr, br in zip(new_rouges, self.best_rouges)
        ]



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


def train_model(data_path, epochs=2, batch_size=32, emb_dim=50, train_from_scratch=False):
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

    val_enc = preprocess_texts(val_in, tok_in, max_length_input, vs_in)
    val_dec = preprocess_texts(val_tgt, tok_tgt, max_length_target, vs_tgt)
    val_dec_in, val_dec_tgt = prepare_decoder_sequences(val_dec)

    num_train = len(train_enc)

    #  cap steps/epoch so Kaggle doesn't take 3h
    MAX_STEPS_PER_EPOCH = 2000  # you can drop to 1000 if still too slow
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

    strategy = tf.distribute.MirroredStrategy()
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
        if (not train_from_scratch) and os.path.exists(weights_path):
            print("Warm-start: loading previous weights from", weights_path)
            try:
                model.load_weights(weights_path)
                print("✅ Successfully loaded previous weights.")
            except Exception as e:
                print("⚠️ Could not load previous weights, training from scratch instead. Reason:", e)
        else:
            if train_from_scratch:
                print("train_from_scratch=True → starting from random init.")
            else:
                print("No previous weights file found → starting from random init.")

        # -------- Optimizer: smaller LR + gradient clipping --------


        base_opt = Adam(
            learning_rate=1e-5, 
            global_clipnorm=1.0, # keep gradient clipping
        )
        opt = base_opt

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
        )

        # quick sanity check
        (enc_batch, dec_in_batch), dec_tgt_batch = next(iter(train_ds))
        logits = model([enc_batch, dec_in_batch], training=False)
        print(
            "Output logits stats:",
            tf.reduce_min(logits).numpy(),
            tf.reduce_max(logits).numpy(),
            "any NaN?",
            tf.reduce_any(tf.math.is_nan(logits)).numpy(),
        )
        print(">>> Global policy:", tf.keras.mixed_precision.global_policy().name)
        print(">>> Optimizer class:", type(model.optimizer).__name__)

        snap_cb = SnapshotCallback(save_dir="app/models/saved_model/plots", interval_epochs=10)
        rouge_cb = RougeCallback(
            val_ds=rouge_ds,
            tgt_tokenizer=tok_tgt,
            max_length_target=max_length_target,
            n_samples=n_rouge,
        )

        save_cb = SaveOnAnyImprovement(model_path, weights_path)

        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(
                monitor="val_token_accuracy",
                mode="max",
                patience=5,
                restore_best_weights=True,
            ),
            save_cb,
        ]

        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
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
