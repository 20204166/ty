import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth on GPUs:", gpus)
    except RuntimeError as e:
        print("Error setting GPU memory growth:", e)
else:
    print("No GPUs found; using CPU.")

# 2) Enable XLA now that memory growth is set
tf.config.optimizer.set_jit(True)

# 3) Safe to do other TF operations
print("TensorFlow version:", tf.__version__)
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])
    c = tf.matmul(a, b)
print("Operation result shape:", c.shape)
os.system("nvidia-smi")

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Attention, LSTMCell
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model   
from rouge_score import rouge_scorer
import psutil
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

max_length_input = 50
max_length_target = 20

def load_training_data(data_path: str):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data or not isinstance(data[0], dict):
        raise ValueError("Training data must be a non-empty list of objects.")
    if "article" in data[0] and "highlights" in data[0]:
        inputs = [item["article"] for item in data]
        targets = [item["highlights"] for item in data]
    elif "text" in data[0] and "summary" in data[0]:
        inputs = [item["text"] for item in data]
        targets = [item["summary"] for item in data]
    else:
        raise ValueError("Training data must contain 'article'/'highlights' or 'text'/'summary'.")
    targets = [f"<start> {t} <end>" for t in targets]
    return inputs, targets

def create_tokenizer(texts, oov_token="<OOV>"):
    tok = Tokenizer(oov_token=oov_token)
    tok.fit_on_texts(texts)
    return tok

def load_tokenizer(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

def preprocess_texts(texts, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding='post')

def prepare_decoder_sequences(sequences):
    dec_in = sequences[:, :-1]
    dec_tgt = sequences[:, 1:]
    pad_width = max_length_target - dec_in.shape[1]
    if pad_width > 0:
        dec_in = np.pad(dec_in, ((0, 0), (0, pad_width)), mode='constant')
        dec_tgt = np.pad(dec_tgt, ((0, 0), (0, pad_width)), mode='constant')
    return dec_in, dec_tgt

def build_seq2seq_model(vocab_in, vocab_tgt, emb_dim, max_in, max_tgt):
    enc_inputs = Input(shape=(max_in,), name="enc_inputs")
    enc_emb = Embedding(vocab_in, emb_dim, name="enc_emb")(enc_inputs)
    enc_cell1 = LSTMCell(64, name="enc_cell1")
    enc_rnn1 = tf.keras.layers.RNN(enc_cell1, return_sequences=True, return_state=True, name="enc_rnn1")
    out1, h1, c1 = enc_rnn1(enc_emb)
    enc_cell2 = LSTMCell(64, name="enc_cell2")
    enc_rnn2 = tf.keras.layers.RNN(enc_cell2, return_sequences=True, return_state=True, name="enc_rnn2")
    enc_outs, h2, c2 = enc_rnn2(out1)
    enc_states = [h2, c2]

    dec_inputs = Input(shape=(max_tgt,), name="dec_inputs")
    dec_emb = Embedding(vocab_tgt, emb_dim, name="dec_emb")(dec_inputs)
    dec_cell1 = LSTMCell(64, name="dec_cell1")
    dec_rnn1 = tf.keras.layers.RNN(dec_cell1, return_sequences=True, return_state=True, name="dec_rnn1")
    dec_out1, _, _ = dec_rnn1(dec_emb, initial_state=enc_states)
    dec_cell2 = LSTMCell(64, name="dec_cell2")
    dec_rnn2 = tf.keras.layers.RNN(dec_cell2, return_sequences=True, return_state=True, name="dec_rnn2")
    dec_out2, _, _ = dec_rnn2(dec_out1)

    attn = Attention(name="attn_layer")([dec_out2, enc_outs])
    concat = Concatenate(name="concat_layer")([attn, dec_out2])
    outputs = Dense(vocab_tgt, activation='softmax', name="decoder_dense")(concat)

    model = Model([enc_inputs, dec_inputs], outputs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=20000,
        decay_rate=0.98,
        staircase=True
    )
    model.compile(
        Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    return model

def plot_history(hist, save_dir):
    import os
    import matplotlib.pyplot as plt

    epochs = range(1, len(hist.history['loss']) + 1)

    # ── 1) Loss curve ──
    plt.figure()
    plt.plot(epochs, hist.history['loss'],     label='Train loss')
    plt.plot(epochs, hist.history['val_loss'], label='Val loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(loss_path)
    plt.close()

    # ── 2) Accuracy curve ──
    plt.figure()
    plt.plot(epochs, hist.history['accuracy'],     label='Train acc')
    plt.plot(epochs, hist.history['val_accuracy'], label='Val acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    acc_path = os.path.join(save_dir, 'accuracy_curve.png')
    plt.savefig(acc_path)
    plt.close()

    # ── 3) Token‐accuracy curve ──
    if 'val_token_acc' in hist.history:
        plt.figure()
        plt.plot(epochs, hist.history['val_token_acc'], label='Val token acc')
        plt.xlabel('Epoch'); plt.ylabel('Token Accuracy')
        plt.title('Validation Token Accuracy')
        plt.legend()
        tok_path = os.path.join(save_dir, 'token_accuracy_curve.png')
        plt.savefig(tok_path)
        plt.close()

    # ── 4) ROUGE curves ──
    if 'val_rouge1' in hist.history:
        plt.figure()
        plt.plot(epochs, hist.history['val_rouge1'], label='ROUGE-1 F1')
        plt.plot(epochs, hist.history['val_rouge2'], label='ROUGE-2 F1')
        plt.plot(epochs, hist.history['val_rougeL'], label='ROUGE-L F1')
        plt.xlabel('Epoch'); plt.ylabel('F1 Score')
        plt.title('Validation ROUGE Scores')
        plt.legend()
        rouge_path = os.path.join(save_dir, 'rouge_curve.png')
        plt.savefig(rouge_path)
        plt.close()

    print("Saved plots to", save_dir,
          os.path.basename(loss_path),
          os.path.basename(acc_path),
          *(os.path.basename(tok_path) if 'tok_path' in locals() else []),
          *(os.path.basename(rouge_path) if 'rouge_path' in locals() else []))

class RougeCallback(Callback):
    def __init__(self, val_ds, tgt_tokenizer, max_length_target, n_samples=100):
        super().__init__()
        self.val_ds     = val_ds
        self.tokenizer  = tgt_tokenizer
        self.start_id   = tgt_tokenizer.word_index.get('<start>', tgt_tokenizer.word_index.get(tgt_tokenizer.oov_token))
        self.end_id     = tgt_tokenizer.word_index.get('<end>',   tgt_tokenizer.word_index.get(tgt_tokenizer.oov_token))
        self.max_length = max_length_target
        self.scorer     = rouge_scorer.RougeScorer(
            ['rouge1','rouge2','rougeL'], use_stemmer=True
        )
        self.n_samples  = n_samples

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        total = {'rouge1':0, 'rouge2':0, 'rougeL':0}
        count = 0
        

        for (enc, _), dec_tgt in self.val_ds.take(self.n_samples):
            # --- greedy decode to `preds` exactly as before ---
            batch     = tf.shape(enc)[0]
            dec_input = tf.fill([batch,1], self.start_id)
            result    = tf.zeros([batch, 0], dtype=tf.int32)
            for t in range(self.max_length):
                pad_amt    = self.max_length - tf.shape(dec_input)[1]
                dec_padded = tf.pad(dec_input, [[0,0],[0,pad_amt]], constant_values=0)

    
                logits     = self.model([enc, dec_padded], training=False)
                next_token = tf.cast(tf.argmax(logits[:, t, :], axis=-1), tf.int32)
                dec_input  = tf.concat([dec_input, next_token[:,None]], axis=1)
                result     = tf.concat([result,    next_token[:,None]], axis=1)
            # --- stringify and score ---
            for ref_seq, pred_seq in zip(dec_tgt.numpy(), result.numpy()):
                # strip pad/<start>/<end>
                def clean(seq):
                    return [w for w in seq if w>0
                            and w != self.start_id
                            and w != self.end_id]
                ref_txt  = " ".join(self.tokenizer.index_word[w] for w in clean(ref_seq))
                pred_txt = " ".join(self.tokenizer.index_word.get(w,"") for w in clean(pred_seq))

                sc = self.scorer.score(ref_txt, pred_txt)
                for k in total:
                    total[k] += sc[k].fmeasure
                count += 1

        # compute averages
        avg = {k: float(v/count) for k,v in total.items()}
        
        # **inject into logs** so SaveOnAnyImprovement can see them
        logs['val_rouge1'] = avg['rouge1']
        logs['val_rouge2'] = avg['rouge2']
        logs['val_rougeL'] = avg['rougeL']

        # and print for yourself
        print(f"Epoch {epoch+1}: " +
              f"ROUGE-1 {avg['rouge1']:.4f}, " +
              f"ROUGE-2 {avg['rouge2']:.4f}, " +
              f"ROUGE-L {avg['rougeL']:.4f}")

class ResourceMonitor(Callback):
    def __init__(self, total_epochs, save_dir):
        super().__init__()
        self.total_epochs = total_epochs
        self.save_dir     = save_dir
        self.cpu_usage    = []
        self.ram_usage    = []
        
        self.gpu_usage    = {}  

    def _get_gpu_utils(self):
        """Returns a list of floats, one per GPU device."""
        try:
            raw = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits"
            ])
            lines = raw.decode("utf-8").strip().splitlines()
            # parse each line into a float
            vals  = [float(l) for l in lines if l.strip()]
            return vals
        except Exception:
            return []

    def on_epoch_end(self, epoch, logs=None):
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        self.cpu_usage.append(cpu)
        self.ram_usage.append(ram)

        gpu_vals = self._get_gpu_utils()
        # record each GPU’s util; initialize lists if first epoch
        for i, u in enumerate(gpu_vals):
            self.gpu_usage.setdefault(i, []).append(u)

        # for display, you might print per-GPU or an average:
        if gpu_vals:
            avg_gpu = sum(gpu_vals) / len(gpu_vals)
            gpu_str = ", ".join(f"GPU{i}={u:.1f}%" for i,u in enumerate(gpu_vals))
        else:
            avg_gpu = 0.0
            gpu_str = "no-GPU"
        print(f"Epoch {epoch+1}: CPU {cpu:.1f}%  RAM {ram:.1f}%  {gpu_str}")

        # only plot & save at the final epoch
        if (epoch + 1) == self.total_epochs:
            fig, axes = plt.subplots(2 + len(gpu_vals), 1, figsize=(8, 4*(2+len(gpu_vals))))
            
            # CPU
            axes[0].plot(range(1, self.total_epochs+1), self.cpu_usage, marker='o')
            axes[0].set_title("CPU Usage (%)")
            axes[0].set_ylabel("CPU %")
            
            # RAM
            axes[1].plot(range(1, self.total_epochs+1), self.ram_usage, marker='o')
            axes[1].set_title("RAM Usage (%)")
            axes[1].set_ylabel("RAM %")

            # one subplot per GPU
            for i in range(len(gpu_vals)):
                axes[2+i].plot(range(1, self.total_epochs+1),
                               self.gpu_usage[i], marker='o')
                axes[2+i].set_title(f"GPU {i} Usage (%)")
                axes[2+i].set_ylabel("GPU %")
            
            for ax in axes:
                ax.set_xlabel("Epoch")
            plt.tight_layout()
            out_path = os.path.join(self.save_dir, "resource_usage.png")
            plt.savefig(out_path)
            print(f"Saved resource usage plot to {out_path}")
            plt.close(fig)

class SaveOnAnyImprovement(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.best = {}
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        any_improved = False
        improvements = []

        for name, value in logs.items():
            if not name.startswith("val_rouge1"):
                continue
            # smaller-is-better for losses, larger-is-better otherwise
            is_loss = name.endswith("loss")
            best_val = self.best.get(name, np.Inf if is_loss else -np.Inf)

            if (is_loss and value < best_val) or (not is_loss and value > best_val):
                self.best[name] = value
                any_improved = True
                arrow = "↓" if is_loss else "↑"
                improvements.append(f"{name} {arrow} {value:.4f}")

        if any_improved:
            self.model.save(self.filepath)
            print(f"✔️ Saved model at epoch {epoch+1} because " +
                  ", ".join(improvements))

class CustomEval(tf.keras.callbacks.Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
        self.metric = tf.keras.metrics.SparseCategoricalAccuracy(name="val_token_acc")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        train_loss = logs.get("loss")
        train_acc  = logs.get("accuracy")
        val_loss   = logs.get("val_loss")
        val_acc    = logs.get("val_accuracy")
        print(
            f"Epoch {epoch+1:>2} — "
            f"loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )
        self.metric.reset_state()     
        for (enc_in, dec_in), dec_tgt in self.val_ds:
            preds = self.model([enc_in, dec_in], training=False)
            mask = tf.cast(tf.not_equal(dec_tgt, 0), tf.float32)
            self.metric.update_state(dec_tgt, preds, sample_weight=mask)

        acc = self.metric.result().numpy()
        print(f"Validation Token Accuracy: {acc:.4f}")
        logs['val_token_acc'] = acc

def train_model(data_path, epochs=25, batch_size=256, emb_dim=50, train_from_scratch = True):
    inputs, targets = load_training_data(data_path)
    split = int(0.9 * len(inputs))
    save_dir     = "app/models/saved_model"
    tok_in_path  = f"{save_dir}/tokenizer_input.json"
    tok_tgt_path = f"{save_dir}/tokenizer_target.json"
    model_path   = f"{save_dir}/summarization_model.keras"
    
    os.makedirs(save_dir, exist_ok=True)
    

    split = int(0.9 * len(inputs))
    train_in, train_tgt = inputs[:split], targets[:split]
    val_in, val_tgt = inputs[split:], targets[split:]

    


    if os.path.exists(tok_in_path) and os.path.exists(tok_tgt_path):
        tok_in  = load_tokenizer(tok_in_path)
        tok_tgt = load_tokenizer(tok_tgt_path)
    else:
        tok_in  = create_tokenizer(inputs)
        tok_tgt = create_tokenizer(targets)
        with open(tok_in_path, 'w', encoding='utf-8') as f:
            f.write(tok_in.to_json())
        with open(tok_tgt_path, 'w', encoding='utf-8') as f:
            f.write(tok_tgt.to_json())

    vs_in = len(tok_in.word_index) + 1
    vs_tgt = len(tok_tgt.word_index) + 1

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')


    train_enc = preprocess_texts(train_in, tok_in, max_length_input)
    train_dec = preprocess_texts(train_tgt, tok_tgt, max_length_target)
    train_dec_in, train_dec_tgt = prepare_decoder_sequences(train_dec)

    val_enc = preprocess_texts(val_in, tok_in, max_length_input)
    val_dec = preprocess_texts(val_tgt, tok_tgt, max_length_target)
    val_dec_in, val_dec_tgt = prepare_decoder_sequences(val_dec)

    train_ds = (
        tf.data.Dataset
          .from_tensor_slices(((train_enc, train_dec_in), train_dec_tgt))
          .shuffle(1000)
          .cache()
          .batch(batch_size)
          .map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
          .prefetch(tf.data.AUTOTUNE)

    )

    val_ds = (
        tf.data.Dataset
          .from_tensor_slices(((val_enc, val_dec_in), val_dec_tgt))
          .cache()
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
    )

    strategy = tf.distribute.MirroredStrategy()
    
    with strategy.scope():
        if (not train_from_scratch) and os.path.exists(model_path):
            print("Loading model from disk")
            model = load_model(model_path)
        else:
            model = build_seq2seq_model(
                vs_in, vs_tgt, emb_dim,
                max_length_input, max_length_target
            )

        total_epochs = epochs
        resource_cb = ResourceMonitor(total_epochs, save_dir)
        
        rouge_cb = RougeCallback(val_ds, tok_tgt, max_length_target, n_samples=50)
        save_cb  = SaveOnAnyImprovement(model_path)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1
            )
        # b) (Re)define callbacks inside scope
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                patience=3,
                restore_best_weights=True,
                verbose=1
                ),
                rouge_cb,
                save_cb,
                CustomEval(val_ds),
                resource_cb,
                reduce_lr
        ]


        # c) Train—this will now shard batches across your GPUs
        history = model.fit(
            train_ds,
            epochs=epochs,
            verbose=2,
            callbacks=callbacks,
            validation_data=val_ds
        )

    # after training: save tokenizers, plot history, return model
    with open(tok_in_path, 'w', encoding='utf-8') as f:
        f.write(tok_in.to_json())
        f.flush()                # push Python buffer to OS
        os.fsync(f.fileno()) 
    with open(tok_tgt_path, 'w', encoding='utf-8') as f:
        f.write(tok_tgt.to_json())
        f.flush()                # push Python buffer to OS
        os.fsync(f.fileno()) 

    plot_history(history, os.path.dirname(model_path))
    return model

if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "app/models/data/text/training_data.json"
    model = train_model(data_path)
    print("Training complete.")
    print("Model saved to:", "app/models/saved_model/summarization_model.keras")
    print("Input tokenizer saved to:", "app/models/saved_model/tokenizer_input.json")   