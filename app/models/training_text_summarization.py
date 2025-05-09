import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import tensorflow as tf

# 1) Discover and configure GPUs before any TF ops
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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam

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
    tok.num_words = len(tok.word_index) + 1
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
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0005,
        decay_steps=20000,
        decay_rate=0.98,
        staircase=True
    )
    model.compile(Adam(lr_sched), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(hist, save_dir):
    epochs = range(1, len(hist.history['loss']) + 1)
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, hist.history['loss'], 'bo-', label='Loss')
    plt.plot(epochs, hist.history['accuracy'], 'go-', label='Accuracy')
    plt.title('Training Metrics')
    plt.xlabel('Epochs')
    plt.legend()
    out_path = os.path.join(save_dir, "training_progress.png")
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

class CustomEval(Callback):
    def __init__(self, val_ds):
        super().__init__()
        self.val_ds = val_ds
    def on_epoch_end(self, epoch, logs=None):
        total, correct = 0, 0
        for (enc_in, dec_in), dec_tgt in self.val_ds:
            preds = self.model.predict([enc_in, dec_in])
            idx = np.argmax(preds, axis=-1)
            mask = dec_tgt != 0
            correct += np.sum((idx == dec_tgt) & mask)
            total += np.sum(mask)
        print(f"Validation Token Accuracy: {correct/total:.4f}")

def train_model(data_path, epochs=10, batch_size=128, emb_dim=50):
    inputs, targets = load_training_data(data_path)
    
    strategy = tf.distribute.MirroredStrategy()
    split = int(0.9 * len(inputs))
    train_in, train_tgt = inputs[:split], targets[:split]
    val_in, val_tgt = inputs[split:], targets[split:]


    save_dir     = "app/models/saved_model"
    tok_in_path  = f"{save_dir}/tokenizer_input.json"
    tok_tgt_path = f"{save_dir}/tokenizer_target.json"
    model_path   = f"{save_dir}/summarization_model.keras"
    
    os.makedirs(save_dir, exist_ok=True)

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

    vs_in  = len(tok_in.word_index) + 1
    vs_tgt = len(tok_tgt.word_index) + 1

    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    train_enc = preprocess_texts(train_in, tok_in, max_length_input)
    train_dec = preprocess_texts(train_tgt, tok_tgt, max_length_target)
    train_dec_in, train_dec_tgt = prepare_decoder_sequences(train_dec)

    val_enc = preprocess_texts(val_in, tok_in, max_length_input)
    val_dec = preprocess_texts(val_tgt, tok_tgt, max_length_target)
    val_dec_in, val_dec_tgt = prepare_decoder_sequences(val_dec)


    train_ds = (tf.data.Dataset.from_tensor_slices(((train_enc, train_dec_in), train_dec_tgt))
                .shuffle(1000)
                .cache()  
                .batch(batch_size)
                .map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(tf.data.AUTOTUNE))
    val_ds = (
        tf.data.Dataset
          .from_tensor_slices(((val_enc, val_dec_in), val_dec_tgt))
          .cache()                      # <-- cache validation too
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE)
    )

    with strategy.scope():
        if os.path.exists(model_path) \
           and os.path.exists(tok_in_path) \
           and os.path.exists(tok_tgt_path):
           print("Loading model from disk")
           model = load_model(model_path)
        else:
            model = build_seq2seq_model(vs_in, vs_tgt, emb_dim,
                                    max_length_input, max_length_target)
    callbacks = [
        EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, verbose=1),
        CustomEval(val_ds)
    ]

    # 1) Create a MirroredStrategy for multi-GPU
    strategy = tf.distribute.MirroredStrategy()

    # 2) Build/load + compile + fit all inside strategy.scope()
    with strategy.scope():
        # a) Load existing or build new model
        if os.path.exists(model_path):
            print("Loading model from disk")
            model = load_model(model_path)
            # loaded model keeps its compile settings
        else:
            model = build_seq2seq_model(
                vs_in, vs_tgt, emb_dim,
                max_length_input, max_length_target
            )

        # b) (Re)define callbacks inside scope
        callbacks = [
            EarlyStopping(monitor='loss', patience=3, restore_best_weights=True),
            ModelCheckpoint(model_path, save_best_only=True, verbose=1),
            CustomEval(val_ds)
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
    with open(tok_tgt_path, 'w', encoding='utf-8') as f:
        f.write(tok_tgt.to_json())

    plot_history(history, os.path.dirname(model_path))
    return model

if __name__ == "__main__":
    model = train_model("app/models/data/text/training_data.json")
    print("Training complete.")