import os
import random
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

# Config
BASE_PATH = "physionet_data"
MAX_TIME_STEPS = 556
SAMPLE_RATE = 2000
NUM_MELS = 64  # Number of Mel bands
FRAME_LENGTH = 256
FRAME_STEP = 128

def augment_audio(audio):
    noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
    audio = audio + noise_amp * np.random.normal(size=audio.shape)

    rate = np.random.uniform(0.9, 1.1)
    audio_len = len(audio)
    indices = np.round(np.arange(0, audio_len, rate))
    indices = indices[indices < audio_len].astype(int)
    audio = audio[indices]

    return audio

def process_audio(file_path, augment=False):
    try:
        _, audio = wavfile.read(file_path)
        audio = audio.astype(np.float32)

        if augment:
            audio = augment_audio(audio)

        p99 = np.percentile(np.abs(audio), 99)
        audio = np.clip(audio / (p99 + 1e-6), -1, 1)

        spectrogram = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, pad_end=True)
        spectrogram = tf.abs(spectrogram) + 1e-6

        mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=NUM_MELS,
            num_spectrogram_bins=FRAME_LENGTH // 2 + 1,
            sample_rate=SAMPLE_RATE,
            lower_edge_hertz=20,
            upper_edge_hertz=500
        )
        mel_spectrogram = tf.tensordot(spectrogram, mel_filterbank, 1)
        mel_spectrogram = tf.math.log(mel_spectrogram)

        mean, var = tf.nn.moments(mel_spectrogram, axes=[0,1])
        mel_spectrogram = (mel_spectrogram - mean) / tf.sqrt(var + 1e-6)

        if mel_spectrogram.shape[0] < MAX_TIME_STEPS:
            mel_spectrogram = tf.pad(mel_spectrogram, [[0, MAX_TIME_STEPS - mel_spectrogram.shape[0]], [0, 0]])
        else:
            mel_spectrogram = mel_spectrogram[:MAX_TIME_STEPS]

        return tf.expand_dims(mel_spectrogram, -1)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def load_list(filename):
    with open(os.path.join(BASE_PATH, filename), 'r') as f:
        return [line.strip() + ".wav" for line in f.readlines()]

def prepare_dataset(normal_list, abnormal_list, augment=False):
    # Balance lengths
    min_len = min(len(normal_list), len(abnormal_list))
    normal_list = normal_list[:min_len]
    abnormal_list = abnormal_list[:min_len]

    all_files = [(f, 0) for f in normal_list] + [(f, 1) for f in abnormal_list]
    random.shuffle(all_files)

    X, y = [], []
    for fname, label in all_files:
        spec = process_audio(os.path.join(BASE_PATH, fname), augment=augment)
        if spec is not None:
            X.append(spec)
            y.append(label)

    if len(X) == 0:
        raise ValueError("No valid audio files processed!")

    X_stacked = tf.stack(X)
    y_tensor = tf.convert_to_tensor(y, dtype=tf.float32)

    print(f"prepare_dataset: X shape = {X_stacked.shape}, y shape = {y_tensor.shape}")
    return X_stacked, y_tensor

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_TIME_STEPS, NUM_MELS, 1)),
        tf.keras.layers.Conv2D(16, (5,5), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', from_logits=False)]
    )
    print(f"Model input shape: {model.input_shape}")
    return model

def train():
    normal_files = load_list("RECORDS-normal")
    abnormal_files = load_list("RECORDS-abnormal")

    # You can also add b-files if you have them:
    bnormal_files = load_list("bnormal")
    babnormal_files = load_list("babnormal")
    normal_files += bnormal_files
    abnormal_files += babnormal_files
    # Log total files before balancing
    print("📊 Dataset Summary BEFORE balancing:")
    print(f"Total NORMAL files   : {len(normal_files)}")
    print(f"Total ABNORMAL files : {len(abnormal_files)}")


    X, y = prepare_dataset(normal_files, abnormal_files, augment=False)

    # Shuffle dataset
    indices = tf.range(len(X))
    indices = tf.random.shuffle(indices)
    X = tf.gather(X, indices)
    y = tf.gather(y, indices)

    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"Train X shape: {X_train.shape}, Train y shape: {y_train.shape}")
    print(f"Val X shape: {X_val.shape}, Val y shape: {y_val.shape}")

    model = build_model()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(1000).batch(32)

    history = model.fit(
        train_dataset,
        validation_data=(X_val, y_val),
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
    )

    model.save("heart_model.keras")

    print(f"\nBest Validation Accuracy: {max(history.history['val_accuracy']):.4f}")
    print(f"Final AUC: {max(history.history['val_auc']):.4f}")

    preds = model.predict(X[:10])
    print("Sample true labels:", y[:10].numpy())
    print("Sample predictions:", preds.squeeze())

if __name__ == "__main__":
    train()
