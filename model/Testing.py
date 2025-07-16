import os
import numpy as np
import tensorflow as tf
from scipy.io import wavfile

BASE_PATH = "physionet_data"
MODEL_PATH = "heart_model.keras"
MAX_TIME_STEPS = 556
SAMPLE_RATE = 2000
NUM_MELS = 64  # Must match training
FRAME_LENGTH = 256
FRAME_STEP = 128

def process_audio(file_path):
    sample_rate, audio = wavfile.read(file_path)
    audio = audio.astype(np.float32)
    p99 = np.percentile(np.abs(audio), 99)
    audio = np.clip(audio / (p99 + 1e-6), -1, 1)

    # STFT
    spectrogram = tf.abs(tf.signal.stft(
        audio, 
        frame_length=FRAME_LENGTH, 
        frame_step=FRAME_STEP, 
        pad_end=True
    )) + 1e-6

    # Convert to Mel
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=NUM_MELS,
        num_spectrogram_bins=FRAME_LENGTH//2 + 1,
        sample_rate=SAMPLE_RATE,
        lower_edge_hertz=20,
        upper_edge_hertz=500
    )
    mel_spectrogram = tf.tensordot(spectrogram, mel_filterbank, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram)

    # Standardize
    mean, var = tf.nn.moments(mel_spectrogram, axes=[0,1])
    mel_spectrogram = (mel_spectrogram - mean) / tf.sqrt(var + 1e-6)

    # Pad/trim
    if mel_spectrogram.shape[0] < MAX_TIME_STEPS:
        mel_spectrogram = tf.pad(mel_spectrogram, [[0, MAX_TIME_STEPS - mel_spectrogram.shape[0]], [0, 0]])
    else:
        mel_spectrogram = mel_spectrogram[:MAX_TIME_STEPS]

    return tf.expand_dims(mel_spectrogram, -1)

def test_model():
    model = tf.keras.models.load_model(MODEL_PATH)

    test_files = [
        ("a0025.wav", 0),
        ("a0027.wav", 0),
        ("a0028.wav", 0),
        ("a0029.wav", 0),
        ("a0017.wav", 1),
        ("a0018.wav", 1),
        ("a0020.wav", 1),
        ("a0021.wav", 1),
        ("b0001.wav", 0),
        ("b0002.wav", 0),
        ("b0003.wav", 0),
        ("b0004.wav", 0),
        ("b0008.wav", 1),
        ("b0013.wav", 1),
        ("b0016.wav", 1),
        ("b0018.wav", 1),

    ]

    for fname, true_label in test_files:
        wav_path = os.path.join(BASE_PATH, fname)
        if not os.path.exists(wav_path):
            print(f"File {fname} not found. Skipping...")
            continue

        try:
            spec = process_audio(wav_path)
            spec = tf.expand_dims(spec, axis=0)  # Add batch dim
            pred = model.predict(spec, verbose=0)[0][0]

            print(f"\nFile: {fname}")
            print(f"True label: {'ABNORMAL' if true_label else 'NORMAL'}")
            print(f"Raw prediction score: {pred:.4f}")
            print(f"Predicted: {'ABNORMAL' if pred > 0.5 else 'NORMAL'}")
            print(f"Confidence: {max(pred, 1-pred):.1%}")

            if pred > 0.99 or pred < 0.01:
                print("⚠️ Extreme prediction - check for potential issues")
        except Exception as e:
            print(f"Error processing {fname}: {str(e)}")

if __name__ == "__main__":
    test_model()