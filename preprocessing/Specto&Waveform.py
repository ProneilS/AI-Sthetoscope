import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
import tensorflow as tf

def show_waveform_and_spectrogram(file_path, label):
    # Load audio
    sr, audio = wavfile.read(file_path)
    audio = audio.astype(np.float32)

    # Normalize
    p99 = np.percentile(np.abs(audio), 99)
    audio = np.clip(audio / (p99 + 1e-6), -1, 1)

    # Plot waveform
    plt.figure(figsize=(14, 4))
    plt.plot(np.arange(len(audio)) / sr, audio)
    plt.title(f"Waveform - {label}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Mel spectrogram
    frame_length = 256
    frame_step = 128
    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, pad_end=True)
    spectrogram = tf.abs(spectrogram) + 1e-6

    num_mels = 64
    mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mels,
        num_spectrogram_bins=frame_length // 2 + 1,
        sample_rate=sr,
        lower_edge_hertz=20,
        upper_edge_hertz=500
    )

    mel_spectrogram = tf.tensordot(spectrogram, mel_filterbank, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram)

    # Plot mel spectrogram
    plt.figure(figsize=(14, 5))
    plt.imshow(tf.transpose(mel_spectrogram).numpy(), aspect='auto', origin='lower', cmap='magma')
    plt.title(f"Mel Spectrogram - {label}")
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Frequency Bands (20–500 Hz)")
    plt.colorbar(label='Log Magnitude')
    plt.tight_layout()
    plt.show()

# Show healthy and unhealthy heart sounds
show_waveform_and_spectrogram("physionet_data/a0007.wav", label="Healthy Heart Sound")
show_waveform_and_spectrogram("physionet_data/a0001.wav", label="Unhealthy Heart Sound (Murmur)")
