# 🎧 Preprocessing Folder - Visualizing and Prepping Audio

This folder contains scripts to preprocess audio files into formats suitable for model training and testing.

---

## 📂 Files

### `show_waveform_and_spectrogram.py`
- Loads `.wav` files using `scipy`
- Normalizes the waveform
- Plots:
  - **Time-domain waveform**
  - **Mel spectrogram (log scale)**
- Uses `matplotlib` for visual output
- Great for visually confirming murmur presence

---

## 📊 Why Mel Spectrogram?
Mel scale mimics human hearing – lower frequencies are more detailed.
- More useful for heart sounds (which mostly lie between 20–500 Hz)
- CNNs work better with frequency-separated info

---

## 🔍 Preprocessing Highlights
- Clipping and log-scaling to stabilize training
- `tf.signal.stft()` → time-frequency analysis
- `tf.signal.linear_to_mel_weight_matrix()` → converts to Mel bins

Use this script to preview what your model "sees" during training.
