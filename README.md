# 🩺 AI Stethoscope – Smart Heart Sound Classifier

![MIT License](https://img.shields.io/github/license/ProneilS/AI-Sthetoscope)
![Repo Size](https://img.shields.io/github/repo-size/ProneilS/AI-Sthetoscope)
![Last Commit](https://img.shields.io/github/last-commit/ProneilS/AI-Sthetoscope)
![Issues](https://img.shields.io/github/issues/ProneilS/AI-Sthetoscope)


This project combines deep learning with embedded hardware to create an **AI-powered stethoscope** that classifies heart sounds as *normal* or *abnormal*, with the long-term goal of detecting **murmurs from specific heart valves** (aortic, mitral, tricuspid, pulmonary).

It uses an **ESP32** microcontroller and an **I2S microphone (INMP441/MAX9814)** to record audio, which is preprocessed into **Mel spectrograms** and fed into a **Convolutional Neural Network (CNN)** built using TensorFlow/Keras.

---

## 🎯 Project Goals

- ✔️ Build a working smart stethoscope capable of heart sound classification  
- 🔄 Currently supports binary classification (normal vs abnormal)  
- 🧠 Ongoing work: classification based on **specific heart valves**  
- 💡 Target: Deploy AI models on-device for real-time diagnostics  

---

## 📦 Features

- Real-time heart sound acquisition using ESP32  
- Audio preprocessing (normalization + spectrogram generation)  
- CNN-based classifier trained on PhysioNet and custom recordings  
- Live waveform + spectrogram visualization tools  
- Designed for low-resource edge deployment (future TFLite support)  

---

## 🧠 How It Works – Tech Behind the Project

### 🎤 Why Spectrograms?

Raw audio is difficult for ML models to interpret directly. Spectrograms transform the waveform into a 2D image that represents frequency content over time — perfect for picking up heart murmurs.

- **X-axis**: Time  
- **Y-axis**: Frequency (20–500 Hz, where heart sounds lie)  
- **Color**: Amplitude of signal at that frequency  

This gives the CNN a visual representation of heart sound patterns.

---

### 🧰 Why CNN?

CNNs excel at image pattern recognition. By treating a spectrogram like an image, the model can:
- Recognize murmur patterns  
- Learn timing between heartbeats  
- Detect anomalies in rhythm or frequency distribution  

---

### 🧱 Why Keras & TensorFlow?

Keras (running on TensorFlow) simplifies model creation and training:
- Quick prototyping with high-level APIs  
- Easy deployment via TensorFlow Lite  
- Stable, industry-grade framework  

---

### 🧬 Why This Model Works for Heart Sounds

Heart sounds contain **repeatable patterns** (like S1 and S2) that occur in regular cycles. Murmurs, on the other hand, show up as:
- Extra frequency content (turbulent flow)
- Unusual timing (like systolic or diastolic whooshing)
- Distortions in amplitude and rhythm

💡 The **Mel spectrogram** converts audio into a time-frequency plot, where:
- Repeated heartbeats appear as rhythmic blobs
- Murmurs appear as smeared or spread-out frequencies

Once we have the spectrogram, the **CNN** can do what it's great at:
- Learn **spatial patterns** in 2D images
- Detect **edges, blobs, anomalies** in these patterns
- Build abstract representations (healthy rhythm vs. murmur)

---

### 🧪 What the CNN Actually “Sees”

When trained on hundreds of labeled examples:
- It starts to recognize that **healthy hearts** have clear, separated frequency bursts (S1/S2)
- **Abnormal hearts** show irregular energy in unexpected frequency ranges — like noise between S1 and S2

The CNN doesn’t "listen" — it **looks for consistent shapes in the spectrogram** and learns what makes them different.

---

### 🧠 Model Design Summary

- **Input**: `(556 time steps, 64 Mel bands, 1 channel)`
- **Conv Layers**: Detect local spectrogram features (e.g., murmur patches)
- **Pooling + BatchNorm**: Reduce dimensionality + stabilize training
- **Dense Layer**: Classifies the final pattern
- **Output**: Sigmoid → gives probability of being abnormal

---

### 💡 Why This Works
- Heart sounds are rhythmic → CNN can learn the **temporal regularity**
- Murmurs disrupt that rhythm → CNN can learn **deviations**
- Mel scale helps capture **subtle low-frequency changes** important for auscultation

---

## 📉 What We Can't Achieve Yet

- ❌ We cannot yet **differentiate murmurs by valve** (e.g. aortic vs mitral)
- ❌ No real-time inference on ESP32 (TFLite not deployed yet)
- ❌ Small sample size from own recordings limits generalization
- ❌ No labeled data for multi-label (valve-specific) classification
- ❌ No GUI or app interface (coming soon)

These are planned in future iterations.

---

## 📊 Results (Sample)

| Metric              | Value         |
|---------------------|---------------|
| Validation Accuracy | ~89.4%        |
| Validation AUC      | ~0.93         |
| Best Epoch          | ~60–70        |

*Trained on PhysioNet + limited self-recorded data. Generalization will improve with more data.*

---

## 🛠 Hardware Used

| Component         | Description                              |
|------------------|------------------------------------------|
| ESP32 Dev Board   | Microcontroller with Wi-Fi/Bluetooth     |
| INMP441 / MAX9814 | Digital mic (I2S compatible)             |
| LCD Display       | 16x2 HD44780 to show predictions         |
| LM7805 Regulator  | Voltage regulator for stable 5V output   |
| Battery           | 9V or Li-ion cell                        |

📄 See [`hardware/components.md`](./hardware/components.md) for full list and specs.

---

## 📂 Folder Structure
    AI-Stethoscope/
    ├── model/ # Training, testing scripts
    ├── preprocessing/ # Waveform + spectrogram visualizers
    ├── hardware/ # Components & prototype details
    ├── docs/ # Diagrams, block flow (WIP)
    ├── requirements.txt # Python dependencies
    ├── LICENSE # MIT License
    └── README.md # This file
    
---

## 🚧 Project Status

This is an **active work-in-progress**.

✅ Now:
- Binary classifier (normal/abnormal)  
- CNN trained on PhysioNet + self data  
- Live spectrogram viewer  

🛠 Coming Soon:
- Valve-wise classification (aortic, mitral, etc.)  
- TFLite model for on-device inference  
- ESP32 firmware integration  
- GUI/Mobile interface  

---

## 🔧 Setup & Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt

2. Train the model:
    ```bash
    python model/Training.py

3. Test a few .wav files:
    ```bash
    python model/Testing.py

4. View waveform & spectogram:
    ```bash
    python preprocessing/Specto&Waveform.py
