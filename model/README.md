# 🧠 Model Folder - Training & Testing

This folder contains scripts for training and evaluating the heart sound classification model.

---

## 📂 Files

### `training.py`
- Loads and balances dataset (normal vs abnormal)
- Applies preprocessing (normalization, spectrogram, MFCC)
- Trains a CNN using TensorFlow and Keras
- Saves final model as `heart_model.keras`

### `testing.py`
- Loads saved model
- Processes new `.wav` files into spectrograms
- Predicts class and confidence
- Helps validate model on known test sounds

---

## 🧪 Model Architecture
- 2 Conv2D layers → MaxPooling → BatchNorm → Dropout
- 1 Dense hidden layer → Sigmoid output
- Input shape: `(556, 64, 1)` Mel Spectrogram

---

## 🧰 Notes
- Uses `RECORDS-normal` and `RECORDS-abnormal` from `physionet_data/`
- Add more data to improve accuracy
