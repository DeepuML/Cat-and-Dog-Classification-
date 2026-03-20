<h1 align="center">🐱🐶 Cat and Dog Image Classification</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white" alt="Jupyter">
</p>

<p align="center">
  A production-ready deep learning pipeline that classifies images as <strong>cat</strong> or <strong>dog</strong> using a custom Convolutional Neural Network (CNN) built with TensorFlow/Keras. Covers the end-to-end ML workflow: data ingestion, preprocessing, model training, evaluation, and model serialization.
</p>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Features](#-features)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📌 Project Overview

This project implements a binary image classifier that distinguishes between cats and dogs using a custom CNN trained from scratch with TensorFlow/Keras. The complete ML pipeline includes:

- **Data ingestion** – structured directory-based dataset loading with `ImageDataGenerator`
- **Preprocessing** – pixel normalization (rescaling to [0, 1]), resizing to 150×150, and an 80/20 train-validation split
- **Model training** – 4-block CNN with ReLU activations, Max-Pooling layers, and a sigmoid output unit optimized with Adam
- **Evaluation** – training/validation accuracy & loss curves, confusion matrix, classification report, and ROC-AUC score
- **Model persistence** – saved in both `.keras` (native format) and `.h5` (HDF5/legacy) for broad compatibility

---

## 🗂️ Project Structure

```
Cat-and-Dog-Classification-/
│
├── code.ipynb                    # Main Jupyter Notebook (end-to-end pipeline)
│
├── cat_dog_classifier.keras      # Saved model – native Keras format
├── cat_dog_classifier.h5         # Saved model – legacy HDF5 format
│
├── dataset/
│   ├── train/                    # Training images (organized by class)
│   │   ├── cat/                  # ~95 cat images (multiple breeds)
│   │   └── dog/                  # ~180 dog images (multiple breeds)
│   ├── val/                      # Validation images (organized by class)
│   │   ├── cat/                  # ~24 cat images
│   │   └── dog/                  # ~46 dog images
│   ├── train.csv                 # Training image paths and labels (0=cat, 1=dog)
│   └── val.csv                   # Validation image paths and labels (0=cat, 1=dog)
│
└── README.md                     # Project documentation (this file)
```

---

## 🗃️ Dataset

The dataset is a curated subset of the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) exported via Roboflow, organized into train/validation splits:

| Split      | Cats | Dogs | Total |
|------------|------|------|-------|
| Training   |  95  | 180  |  275  |
| Validation |  24  |  46  |   70  |

**Key characteristics:**
- Images sourced from multiple breeds (e.g., Bengal, Sphynx, Persian, Boxer, Beagle, Samoyed, Labrador)
- Variable raw image sizes – all resized to **150×150** during preprocessing
- Labels: `0 = cat`, `1 = dog`
- CSV manifests (`train.csv`, `val.csv`) provided alongside the directory structure for easy loading

---

## 🧠 Model Architecture

A custom sequential CNN with **4 convolutional blocks** followed by fully connected layers:

```
Input (150 × 150 × 3)
        │
Conv2D(32, 3×3, ReLU)  →  MaxPooling2D(2×2)
        │
Conv2D(64, 3×3, ReLU)  →  MaxPooling2D(2×2)
        │
Conv2D(128, 3×3, ReLU) →  MaxPooling2D(2×2)
        │
Conv2D(128, 3×3, ReLU) →  MaxPooling2D(2×2)
        │
    Flatten
        │
  Dense(512, ReLU)
        │
  Dense(1, Sigmoid)   ←  Binary output: cat (0) / dog (1)
```

| Component              | Detail                             |
|------------------------|------------------------------------|
| Loss function          | Binary Cross-Entropy               |
| Optimizer              | Adam (lr = 0.001)                  |
| Output activation      | Sigmoid                            |
| Prediction threshold   | > 0.5 → Dog,  ≤ 0.5 → Cat        |
| Training epochs        | 10                                 |
| Batch size             | 20                                 |
| Input image size       | 150 × 150 × 3 (RGB)               |

---

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

**1. Clone the repository:**

```bash
git clone https://github.com/DeepuML/Cat-and-Dog-Classification-.git
cd Cat-and-Dog-Classification-
```

**2. Create and activate a virtual environment (recommended):**

```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**3. Install required dependencies:**

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow opencv-python jupyter
```

---

## 🚀 Usage

### Running the Full Pipeline

Open the Jupyter Notebook and run all cells sequentially:

```bash
jupyter notebook code.ipynb
```

The notebook walks through the following steps:

| Step | Description |
|------|-------------|
| 1 | Library imports (TensorFlow, Keras, NumPy, Matplotlib, etc.) |
| 2 | Dataset loading using `ImageDataGenerator` |
| 3 | Image visualization from the training set |
| 4 | Data batching with 80/20 train-validation split |
| 5 | CNN model construction |
| 6 | Model compilation (Adam optimizer, binary cross-entropy loss) |
| 7 | Model training for 10 epochs |
| 8 | Training/validation loss & accuracy plots |
| 9 | Model evaluation on the validation set |
| 10 | Confusion matrix (Seaborn heatmap) |
| 11 | Classification report (precision, recall, F1-score) |
| 12 | ROC-AUC curve |
| 13 | Model saving (`.keras` and `.h5` formats) |

### Loading a Pre-trained Model for Inference

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model('cat_dog_classifier.keras')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Predict on a new image
img = load_img('your_image.jpg', target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0][0]
label = 'Dog 🐶' if prediction > 0.5 else 'Cat 🐱'
print(f'Predicted: {label}  (score: {prediction:.4f})')
```

---

## 📊 Results

The model is evaluated on the held-out validation set after 10 epochs of training:

| Metric               | Value   |
|----------------------|---------|
| Training Accuracy    | ~90%+   |
| Validation Accuracy  | ~85%+   |
| ROC-AUC Score        | ~0.90+  |

**Evaluation artifacts generated inside the notebook:**

- 📈 **Loss & Accuracy Curves** – training vs. validation across 10 epochs
- 🟦 **Confusion Matrix** – true vs. predicted labels (Seaborn heatmap)
- 📋 **Classification Report** – per-class precision, recall, and F1-score
- 📉 **ROC Curve** – with computed AUC score

---

## ✅ Features

- ✅ End-to-end binary image classification pipeline in a single notebook
- ✅ Structured directory-based dataset with CSV label manifests
- ✅ Custom 4-block CNN built from scratch with TensorFlow/Keras
- ✅ `ImageDataGenerator` with rescaling and 80/20 train-validation split
- ✅ Training and validation accuracy/loss visualization (Matplotlib)
- ✅ Confusion matrix with Seaborn heatmap
- ✅ Full classification report (precision, recall, F1-score per class)
- ✅ ROC curve and AUC score via scikit-learn
- ✅ Model saved in both `.keras` and `.h5` formats for broad compatibility
- ✅ Pre-trained model files included for immediate inference

---

## 🔮 Future Improvements

- [ ] Apply transfer learning using **MobileNetV2**, **ResNet50**, or **EfficientNetB0** for higher accuracy on limited data
- [ ] Add data augmentation (horizontal flip, zoom, rotation) via `ImageDataGenerator` to reduce overfitting
- [ ] Integrate **Dropout** and **Batch Normalization** layers for better generalization
- [ ] Hyperparameter tuning using **Keras Tuner** or **Optuna**
- [ ] Export model to **ONNX** or **TFLite** for mobile/edge deployment
- [ ] Deploy as a web application using **Streamlit** or **Flask** with real-time image upload
- [ ] Extend to multi-class **pet breed classification** (37+ breeds from Oxford-IIIT)
- [ ] Add **Grad-CAM** saliency map visualization to interpret and explain model predictions
- [ ] Package as a **Docker** container for reproducible, portable deployment

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'feat: add your feature description'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure all notebook cells run without errors before submitting a PR.

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

---

<p align="center">Made with ❤️ by <a href="https://github.com/DeepuML">DeepuML</a></p>
