
# Handwritten Character Recognition System

### 📌 Project Overview
This project was developed as **Task 01** for my remote engineering internship. The goal is to build an intelligent Deep Learning system that takes an image of a handwritten digit (0–9) as input and accurately predicts its numerical value.

The project is built using **Python**, **TensorFlow/Keras**, and **OpenCV**, following a clean, modular folder structure that is production-ready and easy to scale.

---

## 🚀 Key Features
* **Modular Codebase:** Separated data loading, model architecture, training, and testing scripts.
* **Over 98% Test Accuracy:** Achieved high performance using a custom Convolutional Neural Network (CNN).
* **Automated Evaluations:** Generates and saves Accuracy/Loss curves and a Confusion Matrix after training.
* **Real-Time Custom Inference:** Includes a standalone script to test the model with your own handwritten images.

---

## 📂 Project Folder Structure
```text
handwritten-character-recognition/
│
├── models/
│   ├── mnist_cnn_model.h5       # Saved trained model weights
│   ├── training_curves.png      # Output evaluation graphs
│   └── confusion_matrix.png     # Output metrics matrix
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Data loading and scaling pipeline
│   ├── model.py                 # CNN architecture definition
│   ├── train.py                 # Model training loop execution
│   └── predict.py               # Custom inference script
│
├── utils/
│   ├── __init__.py
│   └── visualization.py         # Matplotlib/Seaborn charting helpers
│
├── requirements.txt             # Project library dependencies
└── README.md                    # Project documentation

```

---

## 🛠️ Tech Stack & Dependencies

* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Processing:** NumPy, Scikit-Learn
* **Computer Vision:** OpenCV (opencv-python)
* **Visualization:** Matplotlib, Seaborn

---

## 🤖 Model Architecture

The system uses a **Convolutional Neural Network (CNN)** optimized for spatial image feature extraction:

1. **Conv2D Layer (32 filters, 3x3 kernel, ReLU)** - Extracts basic shapes like lines and edges.
2. **MaxPooling2D (2x2)** - Reduces spatial size.
3. **Conv2D Layer (64 filters, 3x3 kernel, ReLU)** - Extracts complex structural patterns.
4. **MaxPooling2D (2x2)** - Downsamples the feature maps.
5. **Flatten Layer** - Converts 2D matrices into a 1D vector.
6. **Dense Layer (128 units, ReLU)** - Fully connected layer for classification.
7. **Dropout Layer (0.3)** - Prevents overfitting during training.
8. **Dense Output Layer (10 units, Softmax)** - Predicts probability for numbers 0 to 9.

---

## 💻 How to Run the Project

### 1. Clone the Repository

```bash
git clone [https://github.com/musfique-anam/CodeAlpha_Handwritten-Character-Recognition.git]
cd handwritten-character-recognition

```

### 2. Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

### 3. Train the Model

This script loads the MNIST dataset, normalizes the images, trains the CNN, and saves the evaluation graphs in the `models/` directory.

```bash
python src/train.py

```

### 4. Test with a Custom Image

You can test the trained model on any new handwritten image. Create a white background image with a black drawing of a number and run:

```bash
python src/predict.py path/to/your_image.png

```

---

## 📊 Evaluation Results

* **Final Test Accuracy:** **>98%**
* **Training and Validation Metrics:** Training charts (`training_curves.png`) show smooth convergence with zero overfitting issues.
* **Confusion Matrix:** The classification matrix (`confusion_matrix.png`) confirms high precision across all 10 target digit classes.

---

## 🔮 Future Improvements

* Upgrade the pipeline to support the **EMNIST dataset** for full alphabet recognition (A-Z).
* Build a graphical user interface (GUI) using **Streamlit** or **Tkinter** for drawing and predicting digits on screen.
* Implement a full OCR system to read multiple words or paragraphs from a document.

