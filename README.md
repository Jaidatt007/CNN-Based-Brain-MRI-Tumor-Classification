# 🧠 Brain Tumor Detection Using CNN

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package_manager-purple.svg)](https://github.com/astral-sh/uv)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red.svg)](https://keras.io/)
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-black.svg)](https://flask.palletsprojects.com/)
[![Computer Vision](https://img.shields.io/badge/Computer_Vision-MRI_Analysis-green.svg)](#)
[![Medical AI](https://img.shields.io/badge/Medical_AI-Assistive_System-blueviolet.svg)](#)

> An **end-to-end Convolutional Neural Network (CNN) based Brain Tumor Detection System** that classifies MRI brain scans into multiple tumor categories and deploys the trained model using a **Flask web application** with confidence visualization.

---

## 🚀 Project Overview

A **Convolutional Neural Network (CNN)** system that detects and classifies brain tumors from MRI scans into four categories: Glioma, Meningioma, Pituitary, and No Tumor. The system combines computer vision techniques with deep learning to provide accurate medical imaging analysis.

### Key Features:

* ✨ **Multi-class Classification** - Identifies 4 distinct tumor types with 96% accuracy
* 🧠 **Deep CNN Architecture** - Custom-built convolutional layers for medical image analysis
* 📊 **Confidence-based Predictions** - Softmax output with probability scores (0.0 - 1.0)
* 🎨 **Interactive Web Interface** - Flask-powered UI for real-time predictions
* 🔍 **Comprehensive Evaluation** - ROC curves, confusion matrices, and classification reports

This project demonstrates **end-to-end deep learning pipeline development**, from MRI preprocessing to model deployment, showcasing practical applications of AI in healthcare diagnostics.

---

## 🧠 Core Concept

```
MRI Image → Preprocessing → CNN Model → Tumor Classification + Confidence Score
```

This project uses **Convolutional Neural Networks** specifically designed for medical image analysis:

* **Input Layer**: MRI scans (grayscale/RGB images) preprocessed and normalized
* **Convolutional Layers**: Extract spatial features like edges, textures, and patterns
* **Pooling Layers**: Reduce dimensionality while preserving critical information
* **Dense Layers**: Learn complex feature combinations for classification
* **Output Layer**: 4-class softmax activation for tumor type prediction

### Benefits:

* 🎯 **High Accuracy** - 96% classification performance on test data
* 🔬 **Medical-grade Pipeline** - Follows standard radiology imaging workflows
* 📈 **Scalable Architecture** - Easily adaptable to other medical imaging tasks
* ⚡ **Fast Inference** - Real-time predictions in web application
* 🔓 **Open Source** - Fully transparent model architecture and training process

### How It Works:

1. **Data Collection**: MRI brain scans organized by tumor type
2. **Preprocessing**: Image resizing, normalization, and augmentation
3. **Model Training**: CNN learns patterns distinguishing tumor types
4. **Validation**: Performance evaluation using multiple metrics
5. **Deployment**: Flask web app for clinical-style predictions

---

## 🎯 Why This Approach?

**Traditional Methods** require manual feature engineering and expert radiologist analysis, which can be:
- ⏱️ Time-consuming
- 💰 Expensive
- 👥 Subject to human variability

**CNN-based Detection** offers:
- ⚡ **Automated Analysis** - Instant processing of MRI scans
- 🎯 **Consistent Results** - Eliminates inter-observer variability
- 📊 **Quantifiable Confidence** - Probability scores for each prediction
- 🔄 **Continuous Learning** - Model improves with more training data
- 🤝 **Assistive Tool** - Supports radiologists in decision-making

> **Note**: This is an assistive AI system designed to augment, not replace, medical professionals' expertise.

---

## 📌 Demo Preview

![Project Demo](snapshots/demo.gif)

---

## 🧠 CNN Architecture

The following diagram represents the complete CNN pipeline used in this project, from MRI input to final prediction.

![CNN Architecture](snapshots/cnn_architecture.png)

---

## 🚀 Project Features

- ✅ **Multi-class Classification** - Detects 4 tumor types (Glioma, Meningioma, Pituitary, No Tumor)
- ✅ **Advanced Preprocessing** - MRI image normalization and augmentation
- ✅ **High Accuracy** - Achieves ~96% classification accuracy
- ✅ **Confidence Scores** - Softmax-based prediction confidence
- ✅ **Web Deployment** - Interactive Flask-based web interface
- ✅ **Visualization Dashboard** - Real-time prediction visualization
- ✅ **Ethical AI** - Includes medical disclaimer for responsible usage

---

## 🗂️ Project Directory Structure

```
CNN/
│
├── dataset/
│   ├── brain-tumor-mri-dataset/
│   └── download_dataset.ipynb
│
├── model/
│   ├── model_building.ipynb
│   └── model.h5
│
├── snapshots/
│   ├── cnn_architecture.png
│   ├── dataset_visualization.png
│   ├── classification_report.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── model_confidence_distribution.png
│   ├── training_accuracy_loss_over_epochs.png
│   ├── model_output_with_confidence_glioma.png
│   ├── model_output_with_confidence_meningioma.png
│   ├── model_output_with_confidence_no_tumor.png
│   ├── model_output_with_confidence_pituitary.png
│   └── demo.gif
│
├── static/
│   ├── CSS/
│   └── uploads/
│
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
│
├── main.py
├── requirements.txt
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## ⚙️ Installation & Environment Setup

This project uses **UV** (Ultra-fast Python package manager) for dependency management.

### Prerequisites

- Python 3.8 or higher
- UV package manager ([Installation Guide](https://github.com/astral-sh/uv))

### Setup Steps

**1. Clone the Repository**

```bash
git clone https://github.com/Jaidatt007/CNN-Based-Brain-MRI-Tumor-Classification.git
cd CNN-Based-Brain-MRI-Tumor-Classification
```

**2. Create Virtual Environment (Recommended)**

```bash
uv venv
```

**3. Install Dependencies**

```bash
uv add -r requirements.txt
```

---

## ▶️ Execution Flow

Follow these steps **strictly in order** for proper project execution:

### 🔹 Step 1: Download Dataset

Run the dataset download notebook:

```bash
jupyter notebook dataset/download_dataset.ipynb
```

**What it does:**
- Downloads the MRI dataset
- Organizes images into class-wise folders

### 🔹 Step 2: Train CNN Model

Run the model building notebook:

```bash
jupyter notebook model/model_building.ipynb
```

**What it does:**
- Preprocesses MRI images
- Trains the CNN model
- Saves the trained model as `model.h5`

### 🔹 Step 3: Run Web Application

Start the Flask application:

```bash
uv run python main.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 🖥️ Web Application Output

The web interface allows users to upload MRI scans and receive instant predictions with confidence scores.

### Features:
- Drag-and-drop image upload
- Real-time prediction results
- Confidence score visualization
- Tumor type classification

---

## 📊 Model Performance & Evaluation

### 🔹 Dataset Visualization

![Dataset Visualization](snapshots/dataset_visualization.png)

### 🔹 Training Progress

![Training Accuracy & Loss](snapshots/training_accuracy_loss_over_epochs.png)

**Training Metrics:**
- Accuracy: **82% → 98%**
- Loss: **0.48 → 0.06**
- Stable convergence without overfitting

### 🔹 Classification Report

![Classification Report](snapshots/classification_report.png)

**Key Metrics:**
- **Overall Accuracy:** 96%
- **Balanced Precision, Recall & F1-score** across all classes

### 🔹 Confusion Matrix

![Confusion Matrix](snapshots/confusion_matrix.png)

**Highlights:**
- Strong diagonal dominance
- Minimal misclassification between classes

### 🔹 ROC Curve (AUC Analysis)

![ROC Curve](snapshots/roc_curve.png)

**Performance:**
- **AUC ≈ 1.0** for all classes
- Excellent class separability

### 🔹 Confidence Distribution

![Confidence Distribution](snapshots/model_confidence_distribution.png)

**Observation:**
- Most predictions achieve >90% confidence
- Indicates strong model certainty

---

## 🧪 Sample Model Predictions

| Tumor Type | Prediction Output |
|------------|-------------------|
| **Glioma** | ![Glioma](snapshots/model_output_with_confidence_glioma.png) |
| **Meningioma** | ![Meningioma](snapshots/model_output_with_confidence_meningioma.png) |
| **No Tumor** | ![No Tumor](snapshots/model_output_with_confidence_no_tumor.png) |
| **Pituitary** | ![Pituitary](snapshots/model_output_with_confidence_pituitary.png) |

---

## ⚠️ Medical Disclaimer

> **IMPORTANT:** This AI-powered system is intended **only for research and educational purposes**.
> 
> - ❌ It should **NOT** be used as a substitute for professional medical diagnosis or treatment
> - ✅ Always consult **qualified healthcare professionals** for medical decisions
> - ✅ This tool is designed to assist, not replace, medical expertise

---

## 🔮 Future Enhancements

- 🔹 **Grad-CAM Visualization** - Add explainability for model predictions
- 🔹 **DICOM Support** - Enable direct medical imaging format compatibility
- 🔹 **Multi-modal MRI** - Support for T1, T2, FLAIR sequences
- 🔹 **Cloud Deployment** - Deploy on AWS/Azure for scalability
- 🔹 **Real-time Monitoring** - Add model performance tracking dashboard
- 🔹 **Mobile Application** - Develop iOS/Android app for accessibility

---

## 💡 Original Work & Development

This project is **entirely built from scratch** by me, to demonstrate comprehensive conceptual knowledge and hands-on expertise in Convolutional Neural Networks. Every component - from model architecture to deployment - was independently developed by me to showcase end-to-end deep learning proficiency in medical image classification.
I took help of AI tools for DOC string generation and UI enhancements.

---

## 👤 Author

**Jaidatt Kale**

- 🔗 [LinkedIn](https://linkedin.com/in/jaidattkale)
- 🔗 [GitHub](https://github.com/jaidatt007)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

---

## ⭐ Show Your Support

If you find this project helpful, please give it a ⭐ on GitHub!

---

## 📧 Contact

For questions, suggestions, or collaborations, feel free to reach out:

- **Email:** jaidattkale555@gmail.com
- **LinkedIn:** [Jaidatt Kale](https://linkedin.com/in/jaidattkale)

---

**Made with ❤️ for advancing Medical AI Research**
