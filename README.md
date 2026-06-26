# 🖼️ Image Caption Generator using CNN-LSTM

An AI-powered image captioning system that automatically generates meaningful textual descriptions for input images using **Deep Learning**. The project combines a **Convolutional Neural Network (CNN)** for image feature extraction and a **Long Short-Term Memory (LSTM)** network for natural language generation.

---

## 📌 Table of Contents

* [About the Project](#about-the-project)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Project Workflow](#project-workflow)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Model Architecture](#model-architecture)
* [Results](#results)
* [Future Enhancements](#future-enhancements)
* [License](#license)

---

## 📖 About the Project

Image Caption Generation is a Computer Vision and Natural Language Processing task that automatically generates descriptive captions for images.

This project implements an **Encoder–Decoder architecture**, where:

* **CNN (VGG16/InceptionV3)** extracts visual features from images.
* **LSTM** generates captions word-by-word based on the extracted features.
* Transfer Learning is used to improve feature extraction and reduce training time.

The system can generate human-readable captions for unseen images and demonstrates the integration of Computer Vision with Deep Learning.

---

## ✨ Features

* Upload an image for caption generation.
* Automatic image preprocessing.
* CNN-based feature extraction.
* LSTM-based sequential caption generation.
* Transfer Learning using a pretrained model.
* Simple and user-friendly interface.
* Generates captions for unseen images.

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras

### Libraries

* NumPy
* Pandas
* Matplotlib
* Pillow (PIL)
* OpenCV

### Model

* CNN (VGG16/InceptionV3)
* LSTM
* Encoder–Decoder Architecture

---

## 🔄 Project Workflow

1. Upload an input image.
2. Preprocess the image.
3. Extract image features using a pretrained CNN.
4. Feed extracted features into the LSTM decoder.
5. Generate caption word-by-word.
6. Display the generated caption.

---

## 📂 Project Structure

```
Image-Caption-Generator/
│
├── dataset/
├── models/
├── notebooks/
├── src/
│   ├── feature_extraction.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── caption_generator.py
│   └── app.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

*(Modify this structure according to your repository.)*

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Jyosna Dastagiri/Image-Caption-Generator.git
```

Navigate to the project folder:

```bash
cd Image-Caption-Generator
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

---

## 🚀 Usage

* Launch the application.
* Upload an image.
* Wait for the model to process the image.
* View the generated caption.

---

## 🧠 Model Architecture

```
Input Image
      │
      ▼
 Image Preprocessing
      │
      ▼
CNN Encoder (VGG16/InceptionV3)
      │
Extracted Features
      │
      ▼
LSTM Decoder
      │
      ▼
Generated Caption
```

---

## 📊 Results

The model successfully generates meaningful captions for a wide variety of images by learning the relationship between visual features and textual descriptions.

Example:

**Input Image**

📷 Image containing a dog playing in a park.
<img width="708" height="397" alt="image" src="https://github.com/user-attachments/assets/2a6701b2-71b9-4940-a3f9-612c20a91922" />



**Generated Caption**

> "A dog is running through the grass."

---

## 🚀 Future Enhancements

* Attention Mechanism
* Transformer-based Caption Generation
* Multilingual Caption Support
* Real-time Video Captioning
* Mobile Deployment
* Improved Web Interface
* Domain-specific Dataset Training

---

## 📄 License

This project is developed for academic and learning purposes.
Feel free to modify and extend it for educational or research use.

---


