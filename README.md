# 🎩 Mustache Power Rating™

### The International Mustache Authority's Official AI System

Protecting society from fraudulent mustaches through the power of deep learning.

---

## Overview

Mustache Power Rating™ is a computer vision project that uses deep learning to detect, classify and evaluate mustaches.

The system combines face detection and convolutional neural networks to determine whether a mustache deserves recognition by the International Mustache Authority.

### Live Demo

https://moustacheclassifier.streamlit.app/

---

## Features

### Mustache Detection

Detects whether a face contains a mustache.

### Epicness Classification

Classifies mustaches into:

- Epic
- Thin

using a custom CNN trained on manually curated data.

### Face Detection

Uses MTCNN face detection to:

- Locate faces
- Crop facial regions
- Normalize inputs before classification

### Mustache Power Rating™

Generates an overall score based on:

- Mustache confidence
- Epicness confidence
- Facial hair quality indicators

---

## Model Development

### Model 1 – Mustache vs Clean

Initial CNN classifier trained on the CelebA dataset.

Result: 83.9% validation accuracy

---

### Dataset Cleaning

The first model revealed significant label noise within the dataset.

Many images labeled as "clean" actually contained visible mustaches.

Actions taken:

- Moved 526 mislabeled images into the Mustache class
- Added 1,052 female images to the Clean class
- Balanced the dataset to 5,526 images per class

---

### Model 1.5 – Cleaned Dataset

The model was retrained using the cleaned dataset.

Result: 90.9% validation accuracy

---

### Model 2 – Epic Mustache Classifier

A second CNN was trained to distinguish:

- Epic Mustaches
- Thin Mustaches

Dataset:

| Class | Images |
|---------|---------:|
| Epic | 334 |
| Thin | 263 |
| Total | 597 |

Result: 86.6% validation accuracy

---

## Technology Stack

- TensorFlow
- Keras
- Streamlit
- MTCNN
- FaceNet-PyTorch
- NumPy
- Pillow

---

## Installation

bash git clone https://github.com/nicklasthegerstrom-byte/mustache_power_rating.git  cd mustache_power_rating  pip install -r requirements.txt  streamlit run app.py 

---

## Classification Levels

### 🏆 Legendary

National treasure.

### 🔥 Epic

Exceptional upper lip performance.

### 👍 Respectable

Strong showing.

### 🤏 Thin

Potential detected.

### 🌱 Fjunig

It's just an upper lip with no ambition.

---

## Disclaimer

The International Mustache Authority is not a real government agency.

At least officially.