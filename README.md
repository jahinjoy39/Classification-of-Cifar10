# 🧠 CIFAR-10 Image Classification using CNN (TensorFlow/Keras)

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset into 10 different categories using TensorFlow and Keras.

---

## 📁 Files Included

- `Cifer10.ipynb`: Jupyter Notebook with CNN model design, training, evaluation, and visualizations.
- `HW3.pdf`: Assignment prompt and evaluation criteria.
- `README.md`: This documentation file.

---

## 📊 Project Description

The objective of this assignment is to build and evaluate a Convolutional Neural Network (CNN) model using the CIFAR-10 dataset. The dataset contains 60,000 color images of size 32x32 in 10 different classes with 6,000 images per class.

### CIFAR-10 Classes:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

---

## 🧪 Step-by-Step Process

### 1. **Import Libraries**
- Import essential packages: `tensorflow`, `keras`, `matplotlib`, `numpy`, and others for preprocessing and visualization.

### 2. **Load CIFAR-10 Dataset**
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
