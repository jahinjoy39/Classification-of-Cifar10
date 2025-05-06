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

- Normalize the data by dividing pixel values by 255.0.


### 2. **Load CIFAR-10 Dataset**
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
```

### 3. **Data Exploration**
- Display sample images from the dataset.

- Check data shapes and distribution of classes.

### 4. **Build the CNN Model**
- Stack multiple convolutional layers, pooling layers, and dropout layers.

- Use ReLU activation and softmax for the output layer.

Example structure:

```python
model = Sequential()
model.add(Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
```

### 5. **Compile the Model**
- Optimizer: Adam

- Loss: Categorical Crossentropy

- Metrics: Accuracy

### 6. **Train the Model**
- Fit the model using model.fit()

- Plot training and validation accuracy/loss over epochs.

### 7. **Evaluate the Model**
- Evaluate accuracy on the test set using model.evaluate().

- Display a confusion matrix and classification report for deeper insights.

### 8. **Results Visualization**
- Plot training history graphs (accuracy/loss).

- Visualize misclassified samples for further analysis.
