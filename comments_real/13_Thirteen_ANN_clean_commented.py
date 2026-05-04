# Import libraries required for ANN operations and data handling
import numpy as np
# Import libraries required for ANN operations and data handling
import matplotlib.pyplot as plt
# Import libraries required for ANN operations and data handling
import tensorflow as tf
# Import libraries required for ANN operations and data handling
from tensorflow.keras import layers, models
# Import libraries required for ANN operations and data handling
from tensorflow.keras.datasets import mnist

# Load data
# Training phase: adjusts weights using learning rules
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
# Training phase: adjusts weights using learning rules
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
# Training phase: adjusts weights using learning rules
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

# Model
# Stores trained models for multi-class classification
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training phase: adjusts weights using learning rules
# Train
# Training phase: adjusts weights using learning rules
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# Test
loss, acc = model.evaluate(x_test, y_test)
print("Accuracy:", acc)

# Import libraries required for ANN operations and data handling
import random

# Prediction phase: outputs classification using trained model
predictions = model.predict(x_test)

plt.figure(figsize=(6,6))

# Loop: iterates over dataset or epochs for training/testing
for i in range(9):
    idx = random.randint(0, len(x_test)-1)
    
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[idx].reshape(28,28), cmap='gray')
    
# Prediction phase: outputs classification using trained model
    pred = np.argmax(predictions[idx])
    actual = y_test[idx]
    
# Conditional logic: checks predictions or controls updates
    color = "green" if pred == actual else "red"
    plt.title(f"P:{pred} A:{actual}", color=color)
    
    plt.axis('off')

# Prediction phase: outputs classification using trained model
plt.suptitle("TensorFlow Predictions (Random Images)")
plt.show()

# Import libraries required for ANN operations and data handling
from keras.models import Sequential
# Import libraries required for ANN operations and data handling
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# Import libraries required for ANN operations and data handling
from keras.datasets import mnist

# Load data
# Training phase: adjusts weights using learning rules
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
# Training phase: adjusts weights using learning rules
x_train = x_train / 255.0
x_test = x_test / 255.0



# Reshape
# Training phase: adjusts weights using learning rules
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)



# Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



# Training phase: adjusts weights using learning rules
model.fit(x_train, y_train, epochs=5)

# Import libraries required for ANN operations and data handling
import random

# Conditional logic: checks predictions or controls updates
random.seed(42)  # different selection

# Prediction phase: outputs classification using trained model
predictions = model.predict(x_test)

plt.figure(figsize=(6,6))

# Loop: iterates over dataset or epochs for training/testing
for i in range(9):
    idx = random.randint(0, len(x_test)-1)
    
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[idx].reshape(28,28), cmap='gray')
    
# Prediction phase: outputs classification using trained model
    pred = np.argmax(predictions[idx])
    actual = y_test[idx]
    
# Conditional logic: checks predictions or controls updates
    color = "green" if pred == actual else "red"
    plt.title(f"P:{pred} A:{actual}", color=color)
    
    plt.axis('off')

# Prediction phase: outputs classification using trained model
plt.suptitle("Keras Predictions (Random Images)")
plt.show()

# Import libraries required for ANN operations and data handling
import torch
# Import libraries required for ANN operations and data handling
import torch.nn as nn
# Import libraries required for ANN operations and data handling
import torchvision
# Import libraries required for ANN operations and data handling
import torchvision.transforms as transforms

# Load dataset
transform = transforms.ToTensor()

# 🔹 Input Data
data = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

# Training phase: adjusts weights using learning rules
train_data = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# Training phase: adjusts weights using learning rules
test_data = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# Training phase: adjusts weights using learning rules
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

# CNN Model
class CNN(nn.Module):
# Function definition: encapsulates ANN logic (training/prediction)
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(5408, 10)
    
# Function definition: encapsulates ANN logic (training/prediction)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 5408)
        x = self.fc(x)
        return x


model = CNN()

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training phase: adjusts weights using learning rules
# Training (1 epoch only for simplicity)
# Loop: iterates over dataset or epochs for training/testing
for epoch in range(1):
# Training phase: adjusts weights using learning rules
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Training phase: adjusts weights using learning rules
print("Training done")

# Get shuffled batch
test_loader_iter = iter(test_loader)
images, labels = next(test_loader_iter)

outputs = model(images)
_, preds = torch.max(outputs, 1)

plt.figure(figsize=(6,6))

# Loop: iterates over dataset or epochs for training/testing
for i in range(9):
    idx = random.randint(0, len(images)-1)
    
    plt.subplot(3,3,i+1)
    plt.imshow(images[idx].squeeze(), cmap='gray')
    
    pred = preds[idx].item()
    actual = labels[idx].item()
    
# Conditional logic: checks predictions or controls updates
    color = "green" if pred == actual else "red"
    plt.title(f"P:{pred} A:{actual}", color=color)
    
    plt.axis('off')

# Prediction phase: outputs classification using trained model
plt.suptitle("PyTorch Predictions (Random Images)")
plt.show()



