# Import libraries for ANN computation and visualization
import numpy as np
# Import libraries for ANN computation and visualization
import matplotlib.pyplot as plt
# Import libraries for ANN computation and visualization
import tensorflow as tf
# Import libraries for ANN computation and visualization
from tensorflow.keras import layers, models
# Import libraries for ANN computation and visualization
from tensorflow.keras.datasets import cifar10

# Training phase: updates weights based on input-output error
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize (0–255 → 0–1)
# Training phase: updates weights based on input-output error
x_train = x_train / 255.0
x_test = x_test / 255.0

# Training phase: updates weights based on input-output error
print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

class_names = ['Airplane','Automobile','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']

plt.figure(figsize=(8,4))
# Loop: iterates through dataset or epochs during training
for i in range(10):
    plt.subplot(2,5,i+1)
# Training phase: updates weights based on input-output error
    plt.imshow(x_train[i])
# Training phase: updates weights based on input-output error
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()

# Stores trained models for multi-class classification
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
# Training phase: updates weights based on input-output error
    x_train, y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.1
)

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)
# Training phase: updates weights based on input-output error
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
# Training phase: updates weights based on input-output error
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss")
plt.legend()

plt.show()

# Import libraries for ANN computation and visualization
import cv2

img = cv2.imread("image.jpg")   # keep image in folder
img = cv2.resize(img, (32,32))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0

img = np.expand_dims(img, axis=0)

# Prediction phase: determines output using trained model
prediction = model.predict(img)
# Prediction phase: determines output using trained model
pred_class = class_names[np.argmax(prediction)]

# Prediction phase: determines output using trained model
print("Predicted Class:", pred_class)

plt.imshow(img[0])
plt.title(pred_class)
plt.axis('off')
plt.show()



