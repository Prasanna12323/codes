# =========================================
# ANN Perceptron Digit Recognition (Commented)
# =========================================

# Import required libraries
# numpy -> used for numerical operations like vectors and dot products
# matplotlib -> used to visualize digit patterns as images
import numpy as np
import matplotlib.pyplot as plt


# =========================================
# DATASET CREATION
# =========================================
# Each row represents a digit (0–9)
# Each digit is stored as a flattened 5x3 grid (15 values total)
# 1 = pixel ON, 0 = pixel OFF
# This is a simple binary image representation of digits

X = np.array([
    [1,1,1, 1,0,1, 1,0,1, 1,0,1, 1,1,1],  # Digit 0
    [0,1,0, 1,1,0, 0,1,0, 0,1,0, 1,1,1],  # Digit 1
    [1,1,1, 0,0,1, 1,1,1, 1,0,0, 1,1,1],  # Digit 2
    [1,1,1, 0,0,1, 1,1,1, 0,0,1, 1,1,1],  # Digit 3
    [1,0,1, 1,0,1, 1,1,1, 0,0,1, 0,0,1],  # Digit 4
    [1,1,1, 1,0,0, 1,1,1, 0,0,1, 1,1,1],  # Digit 5
    [1,1,1, 1,0,0, 1,1,1, 1,0,1, 1,1,1],  # Digit 6
    [1,1,1, 0,0,1, 0,0,1, 0,0,1, 0,0,1],  # Digit 7
    [1,1,1, 1,0,1, 1,1,1, 1,0,1, 1,1,1],  # Digit 8
    [1,1,1, 1,0,1, 1,1,1, 0,0,1, 1,1,1]   # Digit 9
])


# =========================================
# FUNCTION: DISPLAY DIGITS
# =========================================
# This function converts each 1D digit array into a 2D grid (5x3)
# and displays it as an image for visualization
def display_patterns(data, titles, rows=2, cols=5):
    plt.figure(figsize=(10, 5))

    for i in range(len(data)):
        plt.subplot(rows, cols, i + 1)

        # Reshape flat array into 5x3 grid to resemble image
        grid = data[i].reshape(5, 3)

        # Display grid as image
        plt.imshow(grid, cmap='binary')

        # Set title for each digit
        plt.title(titles[i])
        plt.axis('off')  # Hide axis for cleaner view

    plt.show()


# Show all digit patterns
display_patterns(X, [f"Digit {i}" for i in range(10)])


# =========================================
# FUNCTION: TRAIN PERCEPTRON
# =========================================
# This function trains a perceptron model for binary classification
# It learns weights and bias using the perceptron learning rule
def train_perceptron(X, y_binary, lr=0.1, epochs=100):

    # Initialize weights (same size as input features)
    # Initially all weights are zero
    w = np.zeros(X.shape[1])

    # Initialize bias as zero
    b = 0

    # Repeat training for multiple epochs
    for _ in range(epochs):

        # Loop through each training example
        for i in range(len(X)):

            # Calculate weighted sum (activation function)
            # z = w.x + b
            z = np.dot(X[i], w) + b

            # Apply step activation function
            # If z >= 0 → class +1 else -1
            y_pred = 1 if z >= 0 else -1

            # If prediction is wrong → update weights and bias
            if y_pred != y_binary[i]:

                # Update weights based on error
                w += lr * y_binary[i] * X[i]

                # Update bias
                b += lr * y_binary[i]

    # Return trained weights and bias
    return w, b


# =========================================
# TRAIN MULTIPLE MODELS (ONE-VS-ALL)
# =========================================
# Instead of one model, we train 10 perceptrons
# Each perceptron is responsible for detecting one digit

models = []

for digit in range(10):

    # Create binary labels:
    # +1 for current digit, -1 for all others
    y_binary = np.where(np.arange(10) == digit, 1, -1)

    # Train perceptron for this digit
    w, b = train_perceptron(X, y_binary)

    # Store model parameters
    models.append((w, b))


# =========================================
# FUNCTION: PREDICT DIGIT
# =========================================
# This function takes an input and evaluates it on all 10 models
# The model with highest score determines the predicted digit
def predict(x, models):

    # Compute score from each perceptron
    scores = [np.dot(x, w) + b for w, b in models]

    # Return index of maximum score (digit prediction)
    return np.argmax(scores)


# =========================================
# TESTING THE MODEL
# =========================================
# Test all training samples and check predictions
for i in range(10):
    pred = predict(X[i], models)

    # Show correct or incorrect prediction
    status = "✓" if pred == i else "✗"
    print(f"Actual: {i}, Predicted: {pred} {status}")


# =========================================
# TEST WITH NOISY INPUT
# =========================================
# Slightly modify digit 0 to check model robustness
noisy_0 = np.array([
    1,1,1,
    1,0,1,
    1,0,0,  # small noise introduced
    1,0,1,
    1,1,1
])

# Predict noisy digit
print("\nNoisy Input Prediction:", predict(noisy_0, models))

# Visualize noisy digit
plt.imshow(noisy_0.reshape(5, 3), cmap='Reds')
plt.title("Noisy Digit")
plt.axis('off')
plt.show()


# =========================================
# SUMMARY
# =========================================
# - Input: 5x3 binary digit patterns
# - Model: Perceptron (linear classifier)
# - Strategy: One-vs-All classification
# - Output: Digit (0–9)
# - Learning: Weight updates using error correction
