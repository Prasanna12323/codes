# Import libraries for numerical computation and ANN operations
import numpy as np

# Function definition: defines reusable ANN logic
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Function definition: defines reusable ANN logic
def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Decision logic: checks correctness or classification condition
W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
# Decision logic: checks correctness or classification condition
b1 = np.random.uniform(size=(1, hidden_neurons))

# Decision logic: checks correctness or classification condition
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
# Decision logic: checks correctness or classification condition
b2 = np.random.uniform(size=(1, output_neurons))

learning_rate = 0.1
epochs = 10000

# Loop used for iterative training/testing over dataset
for epoch in range(epochs):
    
    # ---- Forward Propagation ---
# Neuron computation: calculates weighted sum of inputs
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    
# Neuron computation: calculates weighted sum of inputs
    final_input = np.dot(hidden_output, W2) + b2
# Prediction phase: model outputs class based on learned weights
    predicted_output = sigmoid(final_input)
    
    # ---- Error Calculation ---
# Prediction phase: model outputs class based on learned weights
    error = y - predicted_output
    
    # ---- Backpropagation ---
# Prediction phase: model outputs class based on learned weights
    d_output = error * sigmoid_derivative(predicted_output)
    
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)
    
    # ---- Update Weights ---
    W2 += hidden_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Prediction phase: model outputs class based on learned weights
print("\nFinal Predicted Output:")
# Prediction phase: model outputs class based on learned weights
predicted_output

# Prediction phase: model outputs class based on learned weights
print("\nRounded Predictions (0 or 1):")
# Prediction phase: model outputs class based on learned weights
np.round(predicted_output).astype(int)

print("\nExpected Output:")
y

