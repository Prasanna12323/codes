# Import required libraries for numerical computation and visualization
import numpy as np
# Import required libraries for numerical computation and visualization
import matplotlib.pyplot as plt


# 🔹 Perceptron Class
class Perceptron:
# Function definition: encapsulates a specific ANN operation
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        
        # initialize weights
        self.w1 = 0
        self.w2 = 0
        self.b = 0

    # 🔹 Activation Function
# Function definition: encapsulates a specific ANN operation
    def activation(self, z):
# Decision-making step: compares predicted vs actual output
        if z >= 0:
            return 1
        else:
            return -1

# Training logic: model learns patterns from data using iterative updates
    # 🔹 Training Function
# Function definition: encapsulates a specific ANN operation
    def fit(self, X, y):
# Loop for iterative processing (training/testing over dataset)
        for epoch in range(self.epochs):
# Loop for iterative processing (training/testing over dataset)
            for i in range(len(X)):
                
                x1 = X[i][0]
                x2 = X[i][1]
                target = y[i]
                
                # weighted sum
                z = (self.w1 * x1) + (self.w2 * x2) + self.b
                
# Prediction logic: uses trained model to classify input
                # prediction
                y_pred = self.activation(z)
                
# Decision-making step: compares predicted vs actual output
                # update if wrong
# Decision-making step: compares predicted vs actual output
                if y_pred != target:
                    self.w1 += self.lr * target * x1
                    self.w2 += self.lr * target * x2
                    self.b += self.lr * target

# Prediction logic: uses trained model to classify input
    # 🔹 Prediction Function
# Function definition: encapsulates a specific ANN operation
    def predict(self, x1, x2):
        z = (self.w1 * x1) + (self.w2 * x2) + self.b
        return self.activation(z)

# 🔹 Function to Plot Decision Regions
# Function definition: encapsulates a specific ANN operation
def plot_decision_boundary(model, X, y):
    
    x_values = np.linspace(0, 8, 100)
    y_values = np.linspace(0, 8, 100)
    
    xx, yy = np.meshgrid(x_values, y_values)

    #model equation
    Z = model.w1 * xx + model.w2 * yy + model.b
    Z = np.where(Z >= 0, 1, -1)
    
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # plot points
# Loop for iterative processing (training/testing over dataset)
    for i in range(len(X)):
# Decision-making step: compares predicted vs actual output
        if y[i] == 1:
# Decision-making step: compares predicted vs actual output
            plt.scatter(X[i][0], X[i][1], marker='o', label='Class 1' if i == 3 else "")
        else:
# Decision-making step: compares predicted vs actual output
            plt.scatter(X[i][0], X[i][1], marker='x', label='Class -1' if i == 0 else "")
    
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Perceptron Decision Regions")
    plt.legend()
    plt.show()

# 🔹 Dataset
X = np.array([
    [2, 3],
    [4, 5],
    [1, 1],
    [2, 1],
    [6, 7],
    [7, 8]
])

y = np.array([-1, -1, -1, 1, 1, 1])

# 🔹 Create Model
model = Perceptron(lr=0.1, epochs=100)

# Training logic: model learns patterns from data using iterative updates
# 🔹 Train Model
model.fit(X, y)

# 🔹 Print Results
print("Final Weights:", model.w1, model.w2)
print("Final Bias:", model.b)

# 🔹 Plot
plot_decision_boundary(model, X, y)

