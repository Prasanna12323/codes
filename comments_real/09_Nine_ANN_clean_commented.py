# Import libraries required for ANN computation and visualization
import numpy as np

patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

class HopfieldNetwork:
    
# Function definition: encapsulates ANN logic (training/prediction)
    def __init__(self):
        self.W = None   # weight matrix

# Training stage: model updates weights based on data
    # 🔹 Train using Hebbian Learning
# Function definition: encapsulates ANN logic (training/prediction)
    def train(self, patterns):
        n = patterns.shape[1]
        self.W = np.zeros((n, n))
        
        for p in patterns:
            p = p.reshape(n, 1)
            self.W += p @ p.T   # outer product
        
        # remove self connections
        np.fill_diagonal(self.W, 0)

    # 🔹 Recall function
# Function definition: encapsulates ANN logic (training/prediction)
    def recall(self, x, steps=5):
        x = x.copy()
        
# Iterative loop: used for training epochs or dataset traversal
        for _ in range(steps):
            x = np.sign(self.W @ x)
        
        return x

model = HopfieldNetwork()
# Training stage: model updates weights based on data
model.train(patterns)

print("Weight Matrix:\n", model.W)

# slightly corrupted pattern
test = np.array([1, -1, 1, -1])

print("Input:", test)
print("\n")

output = model.recall(test)

print("Recovered Output:", output)



