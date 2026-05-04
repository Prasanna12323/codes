# Import necessary libraries for ANN and numerical operations
import numpy as np

# 🔹 ART1 Class
class ART1:
# Function block: defines reusable ANN logic (training or prediction)
    def __init__(self, vigilance=0.6):
        self.vigilance = vigilance
        self.clusters = []

    # 🔹 Similarity Function
# Function block: defines reusable ANN logic (training or prediction)
    def similarity(self, x, y):
        return np.sum(np.minimum(x, y)) / np.sum(x)

# Training process: model learns patterns by updating weights
    # 🔹 Training Function
# Function block: defines reusable ANN logic (training or prediction)
    def fit(self, data):
# Loop: used for iterations over data or epochs
        for i in range(len(data)):
            pattern = data[i]
            assigned = False
            
            # Compare with existing clusters
# Loop: used for iterations over data or epochs
            for j in range(len(self.clusters)):
                sim = self.similarity(pattern, self.clusters[j])
                
# Condition check: determines updates or classification decisions
                if sim >= self.vigilance:
                    # Update cluster (intersection)
                    self.clusters[j] = np.minimum(self.clusters[j], pattern)
                    assigned = True
                    break
            
            # If no match → create new cluster
# Condition check: determines updates or classification decisions
            if not assigned:
                self.clusters.append(pattern.copy())

    # 🔹 Display Clusters
# Function block: defines reusable ANN logic (training or prediction)
    def show_clusters(self):
        print("Clusters formed:\n")
        for i, c in enumerate(self.clusters):
            print(f"Cluster {i+1}: {c}")

# 🔹 Input Data
data = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [0, 0, 1, 1],
    [0, 0, 1, 0]
])

# 🔹 Create Model
model = ART1(vigilance=0.6)

# Training process: model learns patterns by updating weights
# 🔹 Train
model.fit(data)

# 🔹 Output
model.show_clusters()

