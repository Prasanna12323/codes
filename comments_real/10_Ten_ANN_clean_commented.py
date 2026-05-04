# Import libraries required for ANN computation and data handling
import cv2 
# Import libraries required for ANN computation and data handling
import matplotlib.pyplot as plt 
# Import libraries required for ANN computation and data handling
from ultralytics import YOLO 
# Import libraries required for ANN computation and data handling
from collections import Counter 
# Load YOLO model 

model = YOLO("yolov8n.pt") 
print(" Model Loaded") 


# Image path (change accordingly) 
image_path = "./Image.jpg" 


# Read image 
img = cv2.imread(image_path) 

# Conditional logic: checks prediction correctness or update condition
if img is None: 
    print(" Image NOT found. Fix path!") 
else: 
    print(" Image Loaded Successfully") 


# Run detection 
results = model(image_path) 
print(" Detection Done") 


# Extract detected objects 
detected_objects = [] 
for r in results: 
    for box in r.boxes: 
        cls = int(box.cls[0]) 
        label = model.names[cls] 
        detected_objects.append(label) 

print("Detected Objects:", detected_objects) 

# Convert BGR to RGB for display 
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

# Show image 
plt.imshow(img) 
plt.title("Original Image") 
plt.axis("off") 
plt.show() 



