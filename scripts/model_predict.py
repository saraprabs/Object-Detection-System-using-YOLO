# Use trained model to make predictions on an image and plot the results
import os
import matplotlib.pyplot as plt
import cv2

from ultralytics import YOLO

# Define the path clearly
model_path = r"C:\Users\Elev\YOLO_Object_Detection - Copy\notebooks\runs\detect\train\weights\best.pt"

# Verification: Check if the file actually exists before loading
if os.path.exists(model_path):
    print("✅ Success: File found. Loading model now...")
    my_model = YOLO(model_path)
    print("✅ Model loaded successfully!")
else:
    print("❌ Error: File NOT found at that path. Please double-check the folder name.")


# Run inference on an image
results = my_model.predict(source="C:/Users/Elev/Pictures/Red cup/istockphoto-1157949372-612x612.jpg", conf=0.25, save=True)

# 3. Plot directly in the notebook
for result in results:
    # Get the image with plotted boxes as a BGR numpy array
    im_array = result.plot() 
    # Convert BGR to RGB for matplotlib
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(im_rgb)
    plt.axis('off')
    plt.show()