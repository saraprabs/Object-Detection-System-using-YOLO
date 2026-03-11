from ultralytics import YOLO
from collections import Counter

# 1. Load your trained model
model = YOLO(r"C:\Users\Elev\YOLO_Object_Detection\notebooks\runs\detect\train\weights\best.pt")

# Run this to process a video file and save the result
results = model.predict(
    source="C:\\Users\\Elev\\YOLO_Object_Detection\\scripts\\detection_20260311-104100.avi",
    save=True, 
    conf=0.25,
    stream=True
)

total_counts = Counter()

print("🚀 Starting Video Processing...")

# 3. ONE loop to do everything
for i, r in enumerate(results):
    # Log frame progress every 50 frames so the console isn't too messy
    if i % 50 == 0:
        print(f"Processing frame {i}...")

    # Update total counts
    if len(r.boxes) > 0:
        for c in r.boxes.cls:
            class_name = model.names[int(c)]
            total_counts[class_name] += 1
            
        # Optional: Print detections for frames that HAVE objects
        print(f"Frame {i} - Detected: {[model.names[int(c)] for c in r.boxes.cls]}")

print("-" * 30)
print("✅ FINAL VIDEO RESULTS:")
if not total_counts:
    print("No objects were detected in this video.")
else:
    for obj, count in total_counts.items():
        print(f"{obj}: present in {count} total frames")

print(f"\nAnnotated video saved to: {r.save_dir}")