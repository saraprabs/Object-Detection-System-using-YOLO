# This script uses a trained YOLO model to perform real-time object detection on webcam feed.
from ultralytics import YOLO
import cv2
from datetime import datetime # Import for unique filenames


# 1. Load your trained model
model = YOLO(r"C:\Users\Elev\YOLO_Object_Detection - Copy\notebooks\runs\detect\train\weights\best.pt")

# 2. Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# --- NEW: GENERATE UNIQUE FILENAME ---
# This creates a string like '20260311-104522' (YearMonthDay-HourMinSec)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
filename = f"detection_{timestamp}.avi"

# --- NEW: SET UP VIDEO WRITER ---
# Get webcam resolution to match the saved video size
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)

# Define the codec and create VideoWriter object
# 'XVID' is a common format that plays well on Windows
out = cv2.VideoWriter(filename, 
                         cv2.VideoWriter_fourcc(*'XVID'), 
                         10, size) # 10 is the FPS (frames per second)
# --------------------------------

print("Recording... Press 'q' to quit the webcam window")

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Run YOLO inference on the frame
        # 'conf=0.4' ignores weak detections to keep the screen clean
        results = model(frame, stream=True, conf=0.4)

        # Visualize the results on the frame
        for r in results:
            annotated_frame = r.plot()
        
        # --- NEW: WRITE THE FRAME TO FILE ---
        out.write(annotated_frame)
        # ------------------------------------

        # Display the output
        cv2.imshow("YOLO26 Real-Time Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release() # CRITICAL: If you don't release, the video file will be corrupted!
cv2.destroyAllWindows()
print("Video saved as output_detection.avi")