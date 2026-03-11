# 🎯 Object Detection System using YOLO 
The project is developed to create a real-time object detection model using YOLO to detect everyday objects.  

## 📌 Project Overview  
The goal of the project is to specifically detect 3 object classes `Red Cup`, `Blue bottle`, and `Mobile`.  
| Class | Label |
|-------|-------|
|  Red Cup | `Red_cup` |
| Blue Bottle | `blue_bottle` |
| Mobile | `Mobile` |  


Following strategies were used to achieve the goal:  
- Label the objects in the collected image by drawing tight boxes using Roboflow
- Generate the dataset and export it into data.yml
- Feed it to the YOLO model for training
- Use the trained model for prediction

## 📸 Dataset Details
**Total Images** : ~457  
**Images per class** : ~140  
**Annotation tool** : Roboflow  
**Export format** : YOLOv8  
**Preprocessing** : auto-orientation, resize(640*640)  
**Augmentations** : Flip(horizontal)  
**Train / Val / Test split**: 87% / 11% / 2%  

## 📁 Project Structure
``` bash
Object-Detection-System-using-YOLO/
├── Model/
|   ├── Object-Detection-YOLO-2-DATA
│   │     ├── train/
│   │     │   ├── images/
│   │     │   └── labels/
│   │     ├── valid/
│   │     │   ├── images/
│   │     │   └── labels/
│   │     ├── test/
│   │     │   ├── images/
│   │     │   └── labels/
|   │     ├── data.yaml
|   │
|   ├── runs/
│   │    └── detect/
|   │          └──train/ 
│   │              └── weights/
│   │                ├── best.pt
│   │                └── last.pt
|   │
|   ├── training.ipynb
|
├── scripts/
|    ├── detect_realtime.py (using webcam)
|    ├── model_predict.py (for image)
└── README.md
```

# ⚙️ Installation & Setup

Follow these steps to run the project locally.

## 1️⃣ Clone the Repository
git clone https://github.com/saraprabs/Object-Detection-System-using-YOLO.git

cd Object-Detection-System-using-YOLO

## 2️⃣ Install Dependencies

pip install ultralytics opencv-python matplotlib roboflow

## 🛠️ Technologies Used

* Python
* Ultralytics YOLO
* OpenCV
* Roboflow
* Matplotlib

## 📝 Project Description

This repository contains a custom Deep Learning-based Object Detection System trained to identify and localize three everyday objects: Red Cups, Blue Bottles, and Mobile Phones.

The system utilizes the YOLO (You Only Look Once) architecture, specifically optimized for real-time inference on edge devices and standard CPUs. By training on a custom-curated dataset, the model achieves high precision in varied indoor environments, making it suitable for IoT integration, inventory tracking, or smart home applications.

## 🏗️ System Architecture & Functioning

1. **Data Pipeline: Preprocessing and Augmentation** All input images are auto-oriented and resized to 640 x640 pixels to maintain a consistent input layer for the neural network. To improve model generalization, Horizontal Flipping was applied, effectively doubling the training variety for symmetrical objects like bottles and cups.

2. **Training Logic:** Built on ultralytics YOLO engine.

3. **Inference:** Pulls data from a local webcam or video file via OpenCV. The .pt weights analyze the frame at a confidence threshold of 0.34. Non-Maximum Suppression (NMS) removes duplicate detections. Results are rendered in real-time and saved to .avi format with unique timestamps.

## 📊Model Performance

| Metric | Value | Interpretation |
| ---------- | ---------- | ------------ |
| mAP@50 | 67.3% | High reliability in standard detection tasks. |
| Precision | ~70% | Accurate identification with minimal false positives. |
| Recall | ~60% | Effective at finding objects even in cluttered frames. |
| F1-Score Peak | 0.65| Balanced performance at 0.338 confidence. |


## 📄 License

[![LICENSE: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

