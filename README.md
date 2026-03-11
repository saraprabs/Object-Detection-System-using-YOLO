# 🎯 Object Detection System using YOLO 
The project is develop a real time object detection model using YOLO to detect everyday objects.  

## 📌 Project Overview  
The goal of the project is to specifically detect 3 classes objects `Red Cup`, `Blue bottle`, and `Mobile`.  
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

