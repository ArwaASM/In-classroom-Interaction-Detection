# How to Train YOLOv8, Faster-RCNN, and RetinaNet Object Detection Models on our Custom Dataset to Detect In-classroom Interactions

## Introduction
This project uses three object detection algorithms: YOLOv8, Faster-RCNN, and RetinaNet to detect four main in-classroom interactions in video frames. This includes utilizing seven models: two versions of YOLOv8, three Faster-RCNN, and Two Retinanet. The code provides three main components: handling the dataset prepared by Roboflow, training the seven models, and evaluating them. A comparison of these models will ultimately be conducted to determine the most robust model.  
## Features
- Imported two versions (YOLOv8 and COCO formats) of our custom In-classroom Interactions dataset prepared using Roboflow that contains 7271 images.

- Trained YOLOv8  (YOLOv8l, YOLOv8x) on the dataset with YOLOv8 format, Faster R-CNN (R_50, R_101, and RX_101), and RetinaNet (R_50, R_101) models on the dataset with COCO format.

- Evaluated the classroom video recordings using the trained models to detect: 
1. Using Textbook: Opened Book, Closed Book, Electronic Book, No Book, and Worksheet.
2. Participation: Raising Hands and Answering.
3. Teacher Activities: Teacher Follows up Students and Teacher Explains.
4. Student Activities: Student Reads and Student Writes.

- Saved the testing dataset with bounding boxes and confidence scores.

- Compared the models' accuracy, and selected the most accurate one, to be optimized later. 
## Prerequisites
- Ultralytics 8.0.196 YOLO Library
- Roboflow Library
- Json Library
- pyyaml 5.1 Library
- Torch Library
- Detectron2 Library
- Python Library
- Numpy Library
- cv2 Library
- Random Library
- Detectron2 Utilities
- COCO Library
- COCOeval Library
- sklearn.metrics Library
- matplotlib.pyplot Library
## Dataset Preparation
### Sample Data

## Installation

## Models Training

### YOLOv8 models

### Faster R-CNN Models

### RetinaNet Models

## Models Validating

## Models Testing (Inference)

## Comparison Results

The trained YOLOv8 models achieved the following mean average precisions (mAP) on the COCO 2017 validation set:

yolov8n: 0.61287
yolov8s: 0.56026
yolov8m: 0.59617
+++++++++++++++++
### Detection Scores







