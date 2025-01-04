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
Our dataset was created and prepared using the Roboflow tool, where the real classroom video recordings were segmented into frames, where the frame rate was one frame every three seconds. Roboflow is also utilized to assign multiple labels for each image during the image annotation process to feed the object detection models effectively. After cleaning and annotating, the dataset size was reduced to 3,025 images, automatically oriented, and resized to 224 × 224 pixels. 

The next step was to apply an image augmentation process to improve our models' performance and generalization ability, thus increasing their capacity to perform effectively on unseen images. To this end, we applied three types of augmentation resulting in 7,259 images: 
1. Rotation: 15-degree rotation
2. Saturation: 25% increase in saturation
3. Noise: 1.02% added noise.

The dataset was split into 70% for training, 15% for validation, and 15% for testing. There are 6369 training set images, 452 validating, and 450 tests, totaling 7271 images. Moreover, our dataset includes 31265 labels.

### Sample Data
![image](https://github.com/ArwaASM/In-classroom-Interaction-Detection/blob/main/Training%20Sample%20.png?raw=true)

![image](https://github.com/ArwaASM/In-classroom-Interaction-Detection/blob/main/Training%20Sample%202.png?raw=true)

![image](https://github.com/ArwaASM/In-classroom-Interaction-Detection/blob/main/Validating%20Sample.png?raw=true)

![image](https://github.com/ArwaASM/In-classroom-Interaction-Detection/blob/main/Testing%20Sample.png?raw=true)

## Installation

## Models Training

### YOLOv8 models

YOLOv8 (You Only Look Once, Version 8) is a state-of-the-art object detection algorithm known for its speed and accuracy. It builds on the YOLO family with a redesigned architecture that improves detection performance across various tasks. Key strengths include real-time detection capabilities, enhanced accuracy for small objects, and adaptability to diverse datasets. YOLOv8 features dynamic anchor-free detection, improved backbone networks, and advanced loss functions for better convergence. It supports tasks like object detection, instance segmentation, and pose estimation. Its lightweight design ensures efficiency on both edge devices and GPUs, making it versatile for real-world applications.

### Faster R-CNN Models

Faster R-CNN (Faster Region-Based Convolutional Neural Network) is a highly accurate two-stage object detection algorithm that combines a Region Proposal Network (RPN) for generating candidate regions with a classification head for precise detection. It excels at identifying small and overlapping objects while maintaining a balance between speed and accuracy. Detectron2, an open-source library developed by Facebook AI, implements Faster R-CNN and other advanced computer vision models. It is highly modular, scalable, and optimized for tasks like object detection, segmentation, and keypoint detection. Detectron2 supports custom datasets, extensive configuration options, and seamless integration with research and production workflows, making it a powerful tool for diverse applications.

### RetinaNet Models

RetinaNet, or Retinal Neural Network, is a one-stage object detection algorithm designed to address the imbalance between foreground and background classes using the Focal Loss function. This innovation enables RetinaNet to achieve high accuracy comparable to two-stage detectors like Faster R-CNN while maintaining faster inference speeds. It is particularly effective for detecting objects of varying scales and densities. Implemented in Detectron2, RetinaNet benefits from a modular and scalable framework, allowing customization for diverse datasets and applications. Detectron2 enhances RetinaNet with efficient training pipelines, seamless integration with GPU acceleration, and tools for evaluating performance, making it ideal for both research and real-world scenarios.

## Models Validating

## Models Testing (Inference)

## Comparison Results

The trained YOLOv8 models achieved the following mean average precisions (mAP) on the COCO 2017 validation set:

yolov8n: 0.61287
yolov8s: 0.56026
yolov8m: 0.59617
+++++++++++++++++
### Detection Scores







