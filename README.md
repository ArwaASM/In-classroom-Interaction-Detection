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

YOLOv8 (You Only Look Once, Version 8) is a state-of-the-art object detection algorithm known for its speed and accuracy. It builds on the YOLO family with a redesigned architecture that improves detection performance across various tasks. Key strengths include real-time detection capabilities, enhanced accuracy for small objects, and adaptability to diverse datasets. YOLOv8 features dynamic anchor-free detection, improved backbone networks, and advanced loss functions for better convergence. It supports tasks like object detection, instance segmentation, and pose estimation. Its lightweight design ensures efficiency on edge devices and GPUs, making it versatile for real-world applications.

#### Configuration

##### Install YOLOv8 

```
# Pip install method (recommended)

!pip install ultralytics==8.0.196

from IPython import display
display.clear_output()

import ultralytics
ultralytics.checks()

from ultralytics import YOLO

from IPython.display import display, Image

```

##### Install Roboflow  
```
!pip install roboflow --quiet

!pip install roboflow
```

##### Import the Dataset 
```
from roboflow import Roboflow
rf = Roboflow(api_key="Tp9HSxuOcXKVDZCz5***")
project = rf.workspace("arwa-almubarak-yiboc").project("students-behaviors-detection-wkavr")
version = project.version(4)
dataset = version.download("yolov8")

```
##### Model Training (YOLOv8x)   
```
!pip install -U albumentations
!pip install -U ultralytics

%cd /content

!yolo task=detect mode=train model=yolov8x.pt data=/content/datasets/students-behaviors-detection-4/data.yaml epochs=120 imgsz=800 plots=True
```
120 epochs completed in 4.345 hours.
Optimizer stripped from runs/detect/train2/weights/last.pt, 123.8MB
Optimizer stripped from runs/detect/train2/weights/best.pt, 123.8MB

Validating runs/detect/train2/weights/best.pt...
Ultralytics 8.3.9 🚀 Python-3.10.12 torch-2.4.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB, 40514MiB)
Model summary (fused): 286 layers, 61,606,161 parameters, 0 gradients, 226.8 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 15/15 [00:05<00:00,  2.76it/s]
                   all        450       1849       0.79      0.794      0.845      0.545
           Closed-Book        171        296      0.857      0.868      0.919       0.58
       Electronic-Book          5         12      0.993      0.833      0.932      0.486
               No-Book        167        312      0.809      0.875      0.876      0.559
           Opened-Book        272        619      0.881      0.874      0.917      0.601
          Raising-Hand        119        239      0.878      0.811      0.879      0.516
       Student-Answers         36         39      0.795      0.718      0.871      0.581
         Student-Reads         54         69      0.627      0.493      0.552      0.402
        Student-Writes         37         60      0.688      0.633        0.7      0.443
      Teacher-Explains         65         65      0.872      0.944      0.951      0.712
Teacher-Follows-up-Students          7          7      0.477      0.857       0.85      0.597
             Worksheet         84        131      0.809      0.832      0.844      0.517
Speed: 0.2ms preprocess, 5.0ms inference, 0.0ms loss, 2.9ms postprocess per image
Results saved to runs/detect/train2

##### Visualize Training Results   
```
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/confusion_matrix.png', width=600)!pip install -U ultralytics
```
![image](https://github.com/ArwaASM/In-classroom-Interaction-Detection/blob/main/Training%20Sample%20.png?raw=true)

```
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/results.png', width=600)
```
![image](https://github.com/ArwaASM/In-classroom-Interaction-Detection/blob/main/Training%20Sample%20.png?raw=true)

```
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train/val_batch0_pred.jpg', width=600)
```
![image](https://github.com/ArwaASM/In-classroom-Interaction-Detection/blob/main/Training%20Sample%20.png?raw=true)

##### Model Validation (YOLOv8x) 
```
%cd /content
!yolo task=detect mode=val model=/content/runs/detect/train2/weights/best.pt data=/content/datasets/students-behaviors-detection-4/data.yaml
```
/content
Ultralytics 8.3.9 🚀 Python-3.10.12 torch-2.4.1+cu121 CUDA:0 (NVIDIA A100-SXM4-40GB, 40514MiB)
Model summary (fused): 286 layers, 61,606,161 parameters, 0 gradients, 226.8 GFLOPs
val: Scanning /content/datasets/students-behaviors-detection-4/valid/labels.cache... 450 images, 15 backgrounds, 0 corrupt: 100% 450/450 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 29/29 [00:07<00:00,  3.93it/s]
                   all        450       1849       0.79      0.794      0.845      0.545
           Closed-Book        171        296      0.857      0.868      0.919      0.579
       Electronic-Book          5         12      0.995      0.833      0.932      0.487
               No-Book        167        312       0.81      0.874      0.876      0.559
           Opened-Book        272        619      0.881      0.873      0.917        0.6
          Raising-Hand        119        239      0.878       0.81      0.879      0.517
       Student-Answers         36         39      0.795      0.718      0.871      0.574
         Student-Reads         54         69      0.628      0.493      0.555      0.404
        Student-Writes         37         60      0.689      0.633      0.698      0.441
      Teacher-Explains         65         65      0.872      0.942       0.95      0.713
Teacher-Follows-up-Students          7          7      0.479      0.857       0.85      0.597
             Worksheet         84        131      0.809      0.832      0.844      0.519
Speed: 0.9ms preprocess, 8.3ms inference, 0.0ms loss, 3.5ms postprocess per image
Results saved to runs/detect/val
💡 Learn more at https://docs.ultralytics.com/modes/val

##### Model Inference (YOLOv8x) 
```
%cd /content
!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.5 source=/content/datasets/students-behaviors-detection-4/test/images save=True
```

##### Model Inference (YOLOv8x) 
```
%cd /content
!yolo task=detect mode=predict model=/content/runs/detect/train/weights/best.pt conf=0.5 source=/content/datasets/students-behaviors-detection-4/test/images save=True
```

##### Visualize Inference Results (YOLOv8x) 
```
import glob
from IPython.display import Image, display

# Define the base path where the folders are located
base_path = '/content/runs/detect/'

# List all directories that start with 'predict' in the base path
subfolders = [os.path.join(base_path, d) for d in os.listdir(base_path)
              if os.path.isdir(os.path.join(base_path, d)) and d.startswith('predict2')]

# Find the latest folder by modification time
latest_folder = max(subfolders, key=os.path.getmtime)

image_paths = glob.glob(f'{latest_folder}/*.jpg')[:5]

# Display each image
for image_path in image_paths:
    display(Image(filename=image_path, width=600))
    print("\n")
```

![image](https://github.com/ArwaASM/In-classroom-Interaction-Detection/blob/main/Training%20Sample%20.png?raw=true)

##### Deploy Model on Roboflow (YOLOv8x) 
```
project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")

import glob
from IPython.display import Image, display

# Define the base path where the folders are located
base_path = '/content/runs/detect/'

# List all directories that start with 'predict' in the base path
subfolders = [os.path.join(base_path, d) for d in os.listdir(base_path)
              if os.path.isdir(os.path.join(base_path, d)) and d.startswith('predict2')]

# Find the latest folder by modification time
latest_folder = max(subfolders, key=os.path.getmtime)

image_paths = glob.glob(f'{latest_folder}/*.jpg')[:5]

# Display each image
for image_path in image_paths:
    display(Image(filename=image_path, width=600))
    print("\n")
```

### Faster R-CNN Models

Faster R-CNN (Faster Region-Based Convolutional Neural Network) is a highly accurate two-stage object detection algorithm that combines a Region Proposal Network (RPN) for generating candidate regions with a classification head for precise detection. It excels at identifying small and overlapping objects while balancing speed and accuracy. Detectron2, an open-source library developed by Facebook AI, implements Faster R-CNN and other advanced computer vision models. It is highly modular, scalable, and optimized for object detection, segmentation, and keypoint detection. Detectron2 supports custom datasets, extensive configuration options, and seamless integration with research and production workflows, making it a powerful tool for diverse applications.

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







