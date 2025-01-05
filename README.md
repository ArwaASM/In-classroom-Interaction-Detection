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
![Training Sample ](https://github.com/user-attachments/assets/854ec8ca-1243-42d8-b1fd-aa81890b377d)
![Training Sample 2](https://github.com/user-attachments/assets/ba34f0d2-f41e-42e0-8ad4-9b0423999adc)
![Validating Sample](https://github.com/user-attachments/assets/54c65bab-fea4-4cc2-9e84-c5d1ac32719f)
![Testing Sample](https://github.com/user-attachments/assets/d6aec246-eec9-44c7-bd12-0fdf24fd1d37)

## Installation

## Models Training

### YOLOv8 models

YOLOv8 (You Only Look Once, Version 8) is a state-of-the-art object detection algorithm known for its speed and accuracy. It builds on the YOLO family with a redesigned architecture that improves detection performance across various tasks. Key strengths include real-time detection capabilities, enhanced accuracy for small objects, and adaptability to diverse datasets. YOLOv8 features dynamic anchor-free detection, improved backbone networks, and advanced loss functions for better convergence. It supports tasks like object detection, instance segmentation, and pose estimation. Its lightweight design ensures efficiency on edge devices and GPUs, making it versatile for real-world applications.

#### Configuration
#### YOLOv8x
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
![Results YOLOvx1](https://github.com/user-attachments/assets/4a0a0dc4-09ac-4b55-a429-fe71b2776f92)

##### Visualize Training Results   
```
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/confusion_matrix.png', width=600)!pip install -U ultralytics
```
![Figure8a_CM_ YOLOv8x](https://github.com/user-attachments/assets/09e6c208-ae1b-4766-98cc-f6aa776a6211)

```
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/results.png', width=600)
```
![results](https://github.com/user-attachments/assets/ad847446-b2e0-49e9-aecc-0dbdd4682b04)

##### Model Validation (YOLOv8x) 
```
%cd /content
!yolo task=detect mode=val model=/content/runs/detect/train2/weights/best.pt data=/content/datasets/students-behaviors-detection-4/data.yaml
```
![Val_YOLOv8x](https://github.com/user-attachments/assets/468b3972-f00c-447f-9632-5e5c65e781ac)

##### Model Inference (YOLOv8x) 
```
from ultralytics import YOLO
import os
import json

# Load YOLOv8 model (assuming yolov8x.pt or custom-trained weights path)
model_yolov8 = YOLO("/content/drive/MyDrive/AAbFFSSNew/runsYOLOv8x/detect/train2/weights/best.pt")

# Run predictions on the test folder and save images with labels
results = model_yolov8.predict(
    source="/content/students-behaviors-detection-3/test/images",  # Path to test images folder
    conf=0.5,  # Confidence threshold
    save=True,  # Save output images with predictions
    project="/content",  # Directory to save results
    name="YOLOv8XXtested_images_with_labels"  # Subfolder name
)

# Prepare data for JSON output
json_output = []
for result in results:
    image_info = {
        "image_name": os.path.basename(result.path),  # Image filename only
        "predictions": []
    }

    for box in result.boxes:
        prediction = {
            "class": int(box.cls),  # Class ID
            "confidence": float(box.conf),  # Confidence score
            "bbox": [float(coord) for coord in box.xyxy[0].tolist()]  # Bounding box [x1, y1, x2, y2]
        }
        image_info["predictions"].append(prediction)

    json_output.append(image_info)

# Save the JSON output
output_json_path = "/content/YOLOv8_predictions.json"
with open(output_json_path, "w") as json_file:
    json.dump(json_output, json_file, indent=4)

print(f"Labeled images saved in /content/YOLOv8XXtested_images_with_labels")
print(f"Predictions saved as JSON in {output_json_path}")
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
![Figure 10b_YOLOv8X](https://github.com/user-attachments/assets/8d2d3d8e-1216-450e-8a5f-03e888c88fdc)

##### Deploy Model on Roboflow (YOLOv8x) 
```
project.version(dataset.version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train2/")

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

#### YOLOv8l
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
##### Model Training (YOLOv8l)   
```
!pip install -U albumentations
!pip install -U ultralytics

%cd /content

!yolo task=detect mode=train model=yolov8l.pt data=/content/datasets/students-behaviors-detection-4/data.yaml epochs=120 imgsz=800 plots=True
```
![Results YOLOvl1](https://github.com/user-attachments/assets/bec67642-56c5-415c-9fbd-fa79ad630171)

##### Visualize Training Results   
```
%cd {HOME}
Image(filename=f'{HOME}/runs/detect/train2/results.png', width=600)
```
![results (1)](https://github.com/user-attachments/assets/4fcc42d2-86f5-46a0-b1dd-9a85ab359bbe)

##### Model Validation (YOLOv8l) 
```
%cd /content
!yolo task=detect mode=val model=/content/runs/detect/train2/weights/best.pt data=/content/datasets/students-behaviors-detection-4/data.yaml
```
![Val_YOLOv8l](https://github.com/user-attachments/assets/f9c749ba-cea0-4038-a041-a6819ac9ea7e)

##### Model Inference (YOLOv8l) 
```
from ultralytics import YOLO
import os
import json

# Load YOLOv8 model (assuming yolov8x.pt or custom-trained weights path)
model_yolov8 = YOLO("/content/drive/MyDrive/AAbFFSSNew/runsYOLOv8x/detect/train2/weights/best.pt")

# Run predictions on the test folder and save images with labels
results = model_yolov8.predict(
    source="/content/students-behaviors-detection-3/test/images",  # Path to test images folder
    conf=0.5,  # Confidence threshold
    save=True,  # Save output images with predictions
    project="/content",  # Directory to save results
    name="YOLOv8XXtested_images_with_labels"  # Subfolder name
)

# Prepare data for JSON output
json_output = []
for result in results:
    image_info = {
        "image_name": os.path.basename(result.path),  # Image filename only
        "predictions": []
    }

    for box in result.boxes:
        prediction = {
            "class": int(box.cls),  # Class ID
            "confidence": float(box.conf),  # Confidence score
            "bbox": [float(coord) for coord in box.xyxy[0].tolist()]  # Bounding box [x1, y1, x2, y2]
        }
        image_info["predictions"].append(prediction)

    json_output.append(image_info)

# Save the JSON output
output_json_path = "/content/YOLOv8_predictions.json"
with open(output_json_path, "w") as json_file:
    json.dump(json_output, json_file, indent=4)

print(f"Labeled images saved in /content/YOLOv8XXtested_images_with_labels")
print(f"Predictions saved as JSON in {output_json_path}")
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
![Figure 10c_YOLOv8L](https://github.com/user-attachments/assets/321e22fb-dddc-4866-8c5b-a04ad186b06d)

##### Deploy Model on Roboflow (YOLOv8l) 
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







