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

![Figure3](https://github.com/user-attachments/assets/7d3b8e6b-a8f7-4599-b583-8600f4b08b94)

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

1. Clone the repository:
```
git clone https://github.com/ArwaASM/In-classroom-Interaction-Detection
```
2. Install the required dependencies:
```
pip install -r requirements
```
3. Follow the Google Colab Links Table

| Notebook Name                               | Open in Colab                                   |
|---------------------------------------------|------------------------------------------------|
| YOLOv8x                                     | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArwaASM/TPE/blob/main/YOLOv8x_Training.ipynb) |
| YOLOv8l                                     | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArwaASM/TPE/blob/main/YOLOv8l_Training.ipynb) |
| Faster_RCNN_R_50 and RetinaNet_R_50         | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArwaASM/TPE/blob/main/Faster_RCNN_and_RetinaNet_R_50_Training.ipynb) |
| Faster_RCNN_R_101 and RetinaNet_R_101       | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArwaASM/TPE/blob/main/Faster_RCNN_and_RetinaNet_R_101_Training.ipynb) |
| Faster_RCNN_RX_101                          | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArwaASM/TPE/blob/main/Faster_RCNN_X_101_Training.ipynb) |
| YOLOv8 Evaluation                           | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArwaASM/TPE/blob/main/YOLOv8_Evaluation.ipynb) |
| Faster_RCNN Models Evaluation               | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArwaASM/TPE/blob/main/Models_Evaluation_and_Confusion_Matrices_(Faster_RCNN).ipynb) |
| RetinaNet Models Evaluation                 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArwaASM/TPE/blob/main/Models_Evaluation_and_Confusion_Matrices_(RetinaNet).ipynb) |


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

#### Configuration
#### Faster R-CNN_R_50
##### Install Detectron2 

```
!pip install pyyaml==5.1

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
# Install detectron2 that matches the above pytorch version
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
# !pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/TORCH_VERSION/index.html
# If there is not yet a detectron2 release that matches the given torch + CUDA version, you need to install a different pytorch.
!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
```

##### Install Roboflow  
```
!pip install roboflow --quiet

!pip install roboflow
```

##### Import the Dataset (COCO Format) 
```
!curl -L "https://app.roboflow.com/ds/aTaO5L3keM?key=lJEBGAngCR" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```
##### Model Training (Faster R-CNN_R_50)   
```
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_valid",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.01  # pick a good LR
cfg.SOLVER.MAX_ITER = 39850   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11  # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
```
![Results FRCNN_501](https://github.com/user-attachments/assets/9195d756-bde0-4780-ba65-c565167ad0dc)
![Results FRCNN_50_2](https://github.com/user-attachments/assets/cb2c675f-2971-4def-85dc-ca34e49833d6)

##### Visualize Training Results   

```
# Look at training curves in tensorboard:
%load_ext tensorboard
%tensorboard --logdir output
```
![Results FRCNN_50_3](https://github.com/user-attachments/assets/db08764a-dae4-40c5-ab55-b69cf36ca736)

##### Model Inference (YOLOv8x) 
```
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances
import torch
import numpy as np

# Define the Soft-NMS function
def soft_nms(boxes, scores, labels, iou_threshold=0.5, sigma=0.5, method="linear"):
    """
    Perform Soft-NMS on the given boxes, scores, and labels.
    Args:
        boxes (numpy.ndarray): Bounding boxes, shape (N, 4).
        scores (numpy.ndarray): Scores for each box, shape (N,).
        labels (numpy.ndarray): Labels for each box, shape (N,).
        iou_threshold (float): IoU threshold for NMS.
        sigma (float): Sigma for Gaussian Soft-NMS.
        method (str): "linear", "gaussian", or "hard" for Soft-NMS.
    Returns:
        keep (list): Indices of the boxes to keep.
    """
    N = len(scores)
    indices = np.arange(N)
    keep = []

    while len(indices) > 0:
        max_idx = np.argmax(scores[indices])
        current = indices[max_idx]
        keep.append(current)

        if len(indices) == 1:
            break

        current_box = boxes[current]
        remaining_boxes = boxes[indices]
        ious = compute_iou(current_box, remaining_boxes)

        if method == "linear":
            scores[indices] *= (1 - ious)
        elif method == "gaussian":
            scores[indices] *= np.exp(-(ious ** 2) / sigma)
        elif method == "hard":
            scores[indices][ious > iou_threshold] = 0

        indices = indices[scores[indices] > 0]

    return keep

def compute_iou(box, boxes):
    """
    Compute IoU between a box and a set of boxes.
    Args:
        box (numpy.ndarray): A single box, shape (4,).
        boxes (numpy.ndarray): A set of boxes, shape (N, 4).
    Returns:
        ious (numpy.ndarray): IoU values, shape (N,).
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union_area = box_area + boxes_area - inter_area
    ious = inter_area / np.maximum(union_area, 1e-6)
    return ious

def apply_soft_nms(predictions, iou_threshold=0.5, sigma=0.5, method="linear"):
    """
    Applies Soft-NMS to the predictions.
    Args:
        predictions (Instances): Model predictions.
        iou_threshold (float): IoU threshold for NMS.
        sigma (float): Sigma value for Gaussian Soft-NMS.
        method (str): "linear", "gaussian", or "hard" for Soft-NMS.
    Returns:
        Instances: Predictions after applying Soft-NMS.
    """
    boxes = predictions.pred_boxes.tensor.detach().cpu().numpy()
    scores = predictions.scores.detach().cpu().numpy()
    labels = predictions.pred_classes.detach().cpu().numpy()

    # Apply Soft-NMS
    keep = soft_nms(boxes, scores, labels, iou_threshold=iou_threshold, sigma=sigma, method=method)

    # Keep only the predictions after Soft-NMS
    predictions = predictions[keep]
    return predictions

# Load the model configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))  # replace with your config path
cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/AAbFFSSNew/Final_Results_14_11/Models_Training_and_Testing/output_FRCNN_50/model_final.pth"  # path to your trained model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for inference
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 11  # number of classes in your dataset

# Build model and predictor
model = build_model(cfg)
DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
predictor = DefaultPredictor(cfg)

# Initialize COCO Evaluator
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="/content/FRCNN_R_50_SNMS/")
evaluator._predictions = []  # Initialize _predictions attribute to avoid AttributeError
test_loader = build_detection_test_loader(cfg, "my_dataset_test")

# Run evaluation with Soft-NMS applied
def inference_with_soft_nms(predictor, data_loader, evaluator, iou_threshold=0.5, sigma=0.5, method="linear"):
    """
    Perform inference with Soft-NMS applied to predictions.
    """
    model = predictor.model
    model.eval()

    for idx, inputs in enumerate(data_loader):
        outputs = model(inputs)
        for output in outputs:
            output["instances"] = apply_soft_nms(output["instances"], iou_threshold, sigma, method)

        # Ensure tensors are detached before processing
        for output in outputs:
            output["instances"].pred_boxes.tensor = output["instances"].pred_boxes.tensor.detach()

        evaluator.process(inputs, outputs)

    return evaluator.evaluate()

# Perform evaluation
eval_results = inference_with_soft_nms(predictor, test_loader, evaluator, iou_threshold=0.5, sigma=0.5, method="linear")
print("Evaluation Results with Soft-NMS:\n", eval_results)
```
![Results FRCNN_50_4](https://github.com/user-attachments/assets/e149fb98-5d5d-4314-8752-326867bf7e9c)

##### Visualize Inference Results (Faster R-CNN) 
![Figure 10d_FRCNN50](https://github.com/user-attachments/assets/4ce46234-7186-4b51-9077-d5c0eaeaed4d)


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







