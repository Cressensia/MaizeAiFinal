# %%
# Load model
# Faster RCNN prediction
# Extract ROI
# Mask RCNN prediction
# Extract color information
# 3D scatter plot

# %% [markdown]
# # Load model
# 

# %%
import pandas as pd
import numpy as np
import cv2
import os
import re
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn as nn
import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms as T

DIR = f'E:\\Maize-AI\\Backend\\MaizeAi\\RCNN\\'

# %%
device = torch.device('cpu')
# Load faster rcnn model
fasterrcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = fasterrcnn.roi_heads.box_predictor.cls_score.in_features
fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
file_path = f'{DIR}fasterrcnn_phase1.pth'
fasterrcnn.load_state_dict(torch.load(file_path, map_location=device))
fasterrcnn = fasterrcnn.to(device)

# %% [markdown]
# # Generate Faster RCNN Prediction

# %%
# Generate faster RCNN prediction
def predict(model, images):
    model.eval()
    images = list(image.to(device) for image in images)
    outputs = model(images)
    return outputs

# Draw bounding box on processed image (1024x1024)
def draw_boxes_on_image(boxes, images):
    for box in boxes:
        cv2.rectangle(images,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 3)
    return images

# Extract maize tassel image from bounding box
def extract_roi(img, img_id, boxes, output_directory):
    roi_folder = os.path.join(output_directory, f'{DIR}output' + '\ROI')
    os.makedirs(roi_folder, exist_ok=True)
    for i, box in enumerate(boxes):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        # Extract the region of interest (ROI)
        roi = img[y1:y2, x1:x2]

        # Save the ROI
        output_path = os.path.join(roi_folder, f'{img_id}_{i+1}.jpg')
        cv2.imwrite(output_path, roi)

# Resize bounding box coordinates to original image size
def resize_bbox(original_width, original_height, boxes):
    for box in boxes:

        # Extract coordinates
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        # Calculate scale factor
        width_scale = original_width / 1024
        height_scale = original_height / 1024

        # Calculate new coordinates
        resized_x1 = int(x1 * width_scale)
        resized_y1 = int(y1 * height_scale)
        resized_x2 = int(x2 * width_scale)
        resized_y2 = int(y2 * height_scale)

        # Assign new coordinates
        box[0] = resized_x1
        box[1] = resized_y1
        box[2] = resized_x2
        box[3] = resized_y2
        
    return boxes

# faster rcnn inference
def process_images_and_predict(input_directory, output_directory=f'{DIR}output', detection_threshold=0.6):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create "Count" folder in the output directory
    count_folder = os.path.join(output_directory, f'{DIR}output' + '\Count')
    os.makedirs(count_folder, exist_ok=True)

    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for img_name in image_files:
        
        # Read and preprocess image
        img_id = img_name.split('.')[0]
        image_path = os.path.join(input_directory, img_name)
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (1024, 1024), cv2.INTER_AREA)
        img_res /= 255.0

        # Generate Faster RCNN prediction
        output = predict(fasterrcnn, [torch.tensor(img_res, dtype=torch.float32).permute(2, 0, 1).to(device)])
        prediction_boxes = output[0]['boxes'].data.cpu().numpy()
        scores = output[0]['scores'].data.cpu().numpy()
        count = len(prediction_boxes)
        # Filter boxes based on detection threshold
        prediction_boxes = prediction_boxes[scores >= detection_threshold].astype(np.int32)

        # Resize bounding boxes to original image size
        prediction_boxes_resized = resize_bbox(img.shape[1], img.shape[0], prediction_boxes)

        # Draw bounding box on image and save
        img_with_boxes = draw_boxes_on_image(prediction_boxes_resized, img_rgb)
        bbox_path = os.path.join(output_directory, f'{DIR}output' + '\detection')
        os.makedirs(bbox_path, exist_ok=True)
        output_path = os.path.join(bbox_path, f'{img_id}_with_boxes.jpg')
        cv2.imwrite(output_path, cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR))

        # Run extract ROI for image
        extract_roi(img, img_id, prediction_boxes_resized, output_directory)

        # Save count to text file in "Count" folder
        count_filepath = os.path.join(count_folder, f'{img_id}.txt')
        with open(count_filepath, 'w') as f:
            f.write(str(count))

# Call the function with the input directory
input_directory = f'{DIR}input'
process_images_and_predict(input_directory)


