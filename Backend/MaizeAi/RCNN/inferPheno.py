# %% [markdown]
# # Import libraries

# %%
import pandas as pd
import numpy as np
import cv2
import os
import copy
from torchvision import transforms as T
import json
import sys
import boto3
from botocore.exceptions import NoCredentialsError
from urllib.parse import quote_plus

from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

def upload_to_s3(local_path, s3_bucket, s3_key):
    try:
        # Initialize S3 client
        s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
        print(f"uploading file: {local_path} to s3://{s3_bucket}/{s3_key}")
        s3.upload_file(local_path, s3_bucket, s3_key)
        print(f"file uploaded to s3: s3://{s3_bucket}/{s3_key}")
    except NoCredentialsError:
        print("credentials not found")

def get_s3_url(s3_bucket, key, region='ap-southeast-1'):
    parts = key.split('/')
    encoded_parts = [quote_plus(part) for part in parts] #encode special characters eg '@'
    encoded_key = '/'.join(encoded_parts)
    return f"https://{s3_bucket}.s3.{region}.amazonaws.com/{encoded_key}"

# %% [markdown]
# # Load model
# 

# %%
# Load YOLO model, assume that the model is in the same directory
from ultralytics import YOLO

dirname = os.path.dirname(__file__)
model_detection = YOLO(os.path.join(dirname, 'detect.pt'))
model_segmentation = YOLO(os.path.join(dirname, 'segment.pt'))

user_email = sys.argv[3]
bucket = "maize-ai"

# %% [markdown]
# # Generate YOLO Prediction

# %%
def yolo_detect(input_directory, output_directory=os.path.join(dirname, 'output'), detection_threshold=0.5):
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Create "Count" folder in the output directory
    count_folder = os.path.join(output_directory, 'Count')
    os.makedirs(count_folder, exist_ok=True)

    bbox_path = os.path.join(output_directory, 'detection')
    os.makedirs(bbox_path, exist_ok=True)

    roi_folder = os.path.join(output_directory, 'ROI')
    os.makedirs(roi_folder, exist_ok=True)
    
    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Loop through all images
    for image_file in image_files:

        # Read image and extract some information
        img_id = image_file.split('.')[0] # Get image ID

        # Load image
        image_path = os.path.join(input_directory, image_file)
        img = cv2.imread(image_path)

        # Create a copy of the image for drawing bounding boxes
        img_copy = copy.deepcopy(img)

        # Generate bounding boxes
        results = model_detection(img, conf=detection_threshold)

        # Process results
        for r in results:
            # Extract bounding boxes in xyxy format
            bbox_list = r.boxes.xyxy
            count = len(bbox_list)  

            for i, bbox in enumerate(bbox_list):
                # Extract ROI
                roi = img_copy[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                # Save ROI
                roi_path = os.path.join(roi_folder, f'{img_id}_{i+1}.jpg')
                cv2.imwrite(roi_path, roi)
                # Draw bounding boxes
                img_detect = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 5)

            # Save image with bounding boxes
            detect_path = os.path.join(bbox_path, f'{img_id}_with_boxes.jpg')
            cv2.imwrite(detect_path, img_detect)

        # Save count
        count_filepath = os.path.join(count_folder, f'{img_id}.txt')
        with open(count_filepath, 'w') as f:
            f.write(str(count))

# Example use
yolo_detect(os.path.join(dirname, 'input'))

# %% [markdown]
# # Generate YOLO segmentation prediction

# %%
# Find the dominant color in the image and return it in HSV space
def get_dominant_color(region):
        # Convert the region to HSV color space
        region_hsv = cv2.cvtColor(region, cv2.COLOR_RGB2HSV)
        
        # Filter out black pixels
        non_black_pixels = (region_hsv[..., 2] != 0)
        region_without_black = region_hsv[non_black_pixels]
        non_black_pixels_rgb = np.all(region != [0, 0, 0], axis=-1)
        region_without_black_rgb = region[non_black_pixels_rgb]
        # Calculate the mean color in HSV space
        dominant_color_hsv = np.mean(region_without_black, axis=0)
        dominant_color_rgb = np.mean(region_without_black_rgb, axis=0).astype(int)
        return dominant_color_hsv, dominant_color_rgb

# Min-max normalization for a list of Hu moments
def min_max_normalize_hu_moments(hu_moments):
    # Perform min-max normalization for a list of Hu moments
    min_value = min(hu_moments)
    max_value = max(hu_moments)

    normalized_hu_moments = [(value - min_value) / (max_value - min_value) for value in hu_moments]

    return normalized_hu_moments

# Color outliers using IQR
def detect_outliers(data, col_indices, lower_bound_multipliers, upper_bound_multipliers):
    outliers = np.zeros(len(data), dtype=bool)

    for col_index, lower_multiplier, upper_multiplier in zip(col_indices, lower_bound_multipliers, upper_bound_multipliers):
        
        q1 = np.percentile(data[:, col_index], 25)
        q3 = np.percentile(data[:, col_index], 75)
        iqr = q3 - q1
        lower_bound = q1 - lower_multiplier * iqr
        upper_bound = q3 + upper_multiplier * iqr

        if col_index == 2:
            # Skip upper bound comparison for data[:, 2]
            outliers_col = (data[:, col_index] < lower_bound)
        else:
            # Perform both lower and upper bound comparisons for other columns
            outliers_col = (data[:, col_index] < lower_bound) | (data[:, col_index] > upper_bound)

        outliers |= outliers_col

    return outliers

# Shape outliers using IQR
def find_outliers_combined_iqr(hu_moments_list):
    outliers = np.zeros(len(hu_moments_list), dtype=bool)
    # Calculate the Interquartile Range (IQR)
    q1 = np.percentile(hu_moments_list, 25)
    q3 = np.percentile(hu_moments_list, 75)
    iqr_value = q3 - q1

    # Define a multiplier to determine the outlier threshold
    iqr_multiplier = 2.5 

    # Define the lower and upper bounds for outliers
    lower_bound = q1 - iqr_multiplier * iqr_value
    upper_bound = q3 + iqr_multiplier * iqr_value

    # Identify outliers based on the bounds
    outliers_col = (hu_moments_list < lower_bound) | (hu_moments_list > upper_bound)
    outliers |= outliers_col

    return outliers

# YOLO segmentation
def yolo_segment(input_directory, output_directory='output'):
    result_list = []
    # List all image files in the input directory
    image_files = [f for f in os.listdir(input_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    for image_file in image_files:
        img_id = image_file.split('.')[0]
        image_path = os.path.join(input_directory, image_file)
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        results = model_segmentation(img)
        for r in results:
            # Create a blank image
            binary_mask = np.zeros((height, width), dtype=np.uint8)

            # No mask detected
            if not r:
                continue
            
            # Get mask
            mask = r.masks.xy

            # Draw the mask on the blank image
            for points in mask:
                # Convert the points to integer and reshape to (num_points, 1, 2)
                points = points.astype(int).reshape((-1, 1, 2))
                
                # Fill the polygon in the blank image
                cv2.fillPoly(binary_mask, [points], color=255)

            
            # Cut mask region
            img_numpy = np.array(img)
            masked_image = cv2.bitwise_and(img_numpy, img_numpy, mask=binary_mask)
            masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            # Get dominant color info
            dominant_color_hsv, dominant_color_rgb = get_dominant_color(masked_image_rgb)
            
            # Get image moments
            hu_moments = cv2.HuMoments(cv2.moments(binary_mask)).flatten()
            hu_moments_normalized = min_max_normalize_hu_moments(hu_moments)
            hu_sum = np.sum(hu_moments_normalized)
            
            # Save result
            result_list.append({
                'filename': image_file,
                'dominant_color_hsv': dominant_color_hsv,
                'dominant_color_rgb': dominant_color_rgb,
                'hu_moments': hu_moments_normalized,
                'hu_moments_sum': hu_sum,
            })

    # Detect color outliers
    dominant_colors = np.array([entry['dominant_color_hsv'] for entry in result_list])
    col_indices = [0, 1, 2]
    lower_multipliers = [3, 3, 2]
    upper_multipliers = [3, 3, 0]
    color_outliers = detect_outliers(dominant_colors, col_indices, lower_multipliers, upper_multipliers)

    # Detect shape outliers
    hu_moments_idx = [entry['hu_moments_sum'] for entry in result_list]
    shape_outliers = find_outliers_combined_iqr(hu_moments_idx)

    # Save result list to JSON file for color outliers
    output_directory = "output"  
    os.makedirs(output_directory, exist_ok=True)
    analysis_folder = os.path.join(output_directory, 'Outliers')
    os.makedirs(analysis_folder, exist_ok=True)

    for i in range(len(result_list)):
        if color_outliers[i] or shape_outliers[i]:
            roi_path = os.path.join(output_directory, 'ROI', result_list[i]['filename'])
            s3_key = f"{user_email}/output/ROI/{result_list[i]['filename']}"
            upload_to_s3(roi_path, bucket, s3_key)
            s3_url = get_s3_url(bucket, s3_key)
            result_list[i]['s3_url'] = s3_url

            result_list[i]['dominant_color_hsv'] = result_list[i]['dominant_color_hsv'].tolist()
            result_list[i]['dominant_color_rgb'] = result_list[i]['dominant_color_rgb'].tolist()
            result_list[i]['color_diff'] = str(color_outliers[i])
            result_list[i]['shape_diff'] = str(shape_outliers[i])

            filename = result_list[i]['filename']
            json_filename = os.path.join(analysis_folder, f"{filename.split('.')[0]}.json")

            with open(json_filename, 'w') as json_file:
                json.dump(result_list[i], json_file, indent=4)
        
# Example use
yolo_segment(os.path.join(dirname, 'output', 'ROI'))

# %%



