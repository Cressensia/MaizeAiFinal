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

from matplotlib import pyplot as plt
from torchvision import transforms as T
from xml.etree import ElementTree as et

# %%
def get_model(num_classes):
    pretrained_base_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # print(pretrained_base_model)

    in_features = pretrained_base_model.roi_heads.box_predictor.cls_score.in_features
    pretrained_base_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    mask_in_channels = pretrained_base_model.roi_heads.mask_predictor.conv5_mask.in_channels
    pretrained_base_model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in_channels, 256, num_classes)
    return pretrained_base_model


# %%
num_classes = 2  # the background class and the pedestrian class
model = get_model(num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load("model.pth", map_location=torch.device('cpu'))
model.to(device)

# %%
model.eval()
img = Image.open("T0001_XM_20110811140245_01_16.jpg")
resized_img = img.resize((256, 256))
transform = T.ToTensor()
ig = transform(resized_img)
with torch.no_grad():
    pred = model([ig.to(device)])

# %%
mask = (pred[0]["masks"][0].cpu().detach().numpy() * 255).astype("uint8").squeeze()

# %%
plt.imshow(mask)


