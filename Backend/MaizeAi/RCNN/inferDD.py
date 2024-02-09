# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import ToPILImage
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageChops, ImageDraw
import os
import json
from pycocotools.coco import COCO
import numpy as np
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
import time

# %%
num_classes = 4 # 3 diseases rn + healthy class

#class names must be same as roboflow (i think category_id)
category_to_disease = {
    1: "healthy",
    2: "maize-blight",
    3: "maize-common-rust",
    4: "maize-leaf-spot"
    # ... and so on for each category_id
}

#mapping dict
disease_to_id = {
    "healthy": 0,
    "maize-blight": 1,
    "maize-common-rust": 2,
    "maize-leaf-spot": 3

    # Add other diseases here if needed
}

# Inverse mapping of disease_to_id
disease_mapping = {v: k for k, v in disease_to_id.items()}

# %%
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        '''
        # Encoder (Contracting Path)
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle part (Bottleneck)
        self.middle_conv1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.middle_bn1 = nn.BatchNorm2d(64)
        self.middle_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.middle_bn2 = nn.BatchNorm2d(64)

        # Decoder (Expansive Path)
        self.up_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(32)

        # Dropout for regularization - Adjusted dropout rate
        self.dropout = nn.Dropout(0.3)

        # Final convolution
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

        # Classification layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        '''
        #reduced filters - slightly faster
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle part (Bottleneck)
        self.middle_conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.middle_bn1 = nn.BatchNorm2d(32)
        self.middle_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.middle_bn2 = nn.BatchNorm2d(32)

        # Decoder
        self.up_conv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(16)
        self.dec_conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)

        # Keep the dropout rate the same for now
        self.dropout = nn.Dropout(0.3)

        # Final convolution
        self.final_conv = nn.Conv2d(16, 1, kernel_size=1)

        # Classification layers - Adjusted to match the new filter sizes
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, num_classes)
        

    def forward(self, x):
        # Encoder
        e1 = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)))
        e1 = F.leaky_relu(self.enc_bn2(self.enc_conv2(e1)))
        p1 = self.pool1(e1)

        # Middle part
        m = F.leaky_relu(self.middle_bn1(self.middle_conv1(p1)))
        m = F.leaky_relu(self.middle_bn2(self.middle_conv2(m)))

        # Decoder
        d1 = self.up_conv1(m)
        # Resize d1 to match the size of p1 before concatenation
        d1 = F.interpolate(d1, size=(p1.size(2), p1.size(3)), mode='bilinear', align_corners=True)
        d1 = torch.cat((p1, d1), dim=1)
        d1 = F.leaky_relu(self.dec_bn1(self.dec_conv1(d1)))
        d1 = F.leaky_relu(self.dec_bn2(self.dec_conv2(d1)))
        d1 = self.dropout(d1)

        out = self.final_conv(d1)

        # Resize the output to match the input size if necessary
        out = F.interpolate(out, size=(256, 256), mode='bilinear', align_corners=True)

        # Classification branch
        class_features = self.global_avg_pool(m)
        class_features = class_features.view(class_features.size(0), -1)
        class_features = F.relu(self.fc1(class_features))
        class_output = self.fc2(class_features)

        return out, class_output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')  # He initialization
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# Function to predict the segmentation mask
def predict(img_tensor, model):
    # Ensure the model is in evaluation mode
    model.eval()

    # Assuming img_tensor is already preprocessed and ready for model input
    # If img_tensor is a single image, add a batch dimension
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Forward pass
    with torch.no_grad():
        mask_output, class_output = model(img_tensor)

        # Process outputs
        _, predicted_class = torch.max(class_output, 1)
        predicted_disease = predicted_class.item()
        predicted_mask = torch.sigmoid(mask_output[0]).float()  # First item in batch
        predicted_mask = (predicted_mask > 0.5).float()  # Binarize mask

    return predicted_mask.cpu(), predicted_disease

# Function to create prediction image
def create_prediction_image(img_tensor, predicted_mask, disease_name):
    # Convert tensor to PIL image
    img_pil = ToPILImage()(img_tensor.cpu()).convert("RGB")

    # Only add mask if disease is detected
    if disease_name != "healthy":
        mask_pil = ToPILImage()(predicted_mask.cpu().squeeze()).convert("L")
        mask_pil = mask_pil.resize(img_pil.size)
        mask_color = Image.new("RGB", mask_pil.size, (255, 0, 0))
        mask_pil_colored = ImageChops.multiply(mask_color, mask_pil.convert("RGB"))
        img_with_mask = ImageChops.add(img_pil, mask_pil_colored)
        composite_image = Image.new('RGB', (img_pil.width * 2, img_pil.height))
        composite_image.paste(img_pil, (0, 0))
        composite_image.paste(img_with_mask, (img_pil.width, 0))
    else:
        composite_image = img_pil

    # Add text to the composite image
    draw = ImageDraw.Draw(composite_image)
    if disease_name != "healthy":
        draw.text((10, 10), "Original Image:", fill="black")
        draw.text((img_pil.width + 10, 10), f"Predicted Disease: {disease_name}", fill="black")
    else:
        draw.text((10, 10), "No Mask for Healthy Class. Leaf Inputted is Healthy.", fill="white")

    return composite_image

# %%
# Load Unet model
device = torch.device('cpu')
model = UNet().to(device)
model.load_state_dict(torch.load("/Users/cressensia/Downloads/Maize-AI/Backend/MaizeAi/RCNN/unet_best.pth", map_location='cpu'))

# %%
import os
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_tensor, to_pil_image

# def process_image_and_generate_prediction(input_image_path, output_dir='RCNN/outputDD', model=None):
#     # Create output directory if it doesn't exist
#     output_dir = os.path.join('MaizeAi', output_dir)
#     os.makedirs(output_dir, exist_ok=True)

#     # Load image
#     img = Image.open(input_image_path).convert('RGB')
#     img_tensor = to_tensor(img)

#     # Generate model prediction
#     predicted_mask, predicted_disease = predict(img_tensor, model)

#     # Map predicted disease index to disease name
#     disease_mapping = {0: "healthy", 1: "maize-blight", 2: "maize-common-rust", 3: "maize-leaf-spot"}
#     disease_name = disease_mapping.get(predicted_disease, "Unknown")

#     # Create prediction image
#     prediction_image = create_prediction_image(img_tensor, predicted_mask, disease_name)

#   # Save prediction image (prediction_<input image name>)
#     output_path = os.path.join(output_dir, f"prediction_{os.path.basename(input_image_path)}")
#     prediction_image.save(output_path)
    
#     # Return the path to the output image
#     return output_path, disease_name

def process_image_and_generate_prediction(input_image_path, output_dir='RCNN/outputDD'):
    # Create output directory if it doesn't exist
    output_dir = os.path.join('MaizeAi', output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = Image.open(input_image_path).convert('RGB')
    img_tensor = to_tensor(img)

    # Generate model prediction
    predicted_mask, predicted_disease = predict(img_tensor, model)  # Use the globally loaded model

    # Map predicted disease index to disease name
    disease_mapping = {0: "healthy", 1: "maize-blight", 2: "maize-common-rust", 3: "maize-leaf-spot"}
    disease_name = disease_mapping.get(predicted_disease, "Unknown")

    # Create prediction image
    prediction_image = create_prediction_image(img_tensor, predicted_mask, disease_name)

    # Save prediction image (prediction_<input image name>)
    output_path = os.path.join(output_dir, f"prediction_{os.path.basename(input_image_path)}")
    prediction_image.save(output_path)
    
    # Return the path to the output image
    return output_path, disease_name

# %%
# Example use
if __name__ == "__main__":
    process_image_and_generate_prediction('test.jpg', model=model)


