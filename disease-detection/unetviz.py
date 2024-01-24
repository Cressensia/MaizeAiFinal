import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
#from torch.utils.data import DataLoader, Dataset
from unetmodel import UNet

#from pycocotools.coco import COCO
from PIL import Image, ImageChops, ImageDraw
from torchvision.transforms import ToPILImage

num_classes = 4

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

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

device = torch.device('cpu')

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

    return predicted_mask.cpu(), predicted_disease  # Move data back to CPU for further processing


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load and preprocess image (need to change)
#img_dir =
img_path = 'C:\\Users\\Jernis\\Documents\\FYP\\maize leaf - disease.v10i.coco-segmentation\\test\\spots.png'
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img)

# Predict the mask
model = UNet().to(device)
model.load_state_dict(torch.load("unet_best.pth", map_location=device))

# Predict the mask and disease
predicted_mask, predicted_disease = predict(img_tensor, model)

# Map the predicted disease index to the disease name
disease_mapping = {0: "healthy", 1: "maize-blight", 2: "maize-common-rust", 3: "maize-leaf-spot"}
disease_name = disease_mapping.get(predicted_disease, "Unknown")

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
    

# Create and display the prediction image
prediction_image = create_prediction_image(img_tensor, predicted_mask, disease_name)
prediction_image.save("prediction_output_test.png")  # Save the image
prediction_image.show()  # Display the image