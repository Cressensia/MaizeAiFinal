import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageChops
import os
import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

#unet model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # Encoder (Contracting Path)
        self.enc_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(64)  # Batch Normalization
        self.enc_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc_bn2 = nn.BatchNorm2d(64)  # Batch Normalization
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # More layers add here for depth

        # Middle part (Bottleneck)
        self.middle_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.middle_bn1 = nn.BatchNorm2d(128)  # Batch Normalization
        self.middle_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.middle_bn2 = nn.BatchNorm2d(128)  # Batch Normalization

        # Decoder (Expansive Path)
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(64)  # Batch Normalization
        self.dec_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(64)  # Batch Normalization

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

        # Final convolution
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1) #change here if multiclass cause now binary (disease vs no disease)

        # Classification layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

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

def load_disease_types(annotation_file, disease_to_id):
    with open(annotation_file, 'r') as file:
        data = json.load(file)

    # Initialize a dictionary to store the disease types for each image
    disease_types = {img_id: [0] * len(disease_to_id) for img_id in range(len(data['images']))}

    # Iterate over each annotation
    for ann in data['annotations']:
        img_id = ann['image_id']
        cat_id = ann['category_id']
        disease_name = category_to_disease.get(cat_id, "Unknown")

        # If the disease is known, set the corresponding index to 1
        if disease_name in disease_to_id:
            disease_index = disease_to_id[disease_name]
            disease_types[img_id][disease_index] = 1

    # Mark images with no annotations as 'healthy'
    for img_id in disease_types:
        if sum(disease_types[img_id]) == 0:
            disease_types[img_id][disease_to_id['healthy']] = 1

    return disease_types

#MaizeDataset Class
class MaizeDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, disease_to_id=None):
        self.image_dir = image_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Load disease types
        if disease_to_id is not None:
            self.disease_types = load_disease_types(annotation_file, disease_to_id)
        else:
            # Handle the case where disease_to_id is not provided
            self.disease_types = {}

        # Map from disease name to class index
        self.disease_to_id = disease_to_id

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.image_dir, path)).convert('RGB')
        mask = Image.new('L', (image.width, image.height))

        for ann in annotations:
            ann_mask = Image.fromarray(coco.annToMask(ann) * 255).convert('L')
            mask = ImageChops.add(mask, ann_mask)

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        # Get disease types (multi-label vector) for the image
        class_label = torch.tensor(self.disease_types[img_id], dtype=torch.float)

        return image, mask, class_label

    def __len__(self):
        return len(self.ids)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define a function for Dice Loss - overlap between the predicted and target segmentation masks
class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# Define a combined loss function
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets):
        return self.bce_loss(inputs, targets) + self.dice_loss(inputs, targets)


# Function to process dataset files
def process_dataset_files(image_dir):
    image_files = []
    annotation_file = None

    # Check if directory exists
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found: {image_dir}")

    # Iterate through files in directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(image_dir, filename))
        elif filename.lower().endswith('.json'):
            annotation_file = os.path.join(image_dir, filename)

    # Check if annotation file is found
    if annotation_file is None:
        raise FileNotFoundError("No JSON annotation file found in the directory")

    return image_files, annotation_file

# Function to create dataset and return annotation file path
def create_dataset(image_dir, transform, disease_to_id):
    try:
        image_files, annotation_file = process_dataset_files(image_dir)
        dataset = MaizeDataset(image_dir=image_dir, 
                               annotation_file=annotation_file, 
                               transform=transform,
                               disease_to_id=disease_to_id)
        return dataset, annotation_file
    except Exception as e:
        print(f"Error creating dataset: {e}")
        return None, None

# Create the training and validation datasets
train_dataset, train_annotation_file = create_dataset('C:\\Users\\Jernis\\Documents\\FYP\\maize leaf - disease.v7i.coco-segmentation\\train', 
                                                      transform, 
                                                      disease_to_id)

val_dataset, val_annotation_file = create_dataset('C:\\Users\\Jernis\\Documents\\FYP\\maize leaf - disease.v7i.coco-segmentation\\valid', 
                                                  transform, 
                                                  disease_to_id)

# Create the DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True) if train_dataset else None
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_dataset else None

# Set disease types for training dataset
if train_dataset and train_annotation_file:
    train_dataset.disease_types = load_disease_types(train_annotation_file, disease_to_id)

#training
def iou_score(output, target):
    with torch.no_grad():
        output = (output > 0.5).float()
        target = (target > 0.5).float()
        intersection = (output * target).sum((1, 2))  # Logical AND -> use multiplication for float tensors
        union = (output + target).clamp(0, 1).sum((1, 2))  # Logical OR -> use addition and clamp for float tensors
        iou = (intersection + 1e-6) / (union + 1e-6)
        return iou.mean()
    
# Create the U-Net model instance and initialize weights
model = UNet()

# Define the loss function for segmentation
segmentation_criterion = nn.BCEWithLogitsLoss()
# Define the loss function for classification
classification_criterion = nn.BCEWithLogitsLoss()

def validate_model(model, val_loader, segmentation_criterion, classification_criterion):
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    val_corrects = 0
    val_total_pixels = 0
    val_iou_sum = 0.0

    with torch.no_grad():
        for inputs, masks, class_labels in val_loader:
            valid_indices = class_labels != -1
            if not valid_indices.any():
                continue

            valid_indices = (class_labels != -1).any(dim=1)

            valid_inputs = inputs[valid_indices]
            valid_masks = masks[valid_indices]
            valid_class_labels = class_labels[valid_indices]

            # Forward pass
            segmentation_output, classification_output = model(valid_inputs)

            # Calculate losses
            segmentation_val_loss = segmentation_criterion(segmentation_output, valid_masks)
            if len(valid_class_labels) > 0:
                classification_val_loss = classification_criterion(classification_output, valid_class_labels)
            else:
                classification_val_loss = 0

            # Combine losses
            total_val_loss = segmentation_val_loss + classification_val_loss

            # Update metrics
            val_loss += total_val_loss.item() * valid_inputs.size(0)
            segmentation_preds = (segmentation_output > 0.5).float()
            val_corrects += torch.sum(segmentation_preds == valid_masks.data)
            val_total_pixels += torch.numel(segmentation_preds)
            val_iou_sum += iou_score(segmentation_preds, valid_masks)

    # Calculate average losses and metrics
    val_loss /= len(val_loader.dataset)
    val_acc = float(val_corrects) / val_total_pixels if val_total_pixels > 0 else 0
    val_iou = val_iou_sum / len(val_loader)

    return val_loss, val_acc, val_iou

num_epochs = 100
patience = 10
optimizer = optim.Adam(model.parameters(), lr=1e-4)
#find optimal learning rate - reduce learning rate when metric stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience)

def train_model(model, train_loader, val_loader, segmentation_criterion, classification_criterion, optimizer, num_epochs, patience):
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience//2, factor=0.1, verbose=True)

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping initiated")
            break

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_pixels = 0
        train_iou_sum = 0.0

        for inputs, masks, class_labels in train_loader:
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            segmentation_output, classification_output = model(inputs)
            
            # Calculate losses
            segmentation_loss = segmentation_criterion(segmentation_output, masks)
            classification_loss = classification_criterion(classification_output, class_labels)
            total_loss = segmentation_loss + classification_loss
            
            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            # Update running loss and accuracy
            running_loss += total_loss.item() * inputs.size(0)
            segmentation_preds = torch.sigmoid(segmentation_output) > 0.5
            running_corrects += (segmentation_preds == masks).float().sum()
            total_pixels += torch.numel(segmentation_preds)
            train_iou_sum += iou_score(segmentation_preds, masks)

        # Calculate average loss and accuracy over the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / total_pixels
        epoch_iou = train_iou_sum / len(train_loader)

        # Validate after each epoch
        val_loss, val_acc, val_iou = validate_model(model, val_loader, segmentation_criterion, classification_criterion)

        # Learning rate scheduler step
        scheduler.step(val_loss)

        # Check if validation loss improved
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'unetmodel_best.pth')  # Save the best model
            
        else:
            epochs_no_improve += 1

        # Check for early stopping
        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after {epoch + 1} epochs!')
            early_stop = True
        
        # Log epoch metrics
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Train IoU: {epoch_iou:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val IoU: {val_iou:.4f}')

    return model

# Call the train_model function with all required arguments
trained_model = train_model(model, train_loader, val_loader, segmentation_criterion, classification_criterion, optimizer, num_epochs, patience)

#reverse mapping
id_to_disease = {v: k for k, v in disease_to_id.items()}

def visualize_predictions(model, DataLoader, num_images, disease_mapping=disease_mapping):
    model.eval()
    images_processed = 0

    with torch.no_grad():
        for images, masks, labels in DataLoader:
            if images_processed >= num_images:
                break

            segmentation_output, classification_output = model(images)
            _, predicted_labels = torch.max(classification_output, 1)

            # Apply sigmoid to segmentation output to get probabilities
            segmentation_predictions = torch.sigmoid(segmentation_output) > 0.5

            for i in range(images.shape[0]):
                if images_processed >= num_images:
                    break

                plt.figure(figsize=(12, 6))

                # Check if the predicted class is not 'healthy'
                if disease_mapping and disease_mapping.get(predicted_labels[i].item()) != "healthy":
                    
                    # Original Image
                    plt.subplot(1, 2, 1)
                    plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
                    plt.title('Original Image')
                    plt.axis('off')

                    # Overlayed Image
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
                    plt.imshow(segmentation_predictions[i].cpu().numpy().squeeze(), cmap='Reds', alpha=0.5)

                    # Display Predicted Disease Type
                    disease_name = disease_mapping.get(predicted_labels[i].item(), "Unknown")
                    plt.title(f'Predicted Disease: {disease_name}')
                    plt.axis('off')
                    
                else:
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.transpose(images[i].cpu().numpy(), (1, 2, 0)))
                    plt.title('No Mask for Healthy Class. Leaf Inputted is Healthy.')
                    plt.axis('off')


                plt.show()
                images_processed += 1



# Visualize predictions
visualize_predictions(trained_model, val_loader, num_images=10, disease_mapping=disease_mapping)
