import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from transformers import ViTModel, ViTConfig
import os
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import matplotlib.pyplot as plt

def resize_image(image, size=(224, 224)):
    return image.resize(size, Image.LANCZOS)

def perform_ela(image, resaved_quality=95):
    resaved_image = image.copy()
    resaved_image.save("temp.jpg", "JPEG", quality=resaved_quality)
    resaved_image = Image.open("temp.jpg")
    
    ela_image = ImageChops.difference(image, resaved_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = resize_image(image)
    
    ela_image = perform_ela(image)
    ela_image = ela_image.resize((224, 224)).convert('L')
    image_np = np.array(image)
    ela_np = np.array(ela_image)
    ela_np = np.expand_dims(ela_np, axis=2)
    concatenated_image = np.concatenate((image_np, ela_np), axis=2)
    concatenated_image = concatenated_image.astype(np.float32) / 255.0
    
    return concatenated_image

def preprocess_forged_region_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = resize_image(image)
    return image

class CustomViTModel(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomViTModel, self).__init__()
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k')
        config.patch_size = 16
        config.num_channels = 4  # Update to 4 for the 4-channel input
        
        self.vit = ViTModel(config)
        
        self.classifier = nn.Linear(config.hidden_size, num_classes)
        
        self.localization_head = nn.Sequential(
            nn.Conv2d(config.hidden_size, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        outputs = self.vit(pixel_values=x)
        cls_logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        
        # Localization head
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]
        batch_size, num_patches, hidden_size = patch_embeddings.shape
        h = w = int(num_patches ** 0.5)
        patch_embeddings = patch_embeddings.permute(0, 2, 1).reshape(batch_size, hidden_size, h, w)
        forge_map = self.localization_head(patch_embeddings)
        
        return cls_logits, forge_map

# Initialize the custom ViT model
model = CustomViTModel()
print(model)

authentic_dir = "CASIA2/Au"
tempered_dir = "CASIA2/Tp"
forged_region_dir = "CASIA2/CASIA 2 Groundtruth"

class ForgeryDataset(Dataset):
    def __init__(self, authentic_dir, tempered_dir, forged_region_dir, transform=None):
        self.authentic_dir = authentic_dir
        self.tempered_dir = tempered_dir
        self.forged_region_dir = forged_region_dir
        self.transform = transform
        
        self.authentic_images = os.listdir(authentic_dir)
        self.tempered_images = os.listdir(tempered_dir)
        
        self.data = [(os.path.join(authentic_dir, img), 0) for img in self.authentic_images] + \
                    [(os.path.join(tempered_dir, img), 1) for img in self.tempered_images]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = preprocess_image(img_path)  # Use the provided preprocessing functions
        image_tensor = torch.tensor(image).permute(2, 0, 1)
        
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        if label == 0:
            # Placeholder for localization label for authentic images
            localization_label = torch.zeros((1, 14, 14))
        else:
            # Load forged region image, resize, and convert to binary mask
            tempered_base_filename = os.path.basename(img_path).split('.')[0]
            forged_region_path = os.path.join(self.forged_region_dir, tempered_base_filename + '_gt.png')
            if os.path.exists(forged_region_path):
                forged_region_image = Image.open(forged_region_path).convert('L')  # Convert to grayscale
                forged_region_image_resized = forged_region_image.resize((14, 14), Image.BILINEAR)
                localization_label = transforms.ToTensor()(forged_region_image_resized)
            else:
                localization_label = torch.zeros((1, 14, 14))
                # print(localization_label.shape)
            # print(localization_label.shape)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, label_tensor, localization_label

dataset = ForgeryDataset(authentic_dir, tempered_dir, forged_region_dir)
print(f"Total images: {len(dataset)}")

# visulaizing dataset 
def visualize_dataset_item(dataset, idx):
    img_path, image_tensor, label_tensor, localization_label = dataset[idx]
    
    # Print the label
    print(f"Label: {label_tensor.item()}")
    print(img_path)
    
    # Convert tensors back to image format for visualization
    image_np = image_tensor.permute(1, 2, 0).numpy()
    localization_np = localization_label.permute(1, 2, 0).numpy()

    # Display the original image and labels
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image_np[..., :3])  # Display only the RGB channels
    
    plt.subplot(1, 3, 2)
    plt.title('ELA Image')
    plt.imshow(image_np[..., 3], cmap='gray')  # Display only the ELA channel
    
    plt.subplot(1, 3, 3)
    plt.title('Localization Label')
    plt.imshow(localization_np, cmap='jet', alpha=0.5)  # Display the localization label
    
    plt.show()

visualize_dataset_item(dataset, idx=11250)

# Splitting dataset into Training and Validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Training set size: {len(train_dataset)}")
print(f"Validation set size: {len(val_dataset)}")

# Create DataLoaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training and Validation
model = CustomViTModel()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

loss_fn_classification = nn.BCEWithLogitsLoss()
loss_fn_localization = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
print_every = 500

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_train_steps = len(train_loader)
    
    for i, (images_batch, class_labels_batch, local_labels_batch) in enumerate(train_loader):
        
        images_batch = images_batch.to(device)
        class_labels_batch = class_labels_batch.to(device)
        local_labels_batch = local_labels_batch.to(device)
        
        optimizer.zero_grad()
        cls_logits, forge_map = model(images_batch)
        
        cls_logits = cls_logits.view(-1, 1)
        class_labels_batch = class_labels_batch.view(-1, 1)

        loss_classification = loss_fn_classification(cls_logits, class_labels_batch)
        loss_localization = loss_fn_localization(forge_map, local_labels_batch)
        loss = loss_classification + loss_localization
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i + 1) % print_every == 0 or (i + 1) == total_train_steps:
            print(f'Epoch {epoch + 1}/{num_epochs}, Step {i + 1}/{total_train_steps}, Loss: {running_loss / (i + 1):.4f}')
        
    # Validation
    model.eval()
    val_loss = 0.0
    total_val_steps = len(val_loader)
    
    with torch.no_grad():
        for i, (images_batch, class_labels_batch, local_labels_batch) in enumerate(val_loader):
            images_batch = images_batch.to(device)
            class_labels_batch = class_labels_batch.to(device)
            local_labels_batch = local_labels_batch.to(device)

            cls_logits, forge_map = model(images_batch)

            loss_classification = loss_fn_classification(cls_logits, class_labels_batch)
            loss_localization = loss_fn_localization(forge_map, local_labels_batch)
            loss = loss_classification + loss_localization

            val_loss += loss.item()
        
    print(f'Validation Loss: {val_loss / total_val_steps:.4f}')

# Save model weights
torch.save(model.state_dict(), 'forgery_detection_model.pth')

# Visualization function
def preprocess_inference_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = resize_image(image)
    
    ela_image = perform_ela(image)
    ela_image = ela_image.resize((224, 224)).convert('L')
    image_np = np.array(image)
    ela_np = np.array(ela_image)
    ela_np = np.expand_dims(ela_np, axis=2)
    concatenated_image = np.concatenate((image_np, ela_np), axis=2)
    concatenated_image = concatenated_image.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.tensor(concatenated_image).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    return image_tensor

# Define visualization function
def visualize_forgery(image_tensor, forged_map):
    plt.figure(figsize=(10, 10))
    
    forged_map = forged_map.squeeze().cpu().detach().numpy()
    forged_map = np.resize(forged_map, (224, 224))
    
    plt.imshow(image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy())
    plt.imshow(forged_map, cmap='jet', alpha=0.5)
    plt.show()

# Load the pre-trained model
model = CustomViTModel()
model.load_state_dict(torch.load('forgery_detection_model.pth'))
model.eval()

# Preprocess the image
image_path = 'CASIA2/Tp/Tp_S_NRN_S_N_ani10167_ani10167_12446.jpg'
image_tensor = preprocess_inference_image(image_path)

# Move the tensor to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_tensor = image_tensor.to(device)
model = model.to(device)

# Perform inference
with torch.no_grad():
    cls_logits, forge_map = model(image_tensor)
    probability = torch.sigmoid(cls_logits).item()
    print(f"Probability of being forged: {probability}")

    # Visualize forged areas
    visualize_forgery(image_tensor, forge_map)