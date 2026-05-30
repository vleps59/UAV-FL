# pytorchexample/task.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import cv2
import numpy as np
from pathlib import Path
import json
import random

# Load class mapping
def load_class_mapping():
    mapping_path = Path("data/DescribedFlyingObjects/class_mapping.json")
    if mapping_path.exists():
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        class_to_id = mapping["class_to_id"]
        id_to_class = {int(k): v for k, v in mapping["id_to_class"].items()}
        num_classes = mapping["num_classes"]
        print(f"Loaded {num_classes} classes: {list(class_to_id.keys())}")
    else:
        class_to_id = {'drone': 0, 'bird': 1, 'airplane': 2, 'helicopter': 3, 'balloon': 4}
        id_to_class = {v: k for k, v in class_to_id.items()}
        num_classes = 5
        print(f"Using fallback classes: {list(class_to_id.keys())}")
    
    return class_to_id, id_to_class, num_classes

CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES = load_class_mapping()
CLASS_NAMES = list(CLASS_TO_ID.keys())


class FlyingObjectsDataset(Dataset):
    """Dataset loader for Described Flying Objects in YOLO format."""
    
    def __init__(self, img_dir, label_dir, img_size=224, augment=False):  # Changed to 224
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.augment = augment
        
        self.images = [f for f in self.img_dir.glob("*.jpg") 
                      if (self.label_dir / f"{f.stem}.txt").exists()]
        print(f"Loaded {len(self.images)} images from {img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label_path = self.label_dir / f"{img_path.stem}.txt"
        
        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1) / 255.0
        
        boxes = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * self.img_size
                        y_center = float(parts[2]) * self.img_size
                        width = float(parts[3]) * self.img_size
                        height = float(parts[4]) * self.img_size
                        x = x_center - width / 2
                        y = y_center - height / 2
                        boxes.append([class_id, x, y, width, height])
        
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
        return torch.tensor(img, dtype=torch.float32), boxes


def custom_collate_fn(batch):
    """
    Custom collate function that handles variable-sized bounding box tensors.
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = torch.stack(images, 0)
    return images, targets


def load_data(partition_id: int, num_partitions: int, batch_size: int = 8):
    """Load data for a specific client (UAV in the swarm)."""
    data_root = Path("data/DescribedFlyingObjects/train")
    
    img_dir = data_root / "images"
    label_dir = data_root / "labels"
    
    full_dataset = FlyingObjectsDataset(img_dir, label_dir, img_size=224)  # Changed to 224
    indices = list(range(len(full_dataset)))
    random.seed(partition_id)
    random.shuffle(indices)
    
    partition_size = len(indices) // num_partitions
    start_idx = partition_id * partition_size
    end_idx = start_idx + partition_size if partition_id < num_partitions - 1 else len(indices)
    
    subset = Subset(full_dataset, indices[start_idx:end_idx])
    
    dataloader = DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn
    )
    
    return dataloader


class FlyingObjectDetector(nn.Module):
    """Simplified CNN-based detector for flying objects"""
    
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes
        
        # Feature extractor (backbone)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pool to 1x1
        )
        
        # Detection head (output: class scores + bounding box coordinates)
        # 5 values: class (num_classes dims) + 4 box coordinates + 1 confidence
        self.classifier = nn.Sequential(
            nn.Flatten(),  # 512 -> 512
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes + 5)  # class logits + 4 box coords + 1 confidence
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_weights(self):
        """Extract model weights for FL transmission"""
        return {k: v.cpu() for k, v in self.state_dict().items()}
    
    def set_weights(self, weights):
        """Set model weights from FL aggregation"""
        self.load_state_dict(weights, strict=False)


def get_model(num_classes=NUM_CLASSES):
    """Factory function to create the model"""
    return FlyingObjectDetector(num_classes)


def compute_loss(predictions, targets_list, device, num_classes=NUM_CLASSES):
    """
    Compute loss with targets as a list of tensors.
    
    Args:
        predictions: Model outputs [batch, num_classes + 5]
        targets_list: List of [num_boxes, 5] tensors (class, x, y, w, h)
        device: Device to compute on
        num_classes: Number of object classes
    
    Returns:
        Scalar loss value
    """
    batch_size = predictions.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    num_valid = 0
    
    for batch_idx in range(batch_size):
        pred = predictions[batch_idx]  # [num_classes + 5]
        targets = targets_list[batch_idx]  # [num_boxes, 5]
        
        # Skip if no targets in this image
        if targets.shape[0] == 0:
            continue
        
        num_valid += 1
        
        # Take first target for simplicity
        target_class = targets[0, 0].long()
        target_box = targets[0, 1:5]
        
        # Extract predictions
        pred_class_logits = pred[:num_classes]
        pred_box = pred[num_classes:num_classes+4]
        pred_conf = pred[num_classes+4]
        
        # Classification loss
        class_loss = nn.functional.cross_entropy(
            pred_class_logits.unsqueeze(0), 
            target_class.unsqueeze(0)
        )
        
        # Box regression loss (MSE)
        box_loss = nn.functional.mse_loss(pred_box, target_box)
        
        # Confidence loss (encourage confidence for boxes that exist)
        conf_loss = -torch.log(torch.sigmoid(pred_conf) + 1e-6)
        
        total_loss += class_loss + box_loss + 0.1 * conf_loss
    
    if num_valid == 0:
        return torch.tensor(0.0, device=device)
    
    return total_loss / num_valid


def train(model, trainloader, optimizer, epochs, device):
    """Train the model for one client"""
    model.train()
    model.to(device)
    
    for epoch in range(epochs):
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(trainloader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(images)
            
            # Compute loss
            loss = compute_loss(predictions, targets, device, model.num_classes)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = running_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    
    return model.state_dict()


def test(model, testloader, device):
    """Evaluate the model"""
    model.eval()
    model.to(device)
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, targets in testloader:
            images = images.to(device)
            predictions = model(images)
            loss = compute_loss(predictions, targets, device, model.num_classes)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / max(num_batches, 1)