"""
Model utilities for X-ray classification.
Located in: src/model/model_utils.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchxrayvision as xrv
import mlflow


class XRayDataset(Dataset):
    """Custom Dataset for X-ray images"""
    
    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df: DataFrame with columns ['image_path', 'label1', 'label2', ...]
            img_dir: Root directory containing images
            transform: Optional transform to be applied on images
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.label_cols = [col for col in df.columns if col != 'image_path']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'image_path']
        img_path = f"{self.img_dir}/{img_name}"

        img = Image.open(img_path).convert("L")
        img = np.array(img)

        # center crop to square
        h, w = img.shape
        m = min(h, w)
        img = img[(h - m)//2:(h - m)//2 + m,
                  (w - m)//2:(w - m)//2 + m]

        # resize
        if self.transform:
            img = self.transform(img)

        # 🔑 THIS LINE FIXES EVERYTHING
        if img.ndim == 3:
            img = img.squeeze()

        # normalize
        img = xrv.datasets.normalize(img, 255)

        # (1, H, W)
        img = torch.from_numpy(img).float().unsqueeze(0)

        labels = self.df.loc[idx, self.label_cols].values.astype(np.float32)
        labels = torch.from_numpy(labels)

        return img, labels



def get_model(model_name, num_classes):
    model = xrv.models.DenseNet(weights=model_name)

    # Replace classifier
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # 🔑 Disable pretrained output normalization (CRITICAL)
    model.op_threshs = None

    return model



def create_data_loaders(train_df, test_df, img_dir, batch_size, img_size):
    # Only resizer here
    transform = xrv.datasets.XRayResizer(img_size)

    train_dataset = XRayDataset(train_df, img_dir, transform=transform)
    test_dataset = XRayDataset(test_df, img_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    return running_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate model on test set
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Average loss
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
    
    return running_loss / len(test_loader)


def train_loop(model, train_df, test_df, img_dir, batch_size, num_epochs, learning_rate, img_size):
    """
    Main training loop
    
    Args:
        model: PyTorch model
        train_df: Training dataframe
        test_df: Test dataframe
        img_dir: Image directory
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        img_size: Image size
    
    Returns:
        Trained model
    """
    # Set device
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_df, test_df, img_dir, batch_size, img_size
    )
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_loss = evaluate_model(model, test_loader, criterion, device)
        
        # Log metrics (MLflow autolog handles this, but we can add custom metrics)
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    
    print("\n✅ Training completed!")
    return model