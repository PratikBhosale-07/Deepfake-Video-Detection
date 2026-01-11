"""
Dataset and Transforms for Face Frames Deepfake Detection
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

class FaceFramesDataset(Dataset):
    """
    Custom Dataset for loading face frame images with labels.
    
    Args:
        csv_file (str): Path to the CSV file containing image paths and labels
        transform (callable, optional): Optional transform to be applied on images
    """
    
    def __init__(self, csv_file, transform=None):
        """
        Initialize the dataset.
        
        The CSV file should have columns:
        - 'filename': path to the image file
        - 'label': 'real' or 'fake' (will be converted to 0 or 1)
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        """Return the total number of samples."""
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        img_path = self.data_frame.iloc[idx]['filename']
        label_str = self.data_frame.iloc[idx]['label']
        
        # Convert label string to integer (0 for real, 1 for fake)
        label = 0 if label_str == 'real' else 1
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Training transforms with data augmentation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Validation/Test transforms without augmentation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
