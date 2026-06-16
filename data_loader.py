"""
Data loader for UIEB and UTIEB datasets
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class UnderwaterDataset(Dataset):
    """
    Underwater Image Enhancement Dataset
    Expects folder structure:
        dataset/
            raw/
                image1.jpg
                image2.jpg
                ...
            ref/
                image1.jpg
                image2.jpg
                ...
    """
    def __init__(self, root_dir, transform=None, img_size=256):
        self.root_dir = root_dir
        self.raw_dir = os.path.join(root_dir, 'raw')
        self.ref_dir = os.path.join(root_dir, 'ref')
        
        self.img_size = img_size
        
        # Get list of images
        self.images = sorted([f for f in os.listdir(self.raw_dir) 
                              if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                   std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        # Load raw image
        raw_path = os.path.join(self.raw_dir, img_name)
        raw_img = Image.open(raw_path).convert('RGB')
        
        # Load reference image
        ref_path = os.path.join(self.ref_dir, img_name)
        ref_img = Image.open(ref_path).convert('RGB')
        
        # Apply CLAHE and Gamma Correction (preprocessing)
        raw_img = self._preprocess(raw_img)
        
        # Apply transforms
        raw_img = self.transform(raw_img)
        ref_img = self.transform(ref_img)
        
        return raw_img, ref_img
    
    def _preprocess(self, img):
        """Apply CLAHE and Gamma Correction"""
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Convert to LAB color space for CLAHE
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge back
        lab_clahe = cv2.merge([l_clahe, a, b])
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        
        # Apply Gamma Correction
        gamma = 1.2
        img_gamma = np.power(img_clahe / 255.0, gamma) * 255.0
        img_gamma = img_gamma.astype(np.uint8)
        
        return Image.fromarray(img_gamma)

def get_data_loaders(dataset_path, batch_size=16, img_size=256, 
                     num_workers=4, split_ratio=0.8):
    """
    Create train and test DataLoaders
    """
    full_dataset = UnderwaterDataset(dataset_path, img_size=img_size)
    
    # Split dataset
    train_size = int(split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader