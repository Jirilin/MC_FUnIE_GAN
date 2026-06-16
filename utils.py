"""
Utility functions for MC-FUnIE-GAN
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm

def create_dirs():
    """Create necessary directories"""
    dirs = [
        "checkpoints",
        "results/train",
        "results/test",
        "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_checkpoint(state, filename):
    """Save checkpoint"""
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(filename, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)

def denormalize(tensor):
    """Denormalize image tensor (0-1 range)"""
    return torch.clamp(tensor, 0, 1)

def save_sample_images(generator, dataloader, epoch, device, save_path="results/train"):
    """Save sample images during training"""
    generator.eval()
    with torch.no_grad():
        for i, (real, _) in enumerate(dataloader):
            if i >= 1:
                break
            real = real.to(device)
            fake = generator(real)
            
            # Save images
            save_image(real, f"{save_path}/epoch_{epoch}_real.png", normalize=True)
            save_image(fake, f"{save_path}/epoch_{epoch}_fake.png", normalize=True)
    generator.train()

def plot_metrics(metrics, save_path="logs/metrics.png"):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    metrics_list = ['psnr', 'ssim', 'uiqm', 'uicm', 'uism', 'uiconm']
    titles = ['PSNR', 'SSIM', 'UIQM', 'UICM', 'UISM', 'UIConM']
    
    for i, (metric, title) in enumerate(zip(metrics_list, titles)):
        row, col = i // 3, i % 3
        if metric in metrics:
            values = metrics[metric]
            axes[row, col].plot(values)
            axes[row, col].set_title(title)
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()