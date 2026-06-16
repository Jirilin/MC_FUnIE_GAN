"""
Image Quality Metrics: PSNR, SSIM, UIQM, UICM, UISM, UIConM
"""

import torch
import numpy as np
import cv2
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_psnr(img1, img2):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio)
    """
    img1 = img1.cpu().detach().numpy() if torch.is_tensor(img1) else img1
    img2 = img2.cpu().detach().numpy() if torch.is_tensor(img2) else img2
    
    # Convert to 0-255 range if normalized
    if img1.max() <= 1.0:
        img1 = img1 * 255
    if img2.max() <= 1.0:
        img2 = img2 * 255
    
    # Transpose from (C, H, W) to (H, W, C)
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def compute_ssim(img1, img2):
    """
    Compute SSIM (Structural Similarity Index)
    """
    img1 = img1.cpu().detach().numpy() if torch.is_tensor(img1) else img1
    img2 = img2.cpu().detach().numpy() if torch.is_tensor(img2) else img2
    
    # Convert to 0-255 range if normalized
    if img1.max() <= 1.0:
        img1 = img1 * 255
    if img2.max() <= 1.0:
        img2 = img2 * 255
    
    # Transpose from (C, H, W) to (H, W, C)
    if img1.shape[0] == 3:
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] == 3:
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Clamp to valid range
    img1 = np.clip(img1, 0, 255).astype(np.uint8)
    img2 = np.clip(img2, 0, 255).astype(np.uint8)
    
    return structural_similarity(img1, img2, multichannel=True, 
                                 data_range=255, channel_axis=2)

def compute_uicm(img):
    """
    Compute UICM (Underwater Image Colorfulness Measure)
    """
    img = img.cpu().detach().numpy() if torch.is_tensor(img) else img
    
    # Convert to 0-255 range if normalized
    if img.max() <= 1.0:
        img = img * 255
    
    # Transpose from (C, H, W) to (H, W, C)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    L, a, b = cv2.split(lab)
    
    # Compute mean and std of a and b channels
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    std_a = np.std(a)
    std_b = np.std(b)
    
    # UICM formula
    uicm = np.sqrt(std_a**2 + std_b**2) + 0.3 * np.sqrt(mean_a**2 + mean_b**2)
    
    return uicm

def compute_uism(img):
    """
    Compute UISM (Underwater Image Sharpness Measure)
    """
    img = img.cpu().detach().numpy() if torch.is_tensor(img) else img
    
    # Convert to 0-255 range if normalized
    if img.max() <= 1.0:
        img = img * 255
    
    # Transpose from (C, H, W) to (H, W, C)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Compute Laplacian variance (sharpness measure)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = np.var(laplacian)
    
    # Normalize to [0, 1]
    uism = np.clip(variance / 1000, 0, 1)
    
    return uism

def compute_uiconm(img):
    """
    Compute UIConM (Underwater Image Contrast Measure)
    """
    img = img.cpu().detach().numpy() if torch.is_tensor(img) else img
    
    # Convert to 0-255 range if normalized
    if img.max() <= 1.0:
        img = img * 255
    
    # Transpose from (C, H, W) to (H, W, C)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Compute contrast using RMS contrast
    mean = np.mean(gray)
    std = np.std(gray)
    
    # Normalize contrast measure
    uiconm = np.clip(std / 128, 0, 1)
    
    return uiconm

def compute_uiqm(img):
    """
    Compute UIQM (Underwater Image Quality Measure)
    Combined measure: UICM + UISM + UIConM
    """
    uicm = compute_uicm(img)
    uism = compute_uism(img)
    uiconm = compute_uiconm(img)
    
    # Weighted combination (from original UIQM paper)
    uiqm = 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm
    
    return uiqm

def compute_all_metrics(enhanced, reference):
    """
    Compute all metrics: PSNR, SSIM, UIQM, UICM, UISM, UIConM
    """
    metrics = {
        'psnr': compute_psnr(enhanced, reference),
        'ssim': compute_ssim(enhanced, reference),
        'uicm': compute_uicm(enhanced),
        'uism': compute_uism(enhanced),
        'uiconm': compute_uiconm(enhanced),
        'uiqm': compute_uiqm(enhanced)
    }
    return metrics

def evaluate_batch(generator, dataloader, device):
    """
    Evaluate generator on a batch of images
    """
    generator.eval()
    metrics_list = []
    
    with torch.no_grad():
        for real, ref in dataloader:
            real = real.to(device)
            ref = ref.to(device)
            
            enhanced = generator(real)
            
            for i in range(real.size(0)):
                metrics = compute_all_metrics(enhanced[i], ref[i])
                metrics_list.append(metrics)
    
    generator.train()
    
    # Compute average metrics
    avg_metrics = {}
    for key in metrics_list[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in metrics_list])
    
    return avg_metrics