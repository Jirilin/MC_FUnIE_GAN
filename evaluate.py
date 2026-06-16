# Evaluation for trained MC-FUnIE-GAN model

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from data_loader import get_data_loaders
from models import Generator
from metrics import compute_all_metrics, evaluate_batch
from utils import create_dirs

def load_trained_model(checkpoint_path, config):
    # Load trained generator from checkpoint
    generator = Generator(in_channels=config.N_CHANNELS, 
                         n_filters=config.N_FILTERS).to(config.DEVICE)
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.eval()
    
    return generator

def visualize_results(generator, test_loader, device, num_samples=8, save_dir="results/test"):
    """Visualize and save enhanced images"""
    create_dirs()
    os.makedirs(save_dir, exist_ok=True)
    
    generator.eval()
    
    with torch.no_grad():
        for idx, (real, ref) in enumerate(tqdm(test_loader, desc="Generating")):
            if idx >= num_samples:
                break
            
            real = real.to(device)
            ref = ref.to(device)
            fake = generator(real)
            
            # Save images
            for i in range(min(real.size(0), 4)):
                base_name = f"sample_{idx*4+i}"
                save_image(real[i], f"{save_dir}/{base_name}_raw.png", normalize=True)
                save_image(fake[i], f"{save_dir}/{base_name}_enhanced.png", normalize=True)
                save_image(ref[i], f"{save_dir}/{base_name}_reference.png", normalize=True)

def compute_and_report_metrics(generator, test_loader, device):
    """Compute all metrics and report"""
    metrics = evaluate_batch(generator, test_loader, device)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"PSNR:   {metrics['psnr']:.4f} dB")
    print(f"SSIM:   {metrics['ssim']:.4f}")
    print(f"UIQM:   {metrics['uiqm']:.4f}")
    print(f"UICM:   {metrics['uicm']:.4f}")
    print(f"UISM:   {metrics['uism']:.4f}")
    print(f"UIConM: {metrics['uiconm']:.4f}")
    print("="*50)
    
    return metrics

def compare_with_baseline(metrics_ours, metrics_baseline):
    """Compare with baseline methods"""
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE METHODS")
    print("="*60)
    
    methods = ['FUnIE-GAN', 'MCRNet', 'DIRBW-Net', 'DA-GAN', 'U-TWGAN', 'E-PUGAN', 'Ours']
    metrics_names = ['PSNR', 'SSIM', 'UIQM']
    
    # Example baseline values (replace with actual values from paper)
    baseline_data = {
        'FUnIE-GAN': {'PSNR': 22.51, 'SSIM': 0.853, 'UIQM': 3.12},
        'MCRNet': {'PSNR': 24.16, 'SSIM': 0.883, 'UIQM': 3.19},
        'DIRBW-Net': {'PSNR': 24.53, 'SSIM': 0.892, 'UIQM': 3.31},
        'DA-GAN': {'PSNR': 26.38, 'SSIM': 0.921, 'UIQM': 3.52},
        'U-TWGAN': {'PSNR': 27.05, 'SSIM': 0.937, 'UIQM': 3.58},
        'E-PUGAN': {'PSNR': 28.02, 'SSIM': 0.906, 'UIQM': 3.49},
    }
    ours = {'PSNR': metrics_ours['psnr'], 'SSIM': metrics_ours['ssim'], 
            'UIQM': metrics_ours['uiqm']}
    
    print(f"{'Method':<12} {'PSNR':>10} {'SSIM':>8} {'UIQM':>8}")
    print("-"*40)
    
    for method in methods:
        if method == 'Ours':
            data = ours
            marker = " *"
        else:
            data = baseline_data.get(method, {'PSNR': 0, 'SSIM': 0, 'UIQM': 0})
            marker = ""
        
        print(f"{method:<12} {data['PSNR']:>10.2f} {data['SSIM']:>8.3f} {data['UIQM']:>8.2f}{marker}")

def main():
    """Main evaluation function"""
    config = Config()
    
    # Load data
    _, test_loader = get_data_loaders(
        config.DATASET_PATH,
        batch_size=config.BATCH_SIZE,
        img_size=config.IMG_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    # Load model
    checkpoint_path = "checkpoints/best_model.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using train.py")
        return
    
    generator = load_trained_model(checkpoint_path, config)
    
    # Evaluate
    metrics = compute_and_report_metrics(generator, test_loader, config.DEVICE)
    
    # Visualize
    visualize_results(generator, test_loader, config.DEVICE, num_samples=8)
    
    # Compare with baseline
    compare_with_baseline(metrics, None)
    
    print("\nResults saved to: results/test/")

if __name__ == "__main__":
    main()