"""
Main training script for MC-FUnIE-GAN
"""

import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time

from config import Config
from data_loader import get_data_loaders
from models import Generator, Discriminator
from metrics import evaluate_batch, compute_all_metrics
from optimizers.hglo import HGLOOptimizer
from utils import create_dirs, save_checkpoint, save_sample_images, plot_metrics

class MC_FUnIE_GAN_Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Create directories
        create_dirs()
        
        # Initialize models
        self.generator = Generator(in_channels=config.N_CHANNELS, 
                                   n_filters=config.N_FILTERS).to(self.device)
        self.discriminator = Discriminator(in_channels=config.N_CHANNELS,
                                          n_filters=config.N_FILTERS).to(self.device)
        
        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_mse = nn.MSELoss()
        
        # Data loaders
        self.train_loader, self.test_loader = get_data_loaders(
            config.DATASET_PATH,
            batch_size=config.BATCH_SIZE,
            img_size=config.IMG_SIZE,
            num_workers=config.NUM_WORKERS
        )
        
        # Optimizers
        self.optimizer_G = None
        self.optimizer_D = None
        
        # TensorBoard
        self.writer = SummaryWriter(config.LOG_DIR)
        
        # Metrics tracking
        self.train_metrics = {
            'psnr': [], 'ssim': [], 'uiqm': [],
            'uicm': [], 'uism': [], 'uiconm': []
        }
        self.test_metrics = {
            'psnr': [], 'ssim': [], 'uiqm': [],
            'uicm': [], 'uism': [], 'uiconm': []
        }
        
        # Initialize with default hyperparameters
        self.set_hyperparameters(
            lr_G=config.LEARNING_RATE_G,
            lr_D=config.LEARNING_RATE_D,
            lambda_perceptual=config.LAMBDA_PERCEPTUAL,
            lambda_tv=config.LAMBDA_TV
        )
    
    def set_hyperparameters(self, lr_G, lr_D, lambda_perceptual, lambda_tv):
        """Set hyperparameters for training"""
        self.lr_G = lr_G
        self.lr_D = lr_D
        self.lambda_perceptual = lambda_perceptual
        self.lambda_tv = lambda_tv
        
        # Recreate optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=lr_G, betas=(0.5, 0.999)
        )
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999)
        )
    
    def run_hglo_optimization(self):
        """Run HGLO to find optimal hyperparameters"""
        print("=== Starting HGLO Optimization ===")
        hglo_optimizer = HGLOOptimizer(self.generator, self.discriminator, self.config)
        best_params = hglo_optimizer.optimize(verbose=True)
        
        print(f"Optimized Hyperparameters:")
        print(f"  lr_G: {best_params['lr_G']:.6f}")
        print(f"  lr_D: {best_params['lr_D']:.6f}")
        print(f"  lambda_perceptual: {best_params['lambda_perceptual']:.4f}")
        print(f"  lambda_tv: {best_params['lambda_tv']:.4f}")
        
        self.set_hyperparameters(
            lr_G=best_params['lr_G'],
            lr_D=best_params['lr_D'],
            lambda_perceptual=best_params['lambda_perceptual'],
            lambda_tv=best_params['lambda_tv']
        )
        
        return best_params
    
    def compute_loss(self, real, fake):
        """Compute all loss components"""
        # Real and fake labels
        real_label = torch.ones(real.size(0), 1, 1, 1).to(self.device)
        fake_label = torch.zeros(real.size(0), 1, 1, 1).to(self.device)
        
        # Discriminator predictions
        d_real = self.discriminator(real, real)
        d_fake = self.discriminator(real, fake.detach())
        
        # Generator loss
        g_gan_loss = self.criterion_gan(d_fake, real_label)
        g_l1_loss = self.criterion_l1(fake, real) * 100
        g_perceptual_loss = self.criterion_mse(fake, real) * self.lambda_perceptual
        g_tv_loss = self.compute_tv_loss(fake) * self.lambda_tv
        
        g_total_loss = g_gan_loss + g_l1_loss + g_perceptual_loss + g_tv_loss
        
        # Discriminator loss
        d_real_loss = self.criterion_gan(d_real, real_label)
        d_fake_loss = self.criterion_gan(d_fake, fake_label)
        d_total_loss = (d_real_loss + d_fake_loss) / 2
        
        losses = {
            'g_total': g_total_loss,
            'g_gan': g_gan_loss,
            'g_l1': g_l1_loss,
            'g_perceptual': g_perceptual_loss,
            'g_tv': g_tv_loss,
            'd_total': d_total_loss,
            'd_real': d_real_loss,
            'd_fake': d_fake_loss
        }
        
        return losses
    
    def compute_tv_loss(self, x):
        """Total Variation regularization"""
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]
        tv_loss = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_w))
        return tv_loss
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {
            'g_total': 0, 'g_gan': 0, 'g_l1': 0,
            'g_perceptual': 0, 'g_tv': 0,
            'd_total': 0, 'd_real': 0, 'd_fake': 0
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (real, ref) in enumerate(progress_bar):
            real = real.to(self.device)
            ref = ref.to(self.device)
            
            # Generate fake images
            fake = self.generator(real)
            
            # Update Discriminator
            self.optimizer_D.zero_grad()
            losses = self.compute_loss(real, fake)
            losses['d_total'].backward(retain_graph=True)
            self.optimizer_D.step()
            
            # Update Generator
            self.optimizer_G.zero_grad()
            losses = self.compute_loss(real, fake)
            losses['g_total'].backward()
            self.optimizer_G.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'G': f"{losses['g_total'].item():.3f}",
                'D': f"{losses['d_total'].item():.3f}"
            })
        
        # Average losses
        avg_losses = {k: v / len(self.train_loader) for k, v in epoch_losses.items()}
        return avg_losses
    
    def evaluate(self):
        """Evaluate model on test set"""
        print("\n=== Evaluating on Test Set ===")
        metrics = evaluate_batch(self.generator, self.test_loader, self.device)
        
        print(f"PSNR: {metrics['psnr']:.4f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"UIQM: {metrics['uiqm']:.4f}")
        print(f"UICM: {metrics['uicm']:.4f}")
        print(f"UISM: {metrics['uism']:.4f}")
        print(f"UIConM: {metrics['uiconm']:.4f}")
        
        return metrics
    
    def train(self, epochs=None):
        """Main training loop"""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        print(f"=== Starting MC-FUnIE-GAN Training ===")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print("=" * 50)
        
        # Optional: Run HGLO before training
        # self.run_hglo_optimization()
        
        best_psnr = 0
        best_epoch = 0
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # Train
            losses = self.train_epoch(epoch)
            
            # Evaluate every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                metrics = self.evaluate()
                
                # Store metrics
                for key in self.test_metrics:
                    if key in metrics:
                        self.test_metrics[key].append(metrics[key])
                
                # Save sample images
                save_sample_images(self.generator, self.test_loader, epoch, self.device)
                
                # Save checkpoint if best
                if metrics['psnr'] > best_psnr:
                    best_psnr = metrics['psnr']
                    best_epoch = epoch
                    save_checkpoint({
                        'epoch': epoch,
                        'model_state_dict': self.generator.state_dict(),
                        'optimizer_state_dict': self.optimizer_G.state_dict(),
                        'metrics': metrics
                    }, 'checkpoints/best_model.pth')
                    print(f"*** New best PSNR: {best_psnr:.4f} dB ***")
            
            # Log losses
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch}/{epochs} completed in {epoch_time:.2f}s")
            print(f"  G_loss: {losses['g_total']:.4f}, D_loss: {losses['d_total']:.4f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Loss/G_total', losses['g_total'], epoch)
            self.writer.add_scalar('Loss/D_total', losses['d_total'], epoch)
            self.writer.add_scalar('Learning_Rate/G', self.lr_G, epoch)
            self.writer.add_scalar('Learning_Rate/D', self.lr_D, epoch)
        
        print(f"\n=== Training Complete ===")
        print(f"Best PSNR: {best_psnr:.4f} dB at epoch {best_epoch}")
        print(f"Model saved to: checkpoints/best_model.pth")
        
        # Plot metrics
        plot_metrics(self.test_metrics, 'logs/test_metrics.png')
        
        return self.generator

def main():
    """Main entry point"""
    config = Config()
    trainer = MC_FUnIE_GAN_Trainer(config)
    
    # Train
    generator = trainer.train(epochs=100)  # Use 100 for testing, 200 for final
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    metrics = trainer.evaluate()
    
    # Save final model
    save_checkpoint({
        'epoch': trainer.config.EPOCHS,
        'model_state_dict': generator.state_dict(),
        'metrics': metrics
    }, 'checkpoints/final_model.pth')
    
    print("\n=== Results Summary ===")
    print(f"PSNR: {metrics['psnr']:.4f} dB")
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"UIQM: {metrics['uiqm']:.4f}")
    print(f"UICM: {metrics['uicm']:.4f}")
    print(f"UISM: {metrics['uism']:.4f}")
    print(f"UIConM: {metrics['uiconm']:.4f}")

if __name__ == "__main__":
    main()