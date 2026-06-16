"""
Configuration file for MC-FUnIE-GAN
"""

import torch

class Config:
    # Dataset
    DATASET_PATH = "./data/UIEB"  # Path to UIEB dataset
    UTIEB_PATH = "./data/UTIEB"   # Path to UTIEB dataset
    IMG_SIZE = 256
    BATCH_SIZE = 16
    NUM_WORKERS = 4
    
    # Training
    EPOCHS = 200
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 4e-4
    BETA1 = 0.5
    BETA2 = 0.999
    LAMBDA_PERCEPTUAL = 0.1
    LAMBDA_TV = 0.01
    
    # Model
    N_CHANNELS = 3
    N_FILTERS = 64
    
    # HGLO Optimization
    HGLO_POPULATION = 30
    HGLO_MAX_ITER = 150
    HGLO_ALPHA = 0.8
    HGLO_BETA = 0.6
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    CHECKPOINT_DIR = "./checkpoints"
    RESULTS_DIR = "./results"
    LOG_DIR = "./logs"