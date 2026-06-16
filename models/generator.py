# Generator with Multi-Scale Dilated Convolution (MSDC) module

import torch
import torch.nn as nn

class MSDCModule(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(MSDCModule, self).__init__()
        
        # Branch 1: Dilation rate 1 (standard convolution)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: Dilation rate 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: Dilation rate 4
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 convolution for cross-channel fusion
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Concatenate along channel dimension
        fused = torch.cat([b1, b2, b3], dim=1)
        return self.fusion(fused)

class ConvBlock(nn.Module):
    # Convolution block with optional batch norm and dropout
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, 
                 padding=1, use_bn=True, use_dropout=False):
        super(ConvBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class Generator(nn.Module):
    # MC-FUnIE-GAN Generator with MSDC module
    def __init__(self, in_channels=3, n_filters=64):
        super(Generator, self).__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, n_filters, kernel_size=3, stride=1, use_bn=False)
        self.enc2 = ConvBlock(n_filters, n_filters * 2, kernel_size=3, stride=2, use_bn=True)
        self.enc3 = ConvBlock(n_filters * 2, n_filters * 4, kernel_size=3, stride=2, use_bn=True)
        
        # MSDC Module (applied after encoder)
        self.msdc = MSDCModule(n_filters * 4, n_filters * 4)
        
        # Decoder with skip connections
        self.dec3 = ConvBlock(n_filters * 8, n_filters * 2, kernel_size=3, stride=1, use_bn=True)
        self.dec2 = ConvBlock(n_filters * 4, n_filters, kernel_size=3, stride=1, use_bn=True)
        self.dec1 = ConvBlock(n_filters * 2, n_filters, kernel_size=3, stride=1, use_bn=False)
        
        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(n_filters, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Upsampling
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # [B, 64, H, W]
        e2 = self.enc2(e1)          # [B, 128, H/2, W/2]
        e3 = self.enc3(e2)          # [B, 256, H/4, W/4]
        
        # MSDC module
        e3_msdc = self.msdc(e3)     # [B, 256, H/4, W/4]
        
        # Decoder with skip connections
        d3 = self.up3(e3_msdc)      # [B, 256, H/2, W/2]
        d3 = torch.cat([d3, e2], dim=1)  # Skip connection
        d3 = self.dec3(d3)          # [B, 128, H/2, W/2]
        
        d2 = self.up2(d3)           # [B, 128, H, W]
        d2 = torch.cat([d2, e1], dim=1)  # Skip connection
        d2 = self.dec2(d2)          # [B, 64, H, W]
        
        d1 = self.dec1(d2)          # [B, 64, H, W]
        out = self.output(d1)       # [B, 3, H, W]
        
        return out