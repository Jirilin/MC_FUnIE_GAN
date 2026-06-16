# PatchGAN Discriminator for MC-FUnIE-GAN

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    # Convolution block for discriminator
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, 
                 padding=1, use_bn=True, use_dropout=False):
        super(ConvBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if use_dropout:
            layers.append(nn.Dropout2d(0.2))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):

    # PatchGAN Discriminator
    def __init__(self, in_channels=3, n_filters=64):
        super(Discriminator, self).__init__()
        
        self.conv1 = ConvBlock(in_channels * 2, n_filters, kernel_size=4, stride=2, use_bn=False)
        self.conv2 = ConvBlock(n_filters, n_filters * 2, kernel_size=4, stride=2, use_bn=True)
        self.conv3 = ConvBlock(n_filters * 2, n_filters * 4, kernel_size=4, stride=2, use_bn=True)
        self.conv4 = ConvBlock(n_filters * 4, n_filters * 8, kernel_size=4, stride=1, use_bn=True, use_dropout=True)
        
        # Output layer
        self.output = nn.Conv2d(n_filters * 8, 1, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x, y):
        # Concatenate input and target
        x = torch.cat([x, y], dim=1)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output(x)
        
        return x