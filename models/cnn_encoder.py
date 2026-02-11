"""
CNN Encoder for CT Image Feature Extraction

Uses pre-trained backbone (ResNet/EfficientNet) modified for grayscale input.
"""

import torch
import torch.nn as nn
from torchvision import models
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CNN_BACKBONE, CNN_PRETRAINED, CNN_FEATURE_DIM, IMAGE_CHANNELS


class CNNEncoder(nn.Module):
    """
    CNN encoder for extracting features from CT images.
    
    Supports multiple backbones:
    - ResNet18, ResNet34, ResNet50
    - EfficientNet-B0
    
    Modified first conv layer to accept grayscale (1-channel) input.
    """
    
    def __init__(self, backbone=CNN_BACKBONE, pretrained=CNN_PRETRAINED,
                 feature_dim=CNN_FEATURE_DIM, in_channels=IMAGE_CHANNELS):
        """
        Args:
            backbone: Name of backbone architecture
            pretrained: Whether to use ImageNet pretrained weights
            feature_dim: Output feature dimension
            in_channels: Number of input channels (1 for grayscale CT)
        """
        super(CNNEncoder, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_out_dim = 512
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            backbone_out_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_out_dim = 2048
        elif backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            backbone_out_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer for grayscale input
        self._modify_first_conv(in_channels)
        
        # Remove classification head
        self._remove_classifier()
        
        # Add projection layer to desired feature dimension
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def _modify_first_conv(self, in_channels):
        """Modify first conv layer to accept grayscale input."""
        if self.backbone_name.startswith('resnet'):
            # Get original conv1
            orig_conv1 = self.backbone.conv1
            
            # Create new conv1 with desired input channels
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 
                orig_conv1.out_channels,
                kernel_size=orig_conv1.kernel_size,
                stride=orig_conv1.stride,
                padding=orig_conv1.padding,
                bias=orig_conv1.bias is not None
            )
            
            # Initialize with mean of original weights (for pretrained)
            if in_channels == 1:
                with torch.no_grad():
                    self.backbone.conv1.weight = nn.Parameter(
                        orig_conv1.weight.mean(dim=1, keepdim=True)
                    )
                    
        elif self.backbone_name == 'efficientnet':
            # EfficientNet first conv
            orig_conv = self.backbone.features[0][0]
            
            self.backbone.features[0][0] = nn.Conv2d(
                in_channels,
                orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None
            )
            
            if in_channels == 1:
                with torch.no_grad():
                    self.backbone.features[0][0].weight = nn.Parameter(
                        orig_conv.weight.mean(dim=1, keepdim=True)
                    )
    
    def _remove_classifier(self):
        """Remove the classification head from backbone."""
        if self.backbone_name.startswith('resnet'):
            self.backbone.fc = nn.Identity()
        elif self.backbone_name == 'efficientnet':
            self.backbone.classifier = nn.Identity()
    
    def forward(self, x):
        """
        Extract features from CT images.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch, feature_dim)
        """
        # Backbone feature extraction
        features = self.backbone(x)
        
        # Project to desired dimension
        features = self.projection(features)
        
        return features
    
    def freeze_backbone(self):
        """Freeze backbone weights for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"Froze {self.backbone_name} backbone parameters")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone weights for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print(f"Unfroze {self.backbone_name} backbone parameters")


class SimpleCNNEncoder(nn.Module):
    """
    Simple CNN encoder without pretrained weights.
    Useful for testing or when pretrained models aren't needed.
    """
    
    def __init__(self, in_channels=IMAGE_CHANNELS, feature_dim=CNN_FEATURE_DIM):
        super(SimpleCNNEncoder, self).__init__()
        
        self.feature_dim = feature_dim
        
        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.projection(x)
        return x


if __name__ == "__main__":
    # Test CNN encoders
    print("Testing CNN Encoders...")
    
    # Test input
    batch_size = 4
    x = torch.randn(batch_size, 1, 224, 224)  # Grayscale CT images
    
    # Test ResNet encoder
    print("\n1. ResNet18 Encoder:")
    encoder = CNNEncoder(backbone='resnet18', pretrained=False)
    out = encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    # Test Simple CNN encoder
    print("\n2. Simple CNN Encoder:")
    simple_encoder = SimpleCNNEncoder()
    out = simple_encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"\n   ResNet18 params: {count_params(encoder):,}")
    print(f"   SimpleCNN params: {count_params(simple_encoder):,}")
