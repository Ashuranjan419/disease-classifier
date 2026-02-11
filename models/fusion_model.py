"""
Multimodal Fusion Model

Combines CT image features with lab value features for disease classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CNN_FEATURE_DIM, LAB_FEATURE_DIM, FUSION_METHOD, FUSION_DIM,
    CLASSIFIER_HIDDEN_DIM, DROPOUT_RATE, NUM_CLASSES
)
from models.cnn_encoder import CNNEncoder, SimpleCNNEncoder
from models.lab_encoder import LabEncoder, LabEncoderWithAttention


class ConcatFusion(nn.Module):
    """Simple concatenation fusion."""
    
    def __init__(self, img_dim, lab_dim):
        super(ConcatFusion, self).__init__()
        self.output_dim = img_dim + lab_dim
    
    def forward(self, img_feat, lab_feat):
        return torch.cat([img_feat, lab_feat], dim=1)


class GatedFusion(nn.Module):
    """Gated fusion mechanism that learns to weight modalities."""
    
    def __init__(self, img_dim, lab_dim):
        super(GatedFusion, self).__init__()
        
        self.output_dim = img_dim
        
        # Gate networks
        self.img_gate = nn.Sequential(
            nn.Linear(img_dim + lab_dim, img_dim),
            nn.Sigmoid()
        )
        self.lab_gate = nn.Sequential(
            nn.Linear(img_dim + lab_dim, img_dim),
            nn.Sigmoid()
        )
        
        # Project lab to same dim as img
        self.lab_proj = nn.Linear(lab_dim, img_dim)
    
    def forward(self, img_feat, lab_feat):
        # Project lab features
        lab_proj = self.lab_proj(lab_feat)
        
        # Concatenate for gate computation
        combined = torch.cat([img_feat, lab_feat], dim=1)
        
        # Compute gates
        img_gate = self.img_gate(combined)
        lab_gate = self.lab_gate(combined)
        
        # Gated fusion
        fused = img_gate * img_feat + lab_gate * lab_proj
        
        return fused


class AttentionFusion(nn.Module):
    """Cross-attention fusion between image and lab features."""
    
    def __init__(self, img_dim, lab_dim, hidden_dim=128):
        super(AttentionFusion, self).__init__()
        
        self.output_dim = hidden_dim * 2
        
        # Project both to same dimension
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.lab_proj = nn.Linear(lab_dim, hidden_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, img_feat, lab_feat):
        # Project features
        img_proj = self.img_proj(img_feat).unsqueeze(1)  # (batch, 1, hidden)
        lab_proj = self.lab_proj(lab_feat).unsqueeze(1)  # (batch, 1, hidden)
        
        # Cross-attention: image attends to lab
        attended, _ = self.cross_attention(img_proj, lab_proj, lab_proj)
        attended = attended.squeeze(1)
        
        # Concatenate original projected features with attended
        fused = torch.cat([img_proj.squeeze(1), attended], dim=1)
        
        return fused


class MultimodalFusionModel(nn.Module):
    """
    Complete multimodal model for disease classification.
    
    Architecture:
    CT Image -> CNN Encoder -> Image Features
                                    |
                                    v
    Lab Values -> MLP Encoder -> Lab Features -> Fusion Layer -> Classifier -> Disease Class
    
    Supports multiple fusion strategies:
    - concat: Simple concatenation
    - gated: Learned gating mechanism
    - attention: Cross-attention fusion
    """
    
    def __init__(self, 
                 cnn_backbone='resnet18',
                 cnn_pretrained=True,
                 img_feature_dim=CNN_FEATURE_DIM,
                 lab_feature_dim=LAB_FEATURE_DIM,
                 fusion_method=FUSION_METHOD,
                 num_classes=NUM_CLASSES,
                 dropout=DROPOUT_RATE,
                 use_simple_cnn=False,
                 use_lab_attention=False):
        """
        Args:
            cnn_backbone: CNN backbone name
            cnn_pretrained: Use pretrained weights
            img_feature_dim: Image feature dimension
            lab_feature_dim: Lab feature dimension
            fusion_method: Fusion strategy ('concat', 'gated', 'attention')
            num_classes: Number of disease classes
            dropout: Dropout rate
            use_simple_cnn: Use simple CNN instead of pretrained
            use_lab_attention: Use attention-based lab encoder
        """
        super(MultimodalFusionModel, self).__init__()
        
        self.fusion_method = fusion_method
        self.num_classes = num_classes
        
        # Image encoder
        if use_simple_cnn:
            self.img_encoder = SimpleCNNEncoder(feature_dim=img_feature_dim)
        else:
            self.img_encoder = CNNEncoder(
                backbone=cnn_backbone,
                pretrained=cnn_pretrained,
                feature_dim=img_feature_dim
            )
        
        # Lab encoder
        if use_lab_attention:
            self.lab_encoder = LabEncoderWithAttention(output_dim=lab_feature_dim)
        else:
            self.lab_encoder = LabEncoder(output_dim=lab_feature_dim)
        
        # Fusion layer
        if fusion_method == 'concat':
            self.fusion = ConcatFusion(img_feature_dim, lab_feature_dim)
        elif fusion_method == 'gated':
            self.fusion = GatedFusion(img_feature_dim, lab_feature_dim)
        elif fusion_method == 'attention':
            self.fusion = AttentionFusion(img_feature_dim, lab_feature_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(CLASSIFIER_HIDDEN_DIM, num_classes)
        )
    
    def forward(self, image, lab):
        """
        Forward pass.
        
        Args:
            image: CT image tensor (batch, 1, H, W)
            lab: Lab values tensor (batch, 3)
            
        Returns:
            Logits tensor (batch, num_classes)
        """
        # Encode modalities
        img_feat = self.img_encoder(image)
        lab_feat = self.lab_encoder(lab)
        
        # Fuse features
        fused = self.fusion(img_feat, lab_feat)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def get_features(self, image, lab):
        """Get intermediate features for visualization/analysis."""
        img_feat = self.img_encoder(image)
        lab_feat = self.lab_encoder(lab)
        fused = self.fusion(img_feat, lab_feat)
        return {
            'image_features': img_feat,
            'lab_features': lab_feat,
            'fused_features': fused
        }


class ImageOnlyModel(nn.Module):
    """Baseline model using only CT images (no lab values)."""
    
    def __init__(self, cnn_backbone='resnet18', cnn_pretrained=True,
                 feature_dim=CNN_FEATURE_DIM, num_classes=NUM_CLASSES,
                 dropout=DROPOUT_RATE, use_simple_cnn=False):
        super(ImageOnlyModel, self).__init__()
        
        if use_simple_cnn:
            self.encoder = SimpleCNNEncoder(feature_dim=feature_dim)
        else:
            self.encoder = CNNEncoder(
                backbone=cnn_backbone,
                pretrained=cnn_pretrained,
                feature_dim=feature_dim
            )
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, CLASSIFIER_HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(CLASSIFIER_HIDDEN_DIM, num_classes)
        )
    
    def forward(self, image, lab=None):
        """Lab argument ignored but kept for API consistency."""
        features = self.encoder(image)
        return self.classifier(features)


class LabOnlyModel(nn.Module):
    """Baseline model using only lab values (no CT images)."""
    
    def __init__(self, feature_dim=LAB_FEATURE_DIM, num_classes=NUM_CLASSES,
                 dropout=DROPOUT_RATE, use_attention=False):
        super(LabOnlyModel, self).__init__()
        
        if use_attention:
            self.encoder = LabEncoderWithAttention(output_dim=feature_dim)
        else:
            self.encoder = LabEncoder(output_dim=feature_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, CLASSIFIER_HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(CLASSIFIER_HIDDEN_DIM // 2, num_classes)
        )
    
    def forward(self, image=None, lab=None):
        """Image argument ignored but kept for API consistency."""
        features = self.encoder(lab)
        return self.classifier(features)


def create_model(model_type='fusion', **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: 'fusion', 'image_only', or 'lab_only'
        **kwargs: Arguments passed to model constructor
        
    Returns:
        Model instance
    """
    if model_type == 'fusion':
        # Filter kwargs for fusion model
        valid_keys = ['cnn_backbone', 'cnn_pretrained', 'img_feature_dim', 
                      'lab_feature_dim', 'fusion_method', 'num_classes', 
                      'dropout', 'use_simple_cnn', 'use_lab_attention']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return MultimodalFusionModel(**filtered_kwargs)
    elif model_type == 'image_only':
        # Filter kwargs for image-only model
        valid_keys = ['cnn_backbone', 'cnn_pretrained', 'feature_dim', 
                      'num_classes', 'dropout', 'use_simple_cnn']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return ImageOnlyModel(**filtered_kwargs)
    elif model_type == 'lab_only':
        # Filter kwargs for lab-only model
        valid_keys = ['feature_dim', 'num_classes', 'dropout', 'use_attention']
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        return LabOnlyModel(**filtered_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing Multimodal Models...")
    
    batch_size = 4
    image = torch.randn(batch_size, 1, 224, 224)
    lab = torch.randn(batch_size, 3)
    
    # Test Fusion Model
    print("\n1. Multimodal Fusion Model (concat):")
    model = MultimodalFusionModel(fusion_method='concat', use_simple_cnn=True)
    out = model(image, lab)
    print(f"   Output shape: {out.shape}")
    
    # Test Gated Fusion
    print("\n2. Multimodal Fusion Model (gated):")
    model = MultimodalFusionModel(fusion_method='gated', use_simple_cnn=True)
    out = model(image, lab)
    print(f"   Output shape: {out.shape}")
    
    # Test Image Only
    print("\n3. Image Only Model:")
    model = ImageOnlyModel(use_simple_cnn=True)
    out = model(image)
    print(f"   Output shape: {out.shape}")
    
    # Test Lab Only
    print("\n4. Lab Only Model:")
    model = LabOnlyModel()
    out = model(lab=lab)
    print(f"   Output shape: {out.shape}")
    
    # Parameter counts
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print("\nParameter Counts:")
    print(f"   Fusion (simple CNN): {count_params(MultimodalFusionModel(use_simple_cnn=True)):,}")
    print(f"   Image Only: {count_params(ImageOnlyModel(use_simple_cnn=True)):,}")
    print(f"   Lab Only: {count_params(LabOnlyModel()):,}")
