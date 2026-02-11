"""
Lab Value MLP Encoder

Encodes the 3 lab values (CRP, WBC, Hb) into a feature vector.
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LAB_INPUT_DIM, LAB_HIDDEN_DIMS, LAB_FEATURE_DIM


class LabEncoder(nn.Module):
    """
    MLP encoder for lab values (CRP, WBC, Hb).
    
    Takes normalized lab values and produces a feature vector
    that can be fused with image features.
    """
    
    def __init__(self, input_dim=LAB_INPUT_DIM, hidden_dims=LAB_HIDDEN_DIMS,
                 output_dim=LAB_FEATURE_DIM, dropout=0.2):
        """
        Args:
            input_dim: Number of lab values (3: CRP, WBC, Hb)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output feature dimension
            dropout: Dropout rate
        """
        super(LabEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final projection
        layers.extend([
            nn.Linear(prev_dim, output_dim),
            nn.ReLU()
        ])
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Encode lab values.
        
        Args:
            x: Input tensor of shape (batch, 3) containing [CRP, WBC, Hb]
            
        Returns:
            Feature tensor of shape (batch, output_dim)
        """
        return self.encoder(x)


class LabEncoderWithAttention(nn.Module):
    """
    Lab encoder with self-attention mechanism.
    
    Learns which lab values are most important for the current sample.
    """
    
    def __init__(self, input_dim=LAB_INPUT_DIM, hidden_dim=32, 
                 output_dim=LAB_FEATURE_DIM, n_heads=1):
        """
        Args:
            input_dim: Number of lab values
            hidden_dim: Hidden dimension
            output_dim: Output feature dimension
            n_heads: Number of attention heads
        """
        super(LabEncoderWithAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Embed each lab value
        self.lab_embedding = nn.Linear(1, hidden_dim)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, x):
        """
        Encode lab values with attention.
        
        Args:
            x: Input tensor of shape (batch, 3)
            
        Returns:
            Feature tensor of shape (batch, output_dim)
        """
        batch_size = x.size(0)
        
        # Reshape to (batch, 3, 1) and embed each value
        x = x.unsqueeze(-1)  # (batch, 3, 1)
        x = self.lab_embedding(x)  # (batch, 3, hidden_dim)
        
        # Self-attention
        x, attention_weights = self.attention(x, x, x)
        
        # Flatten and project
        x = x.reshape(batch_size, -1)  # (batch, 3 * hidden_dim)
        x = self.output_proj(x)
        
        return x
    
    def get_attention_weights(self, x):
        """Return attention weights for interpretability."""
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x = self.lab_embedding(x)
        _, attention_weights = self.attention(x, x, x)
        return attention_weights


if __name__ == "__main__":
    # Test Lab Encoders
    print("Testing Lab Encoders...")
    
    # Test input: batch of normalized lab values
    batch_size = 4
    x = torch.randn(batch_size, 3)  # [CRP, WBC, Hb] normalized
    
    # Test basic MLP encoder
    print("\n1. Basic MLP Encoder:")
    encoder = LabEncoder()
    out = encoder(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    
    # Test attention encoder
    print("\n2. Attention Encoder:")
    attn_encoder = LabEncoderWithAttention()
    out = attn_encoder(x)
    attn_weights = attn_encoder.get_attention_weights(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Attention weights shape: {attn_weights.shape}")
    
    # Count parameters
    def count_params(model):
        return sum(p.numel() for p in model.parameters())
    
    print(f"\n   MLP params: {count_params(encoder):,}")
    print(f"   Attention params: {count_params(attn_encoder):,}")
