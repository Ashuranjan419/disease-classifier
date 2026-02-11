"""
Synthetic Lab Value Generator

Generates realistic lab values (CRP, WBC, Hb) based on disease class
using published medical ranges.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LAB_RANGES, CLASS_NAMES, SEED


class LabValueGenerator:
    """
    Generates synthetic lab values for each disease class.
    
    Lab Values:
    - CRP (C-Reactive Protein): mg/L - Inflammation marker
    - WBC (White Blood Cell Count): ×10^9/L - Infection marker  
    - Hb (Hemoglobin): g/dL - Anemia marker
    """
    
    def __init__(self, seed=SEED):
        self.rng = np.random.RandomState(seed)
        self.lab_ranges = LAB_RANGES
        
    def generate(self, class_name, n_samples=1):
        """
        Generate lab values for a given disease class.
        
        Args:
            class_name: One of ["Normal", "Tumor", "Infection", "Inflammatory"]
            n_samples: Number of samples to generate
            
        Returns:
            numpy array of shape (n_samples, 3) with [CRP, WBC, Hb]
        """
        if class_name not in self.lab_ranges:
            raise ValueError(f"Unknown class: {class_name}. Expected one of {CLASS_NAMES}")
        
        ranges = self.lab_ranges[class_name]
        
        # Generate values with Gaussian distribution
        crp = self.rng.normal(ranges["CRP"][0], ranges["CRP"][1], n_samples)
        wbc = self.rng.normal(ranges["WBC"][0], ranges["WBC"][1], n_samples)
        hb = self.rng.normal(ranges["Hb"][0], ranges["Hb"][1], n_samples)
        
        # Clip to physiologically realistic ranges
        crp = np.clip(crp, 0.1, 200.0)   # CRP can't be negative, max ~200
        wbc = np.clip(wbc, 1.0, 30.0)    # WBC realistic range
        hb = np.clip(hb, 5.0, 20.0)      # Hb realistic range
        
        return np.column_stack([crp, wbc, hb])
    
    def generate_for_class_id(self, class_id, n_samples=1):
        """Generate lab values using class ID instead of name."""
        return self.generate(CLASS_NAMES[class_id], n_samples)
    
    def get_normalization_stats(self, n_samples_per_class=10000):
        """
        Compute global mean and std for lab normalization.
        
        Returns:
            dict with 'mean' and 'std' arrays of shape (3,)
        """
        all_labs = []
        for class_name in CLASS_NAMES:
            labs = self.generate(class_name, n_samples_per_class)
            all_labs.append(labs)
        
        all_labs = np.vstack(all_labs)
        
        return {
            'mean': all_labs.mean(axis=0),
            'std': all_labs.std(axis=0)
        }


class LabNormalizer:
    """Normalizes lab values using z-score normalization."""
    
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        
    def fit(self, labs):
        """Compute mean and std from training data."""
        self.mean = labs.mean(axis=0)
        self.std = labs.std(axis=0)
        return self
    
    def transform(self, labs):
        """Apply z-score normalization."""
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        return (labs - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, labs):
        """Fit and transform in one step."""
        self.fit(labs)
        return self.transform(labs)
    
    def inverse_transform(self, labs_normalized):
        """Convert back to original scale."""
        return labs_normalized * self.std + self.mean


if __name__ == "__main__":
    # Demo: Generate and visualize lab values
    generator = LabValueGenerator()
    
    print("=" * 60)
    print("Synthetic Lab Value Generation Demo")
    print("=" * 60)
    
    for class_name in CLASS_NAMES:
        labs = generator.generate(class_name, n_samples=5)
        print(f"\n{class_name}:")
        print(f"  CRP (mg/L) | WBC (×10^9/L) | Hb (g/dL)")
        print("-" * 45)
        for i in range(labs.shape[0]):
            print(f"  {labs[i, 0]:8.2f}  |  {labs[i, 1]:8.2f}    | {labs[i, 2]:6.2f}")
    
    # Show normalization stats
    print("\n" + "=" * 60)
    print("Global Normalization Statistics:")
    stats = generator.get_normalization_stats()
    print(f"Mean: CRP={stats['mean'][0]:.2f}, WBC={stats['mean'][1]:.2f}, Hb={stats['mean'][2]:.2f}")
    print(f"Std:  CRP={stats['std'][0]:.2f}, WBC={stats['std'][1]:.2f}, Hb={stats['std'][2]:.2f}")
