"""
Fusion module for combining radar, pseudo-LiDAR, and image BEV features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from ..config.base import BaseConfig


class ConcatFusion(nn.Module):
    """Simple concatenation fusion."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of BEV features [radar, pseudo, image]
        Returns:
            fused_features: (B, C, H, W)
        """
        return torch.cat(features, dim=1)


class WeightedSumFusion(nn.Module):
    """Weighted sum fusion with learnable weights."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Learnable weights for each modality
        self.weights = nn.Parameter(torch.ones(3))  # radar, pseudo, image

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of BEV features [radar, pseudo, image]
        Returns:
            fused_features: (B, C, H, W)
        """
        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * feat for w, feat in zip(weights, features))
        return fused


class AttentionFusion(nn.Module):
    """Attention-based fusion."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Attention mechanism
        self.query = nn.Linear(128, 128)  # Query from radar features
        self.key = nn.Linear(128, 128)   # Key from aligned image features
        self.value = nn.Linear(128, 128) # Value from aligned image features

        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of BEV features [radar, pseudo, image]
        Returns:
            fused_features: (B, C, H, W)
        """
        radar_feat, pseudo_feat, image_feat = features

        # Flatten spatial dimensions
        B, C_radar, H, W = radar_feat.shape
        radar_flat = radar_feat.view(B, C_radar, -1).permute(0, 2, 1)  # (B, H*W, C_radar)

        B, C_image, H, W = image_feat.shape
        image_flat = image_feat.view(B, C_image, -1).permute(0, 2, 1)    # (B, H*W, C_image)

        # Compute attention
        query = self.query(radar_flat)  # (B, H*W, 128)
        key = self.key(image_flat)      # (B, H*W, 128)
        value = self.value(image_flat)  # (B, H*W, 128)

        attended, _ = self.attention(query, key, value)

        # Reshape back
        fused = attended.permute(0, 2, 1).view(B, 128, H, W)

        return fused


class FusionModule(nn.Module):
    """Main fusion module that selects fusion method."""

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

        # Select fusion method
        if config.fusion_method == 'concat':
            self.fusion = ConcatFusion(config)
        elif config.fusion_method == 'weighted_sum':
            self.fusion = WeightedSumFusion(config)
        elif config.fusion_method == 'attention':
            self.fusion = AttentionFusion(config)
        else:
            raise ValueError(f"Unknown fusion method: {config.fusion_method}")

        # Feature alignment if needed
        self.radar_adjust = nn.Conv2d(128, 128, 1) if config.fusion_method != 'concat' else nn.Identity()
        self.pseudo_adjust = nn.Conv2d(128, 128, 1) if config.fusion_method != 'concat' else nn.Identity()
        self.image_adjust = nn.Conv2d(64, 128, 1) if config.fusion_method != 'concat' else nn.Identity()

    def forward(self, radar_bev: torch.Tensor, pseudo_bev: torch.Tensor, image_bev: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radar_bev: (B, 128, H, W) radar BEV features
            pseudo_bev: (B, 128, H, W) pseudo-LiDAR BEV features
            image_bev: (B, 64, H, W) image BEV features
        Returns:
            fused_features: (B, C, H, W)
        """
        # Adjust feature dimensions if needed
        radar_aligned = self.radar_adjust(radar_bev)
        pseudo_aligned = self.pseudo_adjust(pseudo_bev)
        image_aligned = self.image_adjust(image_bev)

        # Apply fusion
        fused = self.fusion([radar_aligned, pseudo_aligned, image_aligned])

        return fused