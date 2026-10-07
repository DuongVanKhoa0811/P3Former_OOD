"""OCCUQ uncertainty head (Heidrich, Beemelmanns et al., ICRA 2025).

Mirrors ``MLPHeadv5`` of the official code (``occ_head.py:199-233`` at
commit 2aa5429):
- a bias-free linear map without spectral normalisation (SurroundOcc's
  widened 1x1 conv);
- four residual blocks ``x + ReLU(SN-Linear(x))``;
- a spectrally normalised linear classifier.

Spectral normalisation is ``torch.nn.utils.spectral_norm`` with its
defaults (sigma = 1, one power iteration). It stays active at inference,
using the stored u and v. The Gaussians of the density score are fitted on
the output of the last block (spec
docs/superpowers/specs/2026-10-07-occuq-density-design.md).
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class _OCCUQHead(nn.Module):
    """Per-voxel OCCUQ head on ``pe_features``.

    Args:
        in_channels: feature width (P3Former's ``embed_dims``, 256).
        num_classes: classifier outputs (25 for DSO, ignore channel
            included).
        num_blocks: residual spectrally normalised blocks (4, as OCCUQ).
    """

    def __init__(self, in_channels: int = 256, num_classes: int = 25,
                 num_blocks: int = 4) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_channels, in_channels, bias=False)
        self.blocks = nn.ModuleList(
            spectral_norm(nn.Linear(in_channels, in_channels))
            for _ in range(num_blocks))
        self.classifier = spectral_norm(nn.Linear(in_channels, num_classes))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[V, C] features -> (logits [V, num_classes], feature [V, C])."""
        x = self.input_proj(x)
        for block in self.blocks:
            x = x + F.relu(block(x))
        return self.classifier(x), x
