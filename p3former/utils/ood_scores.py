"""Post-hoc point-level OOD scores computed from semantic logits.

All scores follow the convention: higher score = more OOD (REL, ISPRS 2026,
Eq. 2). Hyperparameters follow docs/others/baselines/OOD_Baseline.pdf:
ODIN uses temperature 1000 with epsilon = 0 (no input perturbation), the
Energy score uses temperature 1. Entropy is the Shannon entropy of the
softmax distribution (natural log), as in trash/Done/eval_ood_from_logits.py.
"""
from typing import Dict

import numpy as np
import torch

# Canonical method order; evaluation tables and pred_pts_seg keys follow it.
OOD_SCORE_KEYS = ('msp', 'maxlogit', 'odin', 'energy', 'entropy')


def compute_ood_scores(logits: torch.Tensor,
                       odin_temperature: float = 1000.0,
                       energy_temperature: float = 1.0
                       ) -> Dict[str, torch.Tensor]:
    """Compute all OOD scores from classification logits.

    Args:
        logits: [V, C] float tensor of per-voxel (or per-point) class logits.
        odin_temperature: softmax temperature for ODIN (T=1000, eps=0).
        energy_temperature: temperature for the free-energy score (T=1).

    Returns:
        dict with keys ``OOD_SCORE_KEYS``, each a [V] tensor,
        higher = more OOD.
    """
    if logits.dim() != 2:
        raise ValueError(f'expected [V, C] logits, got shape '
                         f'{tuple(logits.shape)}')
    logits = logits.float()
    scores = dict()
    scores['msp'] = -torch.softmax(logits, dim=1).max(dim=1).values
    scores['maxlogit'] = -logits.max(dim=1).values
    scores['odin'] = -torch.softmax(logits / odin_temperature,
                                    dim=1).max(dim=1).values
    scores['energy'] = -energy_temperature * torch.logsumexp(
        logits / energy_temperature, dim=1)
    # Shannon entropy of the softmax; log_softmax keeps p*log(p) finite
    # (and -> 0) for vanishing probabilities without an explicit clip.
    log_probs = torch.log_softmax(logits, dim=1)
    scores['entropy'] = -(log_probs.exp() * log_probs).sum(dim=1)
    return scores


def point_ood_scores(voxel_logits: torch.Tensor,
                     point2voxel_map: torch.Tensor,
                     odin_temperature: float = 1000.0,
                     energy_temperature: float = 1.0
                     ) -> Dict[str, np.ndarray]:
    """Compute voxel OOD scores and project them to points.

    Args:
        voxel_logits: [V, C] per-voxel class logits.
        point2voxel_map: [N] index of each point's voxel (any int/float
            dtype; cast to long, same as the panoptic projection).

    Returns:
        dict with keys ``OOD_SCORE_KEYS``, each a float32 numpy array [N].
    """
    voxel_scores = compute_ood_scores(voxel_logits, odin_temperature,
                                      energy_temperature)
    p2v = point2voxel_map.long()
    return {
        key: value[p2v].detach().cpu().numpy().astype(np.float32)
        for key, value in voxel_scores.items()
    }
