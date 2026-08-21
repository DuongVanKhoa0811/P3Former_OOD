"""Post-hoc point-level OOD scores computed from semantic logits.

All scores follow the convention: higher score = more OOD (REL, ISPRS 2026,
Eq. 2). Hyperparameters follow docs/others/baselines/OOD_Baseline.pdf:
ODIN uses temperature 1000 with epsilon = 0 (no input perturbation), the
Energy score uses temperature 1. Entropy is the Shannon entropy of the
softmax.

Hierarchy-aware variants (Group aggregation and Group Normalization) follow
"Understanding Confidence Fragmentation in OOD Detection for 3D LiDAR
Semantic Segmentation" (papers/RelatedPapers/GroupPaper.pdf, Eq. 2-8) as
implemented in trash/Done/eval_ood_from_logits.py, adapted to this module's
sign convention (``-max`` instead of ``1 - max`` for the MSP/ODIN family —
a constant shift, every ranking is unchanged). Given a partition of the
classes into groups: probability mass is summed within each group (P_g),
logits keep their operator within the group (max for MaxLogit, logsumexp
for Energy), and the group score takes the max over groups. Group
Normalization chance-corrects the probabilities, Q_g = [P_g - K_g / C]_+
(K_g = group size, C = number of classes), and subtracts log K_g from the
logit aggregates.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

# Flat (class-level) scores; evaluation tables and pred_pts_seg keys follow
# this order.
OOD_SCORE_KEYS = ('msp', 'maxlogit', 'odin', 'energy', 'entropy')
# Hierarchy-aware scores, produced only when ``class_groups`` is given.
GROUP_SCORE_KEYS = tuple(f'group_{key}' for key in OOD_SCORE_KEYS)
GN_SCORE_KEYS = tuple(f'gn_{key}' for key in OOD_SCORE_KEYS)
ALL_SCORE_KEYS = OOD_SCORE_KEYS + GROUP_SCORE_KEYS + GN_SCORE_KEYS

_EPS = 1e-12


def validate_class_groups(class_groups: Sequence[Sequence[int]],
                          num_classes: int) -> List[List[int]]:
    """Check that ``class_groups`` are non-empty, disjoint and in range.

    Groups need not cover every class: classes outside all groups simply do
    not contribute to the hierarchy-aware scores.
    """
    groups = [[int(c) for c in group] for group in class_groups]
    seen = set()
    for group in groups:
        if len(group) == 0:
            raise ValueError('class_groups must not contain empty groups')
        for c in group:
            if not 0 <= c < num_classes:
                raise ValueError(f'class index {c} out of range for '
                                 f'{num_classes} classes')
            if c in seen:
                raise ValueError(f'class {c} appears in more than one group')
            seen.add(c)
    return groups


def compute_ood_scores(logits: torch.Tensor,
                       odin_temperature: float = 1000.0,
                       energy_temperature: float = 1.0,
                       class_groups: Optional[Sequence[Sequence[int]]] = None
                       ) -> Dict[str, torch.Tensor]:
    """Compute all OOD scores from classification logits.

    Args:
        logits: [V, C] float tensor of per-voxel (or per-point) class logits.
        odin_temperature: softmax temperature for ODIN (T=1000, eps=0).
        energy_temperature: temperature for the free-energy score (T=1).
        class_groups: optional partition of the class indices into semantic
            groups (list of index lists). When given, the ``group_*`` and
            ``gn_*`` scores are added.

    Returns:
        dict with keys ``OOD_SCORE_KEYS`` (plus ``GROUP_SCORE_KEYS`` and
        ``GN_SCORE_KEYS`` when ``class_groups`` is given), each a [V] tensor,
        higher = more OOD.
    """
    if logits.dim() != 2:
        raise ValueError(f'expected [V, C] logits, got shape '
                         f'{tuple(logits.shape)}')
    logits = logits.float()
    probs = torch.softmax(logits, dim=1)
    probs_T = torch.softmax(logits / odin_temperature, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    scores = dict()
    scores['msp'] = -probs.max(dim=1).values
    scores['maxlogit'] = -logits.max(dim=1).values
    scores['odin'] = -probs_T.max(dim=1).values
    scores['energy'] = -energy_temperature * torch.logsumexp(
        logits / energy_temperature, dim=1)
    scores['entropy'] = -(log_probs.exp() * log_probs).sum(dim=1)
    if class_groups is not None:
        scores.update(
            _hierarchy_scores(logits, probs, probs_T, class_groups,
                              energy_temperature))
    return scores


def _hierarchy_scores(logits: torch.Tensor, probs: torch.Tensor,
                      probs_T: torch.Tensor,
                      class_groups: Sequence[Sequence[int]],
                      energy_temperature: float) -> Dict[str, torch.Tensor]:
    """Group aggregation and Group Normalization scores (GroupPaper Eq. 2-8)."""
    num_classes = logits.shape[1]
    groups = validate_class_groups(class_groups, num_classes)
    # Per-group aggregates, each [V, G].
    group_prob = torch.stack([probs[:, g].sum(dim=1) for g in groups], dim=1)
    group_prob_T = torch.stack([probs_T[:, g].sum(dim=1) for g in groups],
                               dim=1)
    group_maxlogit = torch.stack(
        [logits[:, g].max(dim=1).values for g in groups], dim=1)
    group_lse = torch.stack([
        energy_temperature * torch.logsumexp(
            logits[:, g] / energy_temperature, dim=1) for g in groups
    ], dim=1)
    sizes = logits.new_tensor([float(len(g)) for g in groups])  # K_g
    log_sizes = torch.log(sizes)
    prior = sizes / num_classes  # K_g / C, expected mass under uniform p
    q = (group_prob - prior).clamp_min(0.0)  # Q_g = [P_g - K_g / C]_+
    q_T = (group_prob_T - prior).clamp_min(0.0)
    group_prob_c = group_prob.clamp_min(_EPS)

    out = dict()
    out['group_msp'] = -group_prob.max(dim=1).values
    # Identical to the flat MaxLogit when the groups form a partition.
    out['group_maxlogit'] = -group_maxlogit.max(dim=1).values
    out['group_odin'] = -group_prob_T.max(dim=1).values
    out['group_energy'] = -group_lse.max(dim=1).values
    out['group_entropy'] = -(group_prob_c * torch.log(group_prob_c)).sum(dim=1)
    out['gn_msp'] = -q.max(dim=1).values
    out['gn_maxlogit'] = -(group_maxlogit - log_sizes).max(dim=1).values
    out['gn_odin'] = -q_T.max(dim=1).values
    # The log K_g correction is scaled by the energy temperature so that a
    # group of K_g equal logits z scores exactly z; identical at T = 1.
    out['gn_energy'] = -(group_lse -
                         energy_temperature * log_sizes).max(dim=1).values
    # Entropy-style dispersion of the (un-normalised) corrected responses,
    # with the reference script's epsilon placement.
    out['gn_entropy'] = -((q + _EPS) * torch.log(q + _EPS)).sum(dim=1)
    return out


def point_ood_scores(voxel_logits: torch.Tensor,
                     point2voxel_map: torch.Tensor,
                     odin_temperature: float = 1000.0,
                     energy_temperature: float = 1.0,
                     class_groups: Optional[Sequence[Sequence[int]]] = None
                     ) -> Dict[str, np.ndarray]:
    """Compute voxel OOD scores and project them to points.

    Args:
        voxel_logits: [V, C] per-voxel class logits.
        point2voxel_map: [N] index of each point's voxel (any int/float
            dtype; cast to long, same as the panoptic projection).
        odin_temperature, energy_temperature, class_groups: see
            :func:`compute_ood_scores`.

    Returns:
        dict with the same keys as :func:`compute_ood_scores`, each a float32
        numpy array [N].
    """
    voxel_scores = compute_ood_scores(voxel_logits, odin_temperature,
                                      energy_temperature, class_groups)
    p2v = point2voxel_map.long()
    return {
        key: value[p2v].detach().cpu().numpy().astype(np.float32)
        for key, value in voxel_scores.items()
    }
