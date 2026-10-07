"""Class-conditional Gaussians for the OCCUQ density score, from streamed
voxel features.

OCCUQ (official code ``tools/gmm_utils.py:163-220`` at commit 2aa5429)
stores up to 200M float32 feature vectors per class and calls
``torch.cov``. Here each class keeps running float64 sums instead: the
count, sum z and sum z z^T. So every training voxel is used, with 12.6 MB
of state for 24 classes x 256 dimensions.

``finalize`` turns the sums into, per class:
- the mean, and the N-1 covariance (``torch.cov``'s convention);
- the precision Cholesky factor P = L^-T, found with OCCUQ's jitter search
  (per class here);
- its log-determinant and the log class prior.

Pure functions; ``tools/fit_occuq_gmm.py`` runs the model and feeds them.
"""
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

# OCCUQ's jitter list (tools/gmm_utils.py:13-14), tried in this order.
JITTERS = [0.0, torch.finfo(torch.float64).tiny] + [
    10.0**e for e in range(-308, 0)
]


def voxel_majority_labels(point_labels: torch.Tensor,
                          point2voxel_map: torch.Tensor, num_voxels: int,
                          num_labels: int) -> torch.Tensor:
    """Label of each voxel, by majority vote of its points' labels.

    Args:
        point_labels: [N] mapped labels in [0, num_labels).
        point2voxel_map: [N] voxel index of each point (any int/float dtype).
        num_voxels: number of voxels V.
        num_labels: number of label values (25 for DSO: 24 classes +
            ignore).

    Returns:
        [V] int64. Ties go to the lowest label (``argmax`` returns the first
        maximum).
    """
    if point_labels.shape != point2voxel_map.shape:
        raise ValueError(f'{point_labels.shape[0]} point labels but '
                         f'{point2voxel_map.shape[0]} point-to-voxel '
                         'entries')
    votes = torch.zeros(num_voxels, num_labels, device=point_labels.device)
    votes.index_add_(0, point2voxel_map.long(),
                     F.one_hot(point_labels.long(), num_labels).float())
    empty = (votes.sum(dim=1) == 0).nonzero().flatten()
    if len(empty):
        raise ValueError(f'{len(empty)} voxels have no points (first: '
                         f'{empty[:5].tolist()}): point2voxel_map does not '
                         'match the voxel features')
    return votes.argmax(dim=1)


class GaussianStats:
    """Running per-class count, sum z and sum z z^T, in float64.

    Args:
        num_classes: classes with a Gaussian (labels 0..num_classes-1).
        dim: feature dimension D.
        device: where the sums live (the features' device).
    """

    def __init__(self, num_classes: int, dim: int, device='cpu') -> None:
        kw = dict(dtype=torch.float64, device=device)
        self.count = torch.zeros(num_classes, **kw)
        self.sum = torch.zeros(num_classes, dim, **kw)
        self.outer = torch.zeros(num_classes, dim, dim, **kw)

    def update(self, features: torch.Tensor, labels: torch.Tensor,
               ignore_index: int) -> None:
        """Add voxels (features [V, D], labels [V]); skip ``ignore_index``."""
        if features.shape[1] != self.sum.shape[1]:
            raise ValueError(f'features have {features.shape[1]} dimensions, '
                             f'the sums {self.sum.shape[1]}')
        keep = labels != ignore_index
        z = features[keep].double()
        y = labels[keep].long()
        num_classes = self.count.shape[0]
        if len(y) and (int(y.min()) < 0 or int(y.max()) >= num_classes):
            raise ValueError(f'labels must be in [0, {num_classes}) or '
                             f'{ignore_index}, got {int(y.min())}..'
                             f'{int(y.max())}')
        for c in torch.unique(y).tolist():
            zc = z[y == c]
            self.count[c] += zc.shape[0]
            self.sum[c] += zc.sum(dim=0)
            self.outer[c] += zc.t() @ zc


def cholesky_with_jitter(cov: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Lower Cholesky factor of ``cov + jitter * I`` (float64), with the
    first jitter of ``JITTERS`` that makes it succeed.

    Args:
        cov: [D, D] float64 tensor on CPU. cuSOLVER-backed ops (cholesky,
            inverse, eigh) fail on sm_89 GPUs with torch 1.10.1+cu111, so
            call finalize to move sums to CPU before passing here.

    Returns:
        (chol, jitter): Cholesky factor and the jitter used, both on CPU.
    """
    eye = torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    for jitter in JITTERS:
        chol, info = torch.linalg.cholesky_ex(cov + jitter * eye)
        if int(info) == 0 and bool(torch.isfinite(chol).all()):
            return chol, jitter
    raise ValueError('the covariance is not positive definite even with '
                     f'jitter {JITTERS[-1]}')


def finalize(stats: GaussianStats, min_count: int) -> Dict[str, torch.Tensor]:
    """Turn the running sums into Gaussians, all float64 on CPU.

    Sums are moved to CPU before factorization because cuSOLVER-backed ops
    (cholesky, inverse, eigh) fail on sm_89 GPUs with torch 1.10.1+cu111.
    The returned dict is entirely on CPU, regardless of where stats live.

    Args:
        stats: GaussianStats with count, sum, outer (may be on any device).
        min_count: minimum number of voxels per class.

    Returns a dict with:
    - means [C, D];
    - covs [C, D, D], with the N-1 correction and no jitter;
    - prec_chol [C, D, D]: P = L^-T of cov + jitter, so (z - mu) P has
      identity covariance;
    - log_det_prec [C]: sum log diag P, which is -0.5 log det Sigma;
    - log_prior [C]: log(n_c / sum n);
    - counts [C] and jitter [C].

    All tensors are float64 on CPU.
    """
    if min_count < 2:
        raise ValueError('min_count must be at least 2 (N-1 covariance)')
    n = stats.count.cpu()
    too_few = [(c, int(n[c])) for c in range(len(n)) if n[c] < min_count]
    if too_few:
        raise ValueError(f'classes with fewer than {min_count} voxels '
                         f'(class, count): {too_few}')
    sum_cpu = stats.sum.cpu()
    outer_cpu = stats.outer.cpu()
    means = sum_cpu / n[:, None]
    covs = (outer_cpu - n[:, None, None] * means[:, :, None] *
            means[:, None, :]) / (n[:, None, None] - 1)
    covs = 0.5 * (covs + covs.transpose(1, 2))
    prec_chol, log_det_prec, jitters = [], [], []
    for c in range(len(n)):
        chol, jitter = cholesky_with_jitter(covs[c])
        eye = torch.eye(chol.shape[0], dtype=chol.dtype, device=chol.device)
        chol_inv = torch.triangular_solve(eye, chol, upper=False).solution
        prec = chol_inv.t().contiguous()
        prec_chol.append(prec)
        log_det_prec.append(torch.log(torch.diagonal(prec)).sum())
        jitters.append(jitter)
    return dict(
        means=means,
        covs=covs,
        prec_chol=torch.stack(prec_chol),
        log_det_prec=torch.stack(log_det_prec),
        log_prior=torch.log(n / n.sum()),
        counts=n.clone(),
        jitter=torch.tensor(jitters, dtype=torch.float64))
