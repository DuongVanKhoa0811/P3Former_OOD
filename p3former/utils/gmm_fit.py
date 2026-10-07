"""Class-conditional Gaussians for the OCCUQ density score, from streamed
voxel features.

OCCUQ (official code ``tools/gmm_utils.py:163-220`` at commit 2aa5429)
stores up to 200M float32 feature vectors per class and calls
``torch.cov``. Here each class keeps running float64 sums instead: the
count, sum z and sum z z^T. So every training voxel is used, with 12.6 MB
of state for 24 classes x 256 dimensions.

``finalize`` turns the sums into, per class:
- the mean, and the N-1 covariance (``torch.cov``'s convention);
- the precision Cholesky factor P = L^-T of the covariance plus one
  diagonal shared by all classes: a ridge of 1e-6 x the mean pooled ID
  variance, plus OCCUQ's jitter, searched once for all classes;
- its log-determinant and the log class prior.

The shared ridge exists because of dead feature dimensions. In the base
checkpoint, ``pe_features`` dim 190 is always 0 (the ``pe_conv`` LayerNorm
gives it gamma 0.0291 and beta -0.3372 before a ReLU), so every class
covariance is singular. A jitter alone would be 2.2e-308, which makes P
~6.7e153: +inf in float32, so the density is NaN. Per-class jitters would
add class-dependent offsets of -0.5 log(jitter_c), up to 354 nats.

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
    """Lower Cholesky factors of ``cov + jitter * I`` (float64), with the
    first jitter of ``JITTERS`` for which every matrix factors.

    Args:
        cov: [D, D] or a batch [..., D, D], float64 on CPU. One jitter is
            shared by the whole batch, as in OCCUQ. cuSOLVER-backed ops
            (cholesky, inverse, eigh) fail on sm_89 GPUs with torch
            1.10.1+cu111, so call finalize to move sums to CPU before
            passing here.

    Returns:
        (chol, jitter): the Cholesky factors, shaped like ``cov``, and the
        jitter used, both on CPU.
    """
    eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
    for jitter in JITTERS:
        chol, info = torch.linalg.cholesky_ex(cov + jitter * eye)
        if bool((info == 0).all()) and bool(torch.isfinite(chol).all()):
            return chol, jitter
    raise ValueError('the covariance is not positive definite even with '
                     f'jitter {JITTERS[-1]}')


def finalize(stats: GaussianStats, min_count: int,
             ridge_rel: float = 1e-6) -> Dict[str, torch.Tensor]:
    """Turn the running sums into Gaussians, all float64 on CPU.

    Regularisation: every class covariance gets the same diagonal,
    (ridge + jitter) I.
    - ridge = ridge_rel x the mean over dimensions of the pooled ID
      variance, the variance of all classes' voxels together;
    - jitter is the first value of ``JITTERS`` for which every class
      factors: OCCUQ's jitter search, with one jitter for all classes.
    A dead feature dimension, 0 in every class, then has variance ridge in
    every class. It adds the same constant to every class's log-density, and
    the float32 precision factors stay finite. A jitter alone (2.2e-308)
    would overflow float32, and per-class jitters would add class-dependent
    offsets of up to 354 nats.

    Sums are moved to CPU before factorization because cuSOLVER-backed ops
    (cholesky, inverse, eigh) fail on sm_89 GPUs with torch 1.10.1+cu111.
    The returned dict is entirely on CPU, regardless of where stats live.

    Args:
        stats: GaussianStats with count, sum, outer (may be on any device).
        min_count: minimum number of voxels per class.
        ridge_rel: the ridge relative to the mean pooled variance.

    Returns a dict with:
    - means [C, D];
    - covs [C, D, D], with the N-1 correction and no regularisation;
    - prec_chol [C, D, D]: P = L^-T of covs + jitter I, so (z - mu) P has
      identity covariance;
    - log_det_prec [C]: sum log diag P, which is -0.5 log det(covs +
      jitter I);
    - log_prior [C]: log(n_c / sum n);
    - counts [C];
    - jitter [C]: the whole diagonal added, ridge + the shared jitter, the
      same for every class;
    - ridge: 0-d;
    - pooled_var [D]: the pooled ID variance per dimension (N-1).

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
    total = n.sum()
    pooled_mean = sum_cpu.sum(dim=0) / total
    pooled_var = (torch.diagonal(outer_cpu, dim1=1, dim2=2).sum(dim=0) -
                  total * pooled_mean**2) / (total - 1)
    mean_var = pooled_var.mean()
    if not float(mean_var) > 0:
        raise ValueError(f'the mean pooled ID variance is {float(mean_var)}: '
                         'the features are constant (or not finite)')
    ridge = ridge_rel * mean_var
    eye = torch.eye(covs.shape[-1], dtype=covs.dtype)
    chol, jitter = cholesky_with_jitter(covs + ridge * eye)
    chol_inv = torch.triangular_solve(eye.expand_as(chol), chol,
                                      upper=False).solution
    prec_chol = chol_inv.transpose(1, 2).contiguous()
    return dict(
        means=means,
        covs=covs,
        prec_chol=prec_chol,
        log_det_prec=torch.log(torch.diagonal(prec_chol, dim1=1,
                                              dim2=2)).sum(dim=1),
        log_prior=torch.log(n / total),
        counts=n.clone(),
        jitter=(ridge + jitter).repeat(len(n)),
        ridge=ridge,
        pooled_var=pooled_var)
