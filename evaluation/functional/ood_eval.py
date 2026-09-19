"""Point-level OOD detection metrics (AUROC, AP, FPR@95).

Histogram-based so that ~5*10^8 points fit in memory without giant sorts:
scores are bucketed into ``num_bins`` equal-width bins between the global
min and max; all three metrics are computed tie-aware from the two
(ID / OOD) histograms. Convention: higher score = more OOD; OOD is the
positive class.
"""
from typing import Dict, List

import numpy as np


def binary_ood_metrics(scores_chunks: List[np.ndarray],
                       labels_chunks: List[np.ndarray],
                       num_bins: int = 2**20) -> Dict[str, float]:
    """Compute AUROC / AP / FPR@95TPR for one score over chunked data.

    Args:
        scores_chunks: list of 1-D float arrays (higher = more OOD).
        labels_chunks: matching list of 1-D bool arrays (True = OOD).
        num_bins: histogram resolution.

    Returns:
        dict with float keys ``auroc``, ``ap``, ``fpr95`` in [0, 1].
    """
    if len(scores_chunks) != len(labels_chunks):
        raise ValueError('scores and labels chunk counts differ')
    smin, smax = np.inf, -np.inf
    for s in scores_chunks:
        if s.size == 0:
            continue
        if np.isnan(s).any():
            raise ValueError('NaN OOD scores')
        smin = min(smin, float(s.min()))
        smax = max(smax, float(s.max()))
    if not (np.isfinite(smin) and np.isfinite(smax)):
        raise ValueError('no finite OOD scores')
    if smax <= smin:
        smax = smin + 1.0  # all scores identical: one occupied bin
    scale = (num_bins - 1) / (smax - smin)

    hist_pos = np.zeros(num_bins, dtype=np.int64)
    hist_neg = np.zeros(num_bins, dtype=np.int64)
    for s, y in zip(scores_chunks, labels_chunks):
        if s.shape != y.shape:
            raise ValueError('scores/labels shape mismatch')
        if s.size == 0:
            continue
        y = y.astype(bool)
        b = ((s.astype(np.float64) - smin) * scale).astype(np.int64)
        np.clip(b, 0, num_bins - 1, out=b)
        hist_pos += np.bincount(b[y], minlength=num_bins)
        hist_neg += np.bincount(b[~y], minlength=num_bins)
    return metrics_from_histograms(hist_pos, hist_neg)


def metrics_from_histograms(hist_pos: np.ndarray,
                            hist_neg: np.ndarray) -> Dict[str, float]:
    """AUROC / AP / FPR@95TPR from per-bin OOD (positive) and ID counts.

    The bins must be equal-width in ascending score order (higher = more
    OOD); every point in a bin is treated as tied with the others in it.
    """
    hist_pos = np.asarray(hist_pos, dtype=np.int64)
    hist_neg = np.asarray(hist_neg, dtype=np.int64)
    if hist_pos.shape != hist_neg.shape or hist_pos.ndim != 1:
        raise ValueError('histograms must be 1-D and of equal length')
    num_bins = hist_pos.shape[0]
    n_pos = int(hist_pos.sum())
    n_neg = int(hist_neg.sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            f'need both OOD and ID points, got {n_pos} OOD / {n_neg} ID')

    # AUROC, tie-aware: every positive scores above the negatives in
    # strictly lower bins and gets half credit against same-bin negatives.
    neg_strictly_below = np.cumsum(hist_neg) - hist_neg
    auroc = float(
        (hist_pos * (neg_strictly_below + 0.5 * hist_neg)).sum() /
        (float(n_pos) * float(n_neg)))

    # Descending-threshold cumulative counts (one cut per bin).
    hp = hist_pos[::-1].astype(np.float64)
    hn = hist_neg[::-1].astype(np.float64)
    tp = np.cumsum(hp)
    fp = np.cumsum(hn)
    tpr = tp / n_pos
    fpr = fp / n_neg

    # AP, sklearn-style step integration with ties grouped per bin.
    denom = tp + fp
    precision = np.divide(tp, denom, out=np.zeros_like(tp),
                          where=denom > 0)
    contributes = hp > 0
    ap = float((hp[contributes] / n_pos * precision[contributes]).sum())

    # FPR at the loosest threshold reaching TPR >= 0.95.
    k = int(np.searchsorted(tpr, 0.95, side='left'))
    k = min(k, num_bins - 1)
    fpr95 = float(fpr[k])

    return dict(auroc=auroc, ap=ap, fpr95=fpr95)


def ood_point_eval(scores: Dict[str, List[np.ndarray]],
                   labels: List[np.ndarray],
                   logger=None) -> Dict[str, float]:
    """Evaluate several score methods and log a result table.

    Args:
        scores: method name -> list of per-scan score arrays.
        labels: list of per-scan bool arrays (True = OOD).
        logger: optional ``logging.Logger``-like; ``print`` if None.

    Returns:
        flat dict ``{method}_AUROC`` / ``{method}_AP`` / ``{method}_FPR95``
        in percent, rounded to 4 decimals.
    """

    def _log(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    n_ood = int(sum(int(lab.sum()) for lab in labels))
    n_id = int(sum(int((~lab.astype(bool)).sum()) for lab in labels))
    _log(f'Point-level OOD evaluation: {n_id} ID points, '
         f'{n_ood} OOD points ({n_ood / max(n_id + n_ood, 1):.4%} OOD)')

    results = dict()
    header = f'{"method":>10} | {"AUROC":>8} | {"AP":>8} | {"FPR@95":>8}'
    _log(header)
    _log('-' * len(header))
    for method, chunks in scores.items():
        m = binary_ood_metrics(chunks, labels)
        results[f'{method}_AUROC'] = round(100.0 * m['auroc'], 4)
        results[f'{method}_AP'] = round(100.0 * m['ap'], 4)
        results[f'{method}_FPR95'] = round(100.0 * m['fpr95'], 4)
        _log(f'{method:>10} | {results[f"{method}_AUROC"]:8.2f} | '
             f'{results[f"{method}_AP"]:8.2f} | '
             f'{results[f"{method}_FPR95"]:8.2f}')
    return results
