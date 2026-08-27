"""Walkthrough of the histogram-based AUROC / AP / FPR@95TPR block from
docs/superpowers/plans/2026-08-12-ood-baselines.md.

Setup (recap): every point got an OOD score, scores were quantized into
`num_bins` bins, and two histograms were accumulated:

    hist_pos[i] = number of OOD points (positives) whose score fell in bin i
    hist_neg[i] = number of ID  points (negatives) whose score fell in bin i

Higher bin index = higher score = "more OOD". All three metrics are computed
from these two arrays alone; points in the same bin are treated as exactly
tied. This file replays each vectorized step on a tiny 8-bin example, prints
every intermediate array, and then re-derives the same numbers with slow
explicit loops (plus sklearn, if installed) to show they agree.

Run:  python trash/explain_ood_metrics.py
"""
import numpy as np

# --------------------------------------------------------------------------
# Tiny example: 4 OOD points and 8 ID points, already binned into 8 bins.
# OOD points sit in bins [6, 6, 3, 1]; ID points in [0, 0, 1, 2, 2, 3, 5, 6].
# A good detector would put OOD in high bins and ID in low bins — here the
# separation is decent but imperfect (bins 1, 3, 6 are contested).
# --------------------------------------------------------------------------
num_bins = 8
hist_pos = np.array([0, 1, 0, 1, 0, 0, 2, 0], dtype=np.int64)
hist_neg = np.array([2, 1, 2, 1, 0, 1, 1, 0], dtype=np.int64)
n_pos = int(hist_pos.sum())   # 4
n_neg = int(hist_neg.sum())   # 8

print('bin index :', np.arange(num_bins))
print('hist_pos  :', hist_pos, f' (n_pos={n_pos})')
print('hist_neg  :', hist_neg, f' (n_neg={n_neg})')

# ==========================================================================
# 1) AUROC as a rank statistic (Mann-Whitney U).
#
# AUROC == P(random OOD point scores higher than random ID point), with ties
# counted as half a win. So walk over every (positive, negative) PAIR:
#   - positive in a strictly higher bin  -> 1.0 credit
#   - both in the same bin (tied)        -> 0.5 credit
#   - positive in a lower bin            -> 0.0 credit
# and divide the total credit by the number of pairs, n_pos * n_neg.
#
# The vectorized form counts this per bin:
#   neg_strictly_below[i] = negatives in bins 0..i-1   (EXCLUSIVE prefix sum:
#       cumsum includes bin i itself, so subtract hist_neg to shift it out)
#   each of the hist_pos[i] positives in bin i beats those, and gets half
#   credit against the hist_neg[i] negatives sharing its bin.
# ==========================================================================
neg_strictly_below = np.cumsum(hist_neg) - hist_neg
auroc = float((hist_pos * (neg_strictly_below + 0.5 * hist_neg)).sum()
              / (float(n_pos) * float(n_neg)))

print('\n--- AUROC ---')
print('cumsum(hist_neg)      :', np.cumsum(hist_neg), ' (negatives in bins <= i)')
print('neg_strictly_below    :', neg_strictly_below, ' (negatives in bins <  i)')
print('per-bin pair credit   :', hist_pos * (neg_strictly_below + 0.5 * hist_neg))
print(f'AUROC = total credit / (n_pos*n_neg) = '
      f'{(hist_pos * (neg_strictly_below + 0.5 * hist_neg)).sum()} / {n_pos * n_neg}'
      f' = {auroc:.5f}')

# Same thing, brute force over all pairs (what the formula is shorthand for).
pos_bins = np.repeat(np.arange(num_bins), hist_pos)   # [1, 3, 6, 6]
neg_bins = np.repeat(np.arange(num_bins), hist_neg)   # [0, 0, 1, 2, 2, 3, 5, 6]
credit = 0.0
for p in pos_bins:
    for n in neg_bins:
        credit += 1.0 if p > n else (0.5 if p == n else 0.0)
auroc_brute = credit / (n_pos * n_neg)
print(f'brute-force over {n_pos * n_neg} pairs      = {auroc_brute:.5f}')
assert np.isclose(auroc, auroc_brute)

# ==========================================================================
# 2) The descending threshold sweep shared by AP and FPR@95.
#
# A detector says "OOD" when score >= threshold. Sweep the threshold from
# the highest bin downward — one candidate cut per bin. Reversing the
# histograms ([::-1]) makes index k of the reversed array correspond to the
# cut "predict OOD for everything in the top k+1 bins":
#   tp[k] = positives in the top k+1 bins   (true positives at that cut)
#   fp[k] = negatives in the top k+1 bins   (false positives at that cut)
# cumsum does the "top k+1 bins" accumulation. TPR/FPR just normalize.
# ==========================================================================
hp = hist_pos[::-1].astype(np.float64)   # bin 7 first, bin 0 last
hn = hist_neg[::-1].astype(np.float64)
tp = np.cumsum(hp)
fp = np.cumsum(hn)
tpr = tp / n_pos
fpr = fp / n_neg

print('\n--- Descending threshold sweep ---')
print('cut k covers bins >= :', np.arange(num_bins)[::-1])
print('tp (OOD caught)      :', tp.astype(int))
print('fp (ID mislabeled)   :', fp.astype(int))
print('tpr                  :', np.round(tpr, 3))
print('fpr                  :', np.round(fpr, 3))

# ==========================================================================
# 3) Average precision: step-integrate the precision-recall curve.
#
#   AP = sum over cuts of (recall gained at this cut) * (precision there)
#
# Only cuts where new positives arrive (hp > 0) gain recall, so the others
# contribute nothing — the `contributes` mask skips them. Recall gained at
# cut k is hp[k] / n_pos. Grouping whole bins per cut is exactly how
# sklearn's average_precision_score handles tied scores.
# The np.where guards 0/0 at cuts before anything is predicted positive
# (numpy still evaluates the division, so errstate mutes the 0/0 warning;
# np.where then discards the NaN it produced).
# ==========================================================================
with np.errstate(invalid='ignore'):
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
contributes = hp > 0
ap = float((hp[contributes] / n_pos * precision[contributes]).sum())

print('\n--- Average precision ---')
print('precision at each cut:', np.round(precision, 3))
print('cut contributes (hp>0):', contributes.astype(int))
print('per-cut AP terms     :',
      np.round(np.where(contributes, hp / n_pos * precision, 0.0), 4))
print(f'AP = {ap:.5f}')

# Same thing as an explicit loop from the highest bin down.
ap_loop, tp_run, fp_run = 0.0, 0, 0
for b in range(num_bins - 1, -1, -1):          # threshold sweep, high -> low
    tp_run += hist_pos[b]
    fp_run += hist_neg[b]
    if hist_pos[b] > 0:                        # recall only moves here
        prec_here = tp_run / (tp_run + fp_run)
        ap_loop += (hist_pos[b] / n_pos) * prec_here
print(f'loop form            = {ap_loop:.5f}')
assert np.isclose(ap, ap_loop)

# ==========================================================================
# 4) FPR@95TPR: how many ID points get flagged when the threshold is
# loosened just enough to catch 95% of the OOD points.
#
# tpr is non-decreasing along the sweep, so searchsorted finds the FIRST cut
# with tpr >= 0.95 — i.e. the tightest cut (loosest usable threshold) that
# reaches 95% recall — and we read the FPR there. tpr always ends at 1.0,
# so a valid k always exists; the min() is only a defensive clamp.
# ==========================================================================
k = int(np.searchsorted(tpr, 0.95, side='left'))
k = min(k, num_bins - 1)
fpr95 = float(fpr[k])

print('\n--- FPR@95TPR ---')
print(f'first cut with tpr >= 0.95: k={k} '
      f'(covers bins >= {num_bins - 1 - k}), tpr={tpr[k]:.3f}')
print(f'FPR@95 = {fpr95:.5f}  '
      f'({int(fp[k])} of {n_neg} ID points flagged at that threshold)')

# ==========================================================================
# 5) Cross-check against sklearn on a bigger random problem (optional).
# sklearn sees the bin indices as scores, so ties are grouped identically.
# ==========================================================================
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ImportError:
    print('\n(sklearn not installed - skipping the random cross-check)')
else:
    rng = np.random.default_rng(0)
    nb = 64
    hp_r = rng.poisson(3.0, nb) * (rng.random(nb) < 0.7)
    hn_r = rng.poisson(5.0, nb)
    hp_r[-3:] += 20                            # OOD mass in the top bins
    y = np.concatenate([np.ones(hp_r.sum()), np.zeros(hn_r.sum())])
    s = np.concatenate([np.repeat(np.arange(nb), hp_r),
                        np.repeat(np.arange(nb), hn_r)]).astype(float)

    nsb = np.cumsum(hn_r) - hn_r
    auroc_h = (hp_r * (nsb + 0.5 * hn_r)).sum() / (hp_r.sum() * hn_r.sum())
    tp_r = np.cumsum(hp_r[::-1]); fp_r = np.cumsum(hn_r[::-1])
    prec_r = np.where((tp_r + fp_r) > 0, tp_r / (tp_r + fp_r), 0.0)
    m = hp_r[::-1] > 0
    ap_h = (hp_r[::-1][m] / hp_r.sum() * prec_r[m]).sum()

    print('\n--- sklearn cross-check (64 bins, random histograms) ---')
    print(f'AUROC  histogram={auroc_h:.10f}  sklearn={roc_auc_score(y, s):.10f}')
    print(f'AP     histogram={ap_h:.10f}  sklearn={average_precision_score(y, s):.10f}')
    assert np.isclose(auroc_h, roc_auc_score(y, s))
    assert np.isclose(ap_h, average_precision_score(y, s))

print('\nall checks passed')
