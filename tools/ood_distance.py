#!/usr/bin/env python
"""Point-level OOD in the repo's standard protocol: the feature-distance score
(class-conditional Mahalanobis) AND the softmax hierarchy sweep (all 203 set
partitions of the six base groups), so both families can be compared in one run.

Distance (``--part distance``). On the auxiliary semantic branch's penultimate
feature (``pe_features``, 256-d) fit one Gaussian per training class with a
shared (tied) covariance from the inliers (pass 1), then score every point by
the smallest Mahalanobis distance to a class mean (pass 2),
``maha(f) = min_c (f - mu_c)^T S^-1 (f - mu_c)`` -- higher = more OOD. This
catches OOD that the network mis-classifies confidently as an inlier (softmax
looks normal) but whose feature is far from the inlier manifold.

Softmax hierarchy sweep (``--part softmax``). From the head's ``sem_preds`` all
203 partitions of the six base groups {v,h,g,c,n,o} are scored with
``group_msp`` and ``gn_msp`` (group-normalised, chance floor K_g/C), matching
``p3former.utils.ood_scores`` (sign ``-max``). Reports the two reference
hierarchies -- base-6 ``v|h|g|c|n|o`` and c5 keep-vehicle ``v|hgcno`` -- against
flat ``msp``, plus the FPR@95 landscape (spread by number of blocks, robust
worst-domain champion). Same 203-partition set as
``tools/make_dso_hierarchy_variants.py``; this is the one-run overview, the
per-hierarchy authority remains the ``test.py`` + ``summarize`` pipeline.

Consistency. Ground truth and metrics are taken verbatim from
``evaluation.metrics._OODPointMetric`` / ``evaluation.functional.ood_eval``:
OOD = raw semantic ids in ``--ood-raw-ids`` (DSO default 17 Stop, 28 Others),
ID = points whose mapped train label is not ``--ignore-index``, everything else
excluded; AUROC / AP / FPR@95 use the same tie-aware histogram estimator.

HOW TO RUN (from the repo root; same config + checkpoint pair as test.py, one GPU):

    # both families, DSO test + Cetran (the paper split)
    CUDA_VISIBLE_DEVICES=0 python tools/ood_distance.py \\
        configs/p3former/p3former_2xb1_3x_dso_ood.py \\
        work_dirs/p3former_2xb1_3x_dso/epoch_36.pth \\
        --ann dso_infos_test_cetran.pkl --part all

    # distance only / softmax sweep only
    ... --ann dso_infos_test_cetran.pkl --part distance
    ... --ann dso_infos_test_cetran.pkl --part softmax

    # held-out DSO test split instead of test + Cetran
    ... --ann dso_infos_test.pkl --part all

    # quick check on the first N frames (verify the OOD point count is non-zero)
    ... --ann dso_infos_test_cetran.pkl --part all --smoke 120

Arguments:  positional  = config, checkpoint (as test.py).
  --part {distance,softmax,all}  which family to run (default all).
  --ann FILE     override the eval annotation pkl (default = the config's).
  --smoke N      cap frames per pass for a fast check (0 = full split).
  --out FILE.npz dump the 203-partition sweep histograms.
  --ood-raw-ids / --ignore-index / --num-classes / --reg  override the DSO defaults.
Notes: --part distance needs two passes (fit + score); --part all is one extra
GPU pass; the 203-partition sweep is a per-scan CPU loop, so the full split
(~3.6k frames) takes ~30-45 min. Uses ~25 GB of GPU memory (as test.py).
"""
import argparse
import importlib
from typing import Dict, List

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from mmdet3d.registry import MODELS

from p3former.utils.ood_scores import compute_ood_scores
from evaluation.functional.ood_eval import ood_point_eval

# Six base groups, identical to configs/p3former/p3former_2xb1_3x_dso_ood.py.
INIT = ['v', 'h', 'g', 'c', 'n', 'o']
CLASS_GROUPS = [
    [0, 1, 2, 3, 4],           # v vehicle
    [5, 6],                    # h human
    [9, 10, 11, 19],           # g ground
    [12, 13, 14, 15, 16, 17],  # c construction
    [20, 21, 22],              # n nature
    [7, 8, 18, 23],            # o object
]
SWEEP_BINS = 2**16             # histogram resolution for the 203-partition sweep


def parse_args():
    p = argparse.ArgumentParser(
        description='Feature-distance (Mahalanobis) + softmax 203-partition '
        'hierarchy sweep, evaluated with the standard _OODPointMetric protocol.')
    p.add_argument('config', help='test config file path (same as test.py)')
    p.add_argument('checkpoint', help='checkpoint file')
    p.add_argument('--part', choices=['distance', 'softmax', 'all'], default='all',
                   help='distance = Mahalanobis; softmax = 203-partition sweep; all = both')
    p.add_argument('--ann', default=None,
                   help='override test_dataloader ann_file (e.g. test+Cetran pkl)')
    p.add_argument('--ood-raw-ids', type=int, nargs='+', default=[17, 28],
                   help='raw semantic ids treated as OOD (DSO: 17 Stop, 28 Others)')
    p.add_argument('--seg-offset', type=int, default=2**16,
                   help='panoptic packing offset (raw semantic = label %% offset)')
    p.add_argument('--ignore-index', type=int, default=24,
                   help='mapped train id of ignored points (DSO 24)')
    p.add_argument('--num-classes', type=int, default=24,
                   help='number of ID classes (drops the ignore logit channel)')
    p.add_argument('--reg', type=float, default=1e-3,
                   help='diagonal covariance regulariser (fraction of mean variance)')
    p.add_argument('--smoke', type=int, default=0, help='>0: cap frames per pass')
    p.add_argument('--out', default=None, help='optional .npz dump of the sweep histograms')
    return p.parse_args()


# --------------------------- model / data (as test.py) ---------------------------
def build(args):
    cfg = Config.fromfile(args.config)
    for module in cfg.get('custom_imports', dict()).get('imports', []):
        importlib.import_module(module)
    init_default_scope('mmdet3d')
    if args.ann is not None:
        cfg.test_dataloader.dataset.dataset.ann_file = args.ann
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = 4
    model = MODELS.build(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.cuda().eval()
    cap = dict()
    _orig = model.decode_head.init_inputs

    def _patched(*a, **k):
        out = _orig(*a, **k)
        cap['pe'] = out[1]      # pe_features (per-voxel, 256-d)
        cap['sem'] = out[3]     # sem_preds   (per-voxel class logits)
        return out

    model.decode_head.init_inputs = _patched
    dl = Runner.build_dataloader(cfg.test_dataloader)
    return model, dl, cap


def point_tensors(ds, cap):
    p2v = ds.gt_pts_seg.point2voxel_map.long()
    return cap['pe'][0][p2v].double(), cap['sem'][0][p2v]


def gt_masks(ds, args):
    """ID/OOD/valid masks, identical to _OODPointMetric.process()."""
    ea = ds.eval_ann_info
    raw_sem = np.asarray(ea['pts_instance_mask']) % args.seg_offset
    mapped = np.asarray(ea['pts_semantic_mask'])
    ood = np.isin(raw_sem, args.ood_raw_ids)
    valid = (mapped != args.ignore_index) | ood
    return mapped, ood, valid


# --------------------------- 203-partition sweep machinery ---------------------------
def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for smaller in set_partitions(rest):
        yield [[first]] + smaller
        for i in range(len(smaller)):
            yield smaller[:i] + [[first] + smaller[i]] + smaller[i + 1:]


def sweep_setup(num_classes):
    parts = list(set_partitions(list(range(6))))                 # 203
    pmask = [np.array([sum(1 << i for i in blk) for blk in part], np.int64) for part in parts]
    nblk = np.array([len(p) for p in parts])
    pname = ['|'.join(''.join(INIT[i] for i in blk) for blk in part) for part in parts]
    mmat = np.zeros((6, 64), np.float64)                         # subset-mask membership
    for m in range(64):
        for i in range(6):
            if m >> i & 1:
                mmat[i, m] = 1.0
    ksz = np.array([len(g) for g in CLASS_GROUPS], np.float64)   # K per base group
    kmask = ksz @ mmat                                          # classes per subset-mask
    return dict(parts=parts, pmask=pmask, nblk=nblk, pname=pname,
                mmat=mmat, kmask=kmask, prior=kmask / num_classes)


def metrics_from_hist(hpos, hneg):
    """AUROC / AP / FPR@95 from two histograms -- same math as binary_ood_metrics."""
    n_pos, n_neg = int(hpos.sum()), int(hneg.sum())
    if n_pos == 0 or n_neg == 0:
        return (float('nan'),) * 3
    neg_below = np.cumsum(hneg) - hneg
    auroc = (hpos * (neg_below + 0.5 * hneg)).sum() / (float(n_pos) * float(n_neg))
    hp = hpos[::-1].astype(np.float64); hn = hneg[::-1].astype(np.float64)
    tp, fp = np.cumsum(hp), np.cumsum(hn)
    tpr, fpr = tp / n_pos, fp / n_neg
    denom = tp + fp
    prec = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    contributes = hp > 0
    ap = (hp[contributes] / n_pos * prec[contributes]).sum()
    k = min(int(np.searchsorted(tpr, 0.95, side='left')), len(tpr) - 1)
    return auroc * 100, ap * 100, fpr[k] * 100


def _binidx(score):                                             # score in [-1, 0] -> bin
    return np.clip(((score + 1.0) * (SWEEP_BINS - 1)).astype(np.int64), 0, SWEEP_BINS - 1)


def main():
    args = parse_args()
    need_dist = args.part in ('distance', 'all')
    need_soft = args.part in ('softmax', 'all')
    model, dl, cap = build(args)
    C, D = args.num_classes, 256

    # ---- PASS 1 (distance only): per-class means + tied covariance from inliers ----
    if need_dist:
        Sx = torch.zeros(C, D, dtype=torch.float64, device='cuda')
        nc = torch.zeros(C, dtype=torch.float64, device='cuda')
        G = torch.zeros(D, D, dtype=torch.float64, device='cuda')
        n_id = 0
        print('pass 1/2: fitting the inlier Gaussian ...', flush=True)
        with torch.no_grad():
            for i, data in enumerate(dl):
                for ds in model.test_step(data):
                    feat, _ = point_tensors(ds, cap)
                    mapped, ood, _ = gt_masks(ds, args)
                    if feat.shape[0] != mapped.shape[0]:
                        continue
                    idm = (mapped != args.ignore_index) & (~ood)
                    if not idm.any():
                        continue
                    f = feat[torch.from_numpy(idm).cuda()]
                    lab = torch.from_numpy(mapped[idm]).long().cuda()
                    G += f.T @ f
                    n_id += f.shape[0]
                    for c in range(C):
                        m = lab == c
                        if m.any():
                            Sx[c] += f[m].sum(0)
                            nc[c] += int(m.sum())
                if (i + 1) % 400 == 0:
                    print(f'  {i + 1} frames, n_id={n_id:,}', flush=True)
                if args.smoke and i + 1 >= args.smoke:
                    break
        mu = Sx / nc.clamp_min(1).unsqueeze(1)
        cov = G / max(n_id, 1) - (nc.unsqueeze(1) * mu).T @ mu / max(n_id, 1)
        cov = cov + torch.eye(D, device='cuda', dtype=torch.float64) * (
            args.reg * torch.diagonal(cov).mean())
        prec = torch.linalg.inv(cov)
        const = (mu @ prec * mu).sum(1)
        print(f'  done: n_id={n_id:,}, classes present {(nc > 0).sum().item()}/{C}', flush=True)

    # ---- softmax sweep buffers ----
    if need_soft:
        sw = sweep_setup(C)
        NP = len(sw['parts'])
        Hg = [np.zeros(SWEEP_BINS, np.int64) for _ in range(NP)]   # group_msp OOD hist
        Hg0 = [np.zeros(SWEEP_BINS, np.int64) for _ in range(NP)]  # group_msp ID hist
        Hn = [np.zeros(SWEEP_BINS, np.int64) for _ in range(NP)]   # gn_msp OOD hist
        Hn0 = [np.zeros(SWEEP_BINS, np.int64) for _ in range(NP)]  # gn_msp ID hist
        base6_i = sw['pname'].index('v|h|g|c|n|o')
        c5_i = next(i for i, nm in enumerate(sw['pname']) if set(nm.split('|')) == {'v', 'hgcno'})

    # ---- key per-point methods evaluated with his ood_point_eval ----
    key_methods = (['mahalanobis'] if need_dist else []) + \
        ['msp', 'group_msp_base6', 'gn_msp_base6', 'group_msp_c5', 'gn_msp_c5']
    scores: Dict[str, List[np.ndarray]] = {m: [] for m in key_methods}
    labels: List[np.ndarray] = []
    prior6 = torch.tensor([len(g) for g in CLASS_GROUPS], dtype=torch.float64, device='cuda') / C

    print('pass 2/2: scoring ...', flush=True)
    with torch.no_grad():
        for i, data in enumerate(dl):
            for ds in model.test_step(data):
                feat, logits = point_tensors(ds, cap)
                mapped, ood, valid = gt_masks(ds, args)
                if feat.shape[0] != mapped.shape[0]:
                    continue
                p = torch.softmax(logits[:, :C], dim=1)
                Bg = torch.stack([p[:, g].sum(1) for g in CLASS_GROUPS], dim=1)   # [N,6]

                col = dict()
                col['msp'] = -p.max(1).values
                col['group_msp_base6'] = -Bg.max(1).values
                col['gn_msp_base6'] = -(Bg - prior6.unsqueeze(0)).clamp_min(0).max(1).values
                merged = Bg[:, 1:].sum(1)                                         # hgcno
                c5p = torch.stack([Bg[:, 0], merged], dim=1)
                c5prior = torch.tensor([prior6[0].item(), prior6[1:].sum().item()],
                                       dtype=torch.float64, device='cuda')
                col['group_msp_c5'] = -c5p.max(1).values
                col['gn_msp_c5'] = -(c5p - c5prior.unsqueeze(0)).clamp_min(0).max(1).values
                if need_dist:
                    A = feat @ prec
                    maha = (A * feat).sum(1, keepdim=True) - 2 * (A @ mu.T) + const.unsqueeze(0)
                    col['mahalanobis'] = maha.min(1).values.clamp_min(0).sqrt()
                for m in key_methods:
                    scores[m].append(col[m].cpu().numpy().astype(np.float32)[valid])
                labels.append(ood[valid])

                if need_soft:
                    idv = valid & (~ood); oov = ood                              # ID / OOD points
                    Bg_np = Bg.cpu().numpy()
                    PB = Bg_np @ sw['mmat']                                       # [N,64]
                    EX = np.clip(PB - sw['kmask'] / C, 0.0, None)
                    for pop, mask, Hgd, Hnd in ((idv, idv, Hg0, Hn0), (oov, oov, Hg, Hn)):
                        if not mask.any():
                            continue
                        pbm, exm = PB[mask], EX[mask]
                        for pi in range(NP):
                            ms = sw['pmask'][pi]
                            gi = _binidx(-pbm[:, ms].max(1))
                            ni = _binidx(-exm[:, ms].max(1))
                            Hgd[pi] += np.bincount(gi, minlength=SWEEP_BINS)
                            Hnd[pi] += np.bincount(ni, minlength=SWEEP_BINS)
            if (i + 1) % 400 == 0:
                print(f'  {i + 1} frames', flush=True)
            if args.smoke and i + 1 >= args.smoke:
                break

    # ---- report: key methods via his exact estimator ----
    print('\n== key scores (his _OODPointMetric estimator) ==', flush=True)
    ood_point_eval(scores, labels)

    # ---- report: 203-partition sweep landscape ----
    if need_soft:
        gm = np.array([metrics_from_hist(Hg[i], Hg0[i]) for i in range(NP)])      # [NP,3] AUROC/AP/FPR
        nm = np.array([metrics_from_hist(Hn[i], Hn0[i]) for i in range(NP)])
        nblk = sw['nblk']; pname = sw['pname']
        print('\n== 203-partition sweep (group_msp | gn_msp  AUROC/AP/FPR95) ==')
        for tag, i in (('base-6 v|h|g|c|n|o', base6_i), ('c5     v|hgcno', c5_i)):
            print(f'  {tag:20} group {gm[i,0]:5.1f}/{gm[i,1]:5.1f}/{gm[i,2]:5.1f}   '
                  f'gn {nm[i,0]:5.1f}/{nm[i,1]:5.1f}/{nm[i,2]:5.1f}')
        print('  gn_msp FPR95 by #blocks (min / median / max over partitions):')
        for b in range(1, 7):
            idx = np.where(nblk == b)[0]; v = nm[idx, 2]
            print(f'    {b} blocks ({len(idx):3}): {v.min():5.1f} / {np.median(v):5.1f} / {v.max():5.1f}')
        wi = int(np.argmin(nm[:, 2]))
        print(f'  best gn_msp FPR95 partition: [{pname[wi]}] ({nblk[wi]} blocks) '
              f'AUROC {nm[wi,0]:.1f} FPR95 {nm[wi,2]:.1f}')
        if args.out:
            np.savez(args.out, pname=np.array(pname), nblk=nblk, group=gm, gn=nm)
            print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()
