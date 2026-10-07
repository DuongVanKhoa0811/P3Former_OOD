#!/usr/bin/env python
"""Fit OCCUQ's class-conditional Gaussians on the training split.

Runs a P3Former checkpoint over the training split (eval mode, test
pipeline, batch 1) and fits one full-covariance Gaussian per ID class on a
per-voxel feature:
- --features pe: pe_features, the semantic branch's input;
- --features head: the output of the OCCUQ head's last block.

Each voxel is one sample, labelled by the majority vote of its points'
mapped labels. Voxels labelled 24 are skipped: that is ignore, which
includes the OOD classes Stop and Others. The statistics are running
float64 sums, so every voxel is used (p3former/utils/gmm_fit.py).

Right after the pass, the sums go to <out stem>.sums.pth (gmm_pe.pth ->
gmm_pe.sums.pth). If a later step fails, --from-sums finalises that file
without redoing the pass.

The output is a torch.save'd dict of plain tensors, plus the fingerprint of
the weights that produced the features; _P3FormerHead refuses the file for
other weights. Nothing is saved if a float32 Gaussian tensor is not finite,
or, with --check N, if the float32 GPU log-density of N frames is not finite
or differs from float64 on the CPU by more than 0.05 nats. Spec:
docs/superpowers/specs/2026-10-07-occuq-density-design.md.

Run from the repo root (one GPU; ~15-30 min for the 8,474 training frames):
    CUDA_VISIBLE_DEVICES=0 python tools/fit_occuq_gmm.py \\
        configs/p3former/p3former_2xb1_3x_dso_occuq_pe.py \\
        work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --features pe \\
        --out work_dirs/p3former_2xb1_3x_dso_occuq_pe/gmm_pe.pth --check 5
"""
import argparse
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from p3former.utils.gmm_density import GMM_KEYS, log_density  # noqa: E402
from p3former.utils.gmm_fit import (GaussianStats, finalize,  # noqa: E402
                                    voxel_majority_labels)

NUM_CLASSES = 24  # DSO ID classes
IGNORE_INDEX = 24  # mapped label of ignored points, incl. the OOD classes
MIN_COUNT = 2560  # 10 x 256 voxels per class
CHECK_TOL = 0.05  # nats: float32 GPU log-density vs float64 CPU


def build(config, ann, workers):
    """(cfg, model, dataloader) as test.py builds them, batch size 1, on
    ``ann``; the weights are loaded by ``load_weights``."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(config)  # also imports cfg.custom_imports
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    cfg.test_dataloader.dataset.dataset.ann_file = ann
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = workers
    model = MODELS.build(cfg.model)
    return cfg, model, Runner.build_dataloader(cfg.test_dataloader)


def load_weights(model, checkpoint, features):
    """Load ``checkpoint``. Every weight but the OCCUQ head's must be in it,
    and the OCCUQ head's too for --features head."""
    state = torch.load(checkpoint, map_location='cpu')
    state = state.get('state_dict', state)
    missing, _ = model.load_state_dict(state, strict=False)
    head_missing = [k for k in missing if k.startswith('decode_head.occuq_head.')]
    other_missing = [k for k in missing if k not in head_missing]
    if other_missing:
        raise KeyError(f'{len(other_missing)} model weights are missing from '
                       f'{checkpoint}, e.g. {other_missing[:3]}')
    if features == 'head' and head_missing:
        raise ValueError(f'{checkpoint} has no trained OCCUQ head: fit '
                         '--features head on a variant A or C checkpoint')


def frame_voxels(model, data, features):
    """One frame (batch size 1). Returns the voxel features [V, D], the
    voxel labels [V] (majority vote of the points' mapped labels) and the
    OCCUQ head's logits [V, K] (None for --features pe)."""
    data = model.data_preprocessor(data, False)
    inputs, sample = data['inputs'], data['data_samples'][0]
    x = model.extract_feat(inputs)
    head = model.decode_head
    (pe, ) = head.extract_pe_features(x.features,
                                      inputs['voxels']['voxel_coors'])
    logits, feats = (None, pe) if features == 'pe' else head.occuq_head(pe)
    point_labels = torch.as_tensor(
        np.asarray(sample.eval_ann_info['pts_semantic_mask']).astype(np.int64),
        device=feats.device)
    labels = voxel_majority_labels(point_labels,
                                   sample.gt_pts_seg.point2voxel_map,
                                   feats.shape[0], IGNORE_INDEX + 1)
    return feats, labels, logits


def frame_name(data, index):
    """'frame <index> (<lidar path>)' for a batch-size-1 loader batch."""
    path = data['data_samples'][0].metainfo.get('lidar_path')
    return f'frame {index}' + (f' ({path})' if path else '')


def compare_log_densities(fast, exact, frame=''):
    """max |fast - exact| over one frame's voxels, between the float32 GPU
    log-density and the float64 CPU one. Raises if either holds a
    non-finite value, or unless the difference is <= CHECK_TOL, so a NaN
    can never pass."""
    for name, values in (('float32 GPU', fast), ('float64 CPU', exact)):
        bad = int((~torch.isfinite(values)).sum())
        if bad:
            raise FloatingPointError(
                f'{frame}: the {name} log-density has {bad} non-finite '
                f'values out of {values.numel()}; nothing saved')
    if fast.shape != exact.shape:
        raise ValueError(f'{frame}: {tuple(fast.shape)} vs '
                         f'{tuple(exact.shape)} log-densities')
    if not fast.numel():
        return 0.0
    diff = float((fast.double().cpu() - exact.double().cpu()).abs().max())
    if not diff <= CHECK_TOL:
        raise RuntimeError(f'{frame}: the float32 log-density differs from '
                           f'float64 by {diff:.4g} nats (> {CHECK_TOL}); '
                           'nothing saved')
    return diff


def check_precision(model, loader, features, gmm, frames, device):
    """Largest |log q| difference over the first ``frames`` frames between
    the production path (float32 on ``device``, TF32 off) and float64 on
    the CPU. Raises on a non-finite value or above CHECK_TOL."""
    gmm32 = {k: gmm[k].to(device=device, dtype=torch.float32) for k in GMM_KEYS}
    gmm64 = {k: gmm[k].to(dtype=torch.float64).cpu() for k in GMM_KEYS}
    worst = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader):
            if i >= frames:
                break
            name = frame_name(data, i)
            feats, _, _ = frame_voxels(model, data, features)
            fast = log_density(feats.float(), gmm32).cpu()
            exact = log_density(feats.double().cpu(), gmm64)
            worst = max(worst, compare_log_densities(fast, exact, name))
    return worst


def shared_jitter(result):
    """The jitter added on top of the ridge (the same for every class)."""
    return float(result['jitter'][0] - result['ridge'])


def check_float32_finite(result):
    """Raise unless the float32 casts of the GMM_KEYS tensors, which the
    density score uses, are all finite. Names the classes and the
    regularisation."""
    names = result.get('class_names')
    bad = {}
    for key in GMM_KEYS:
        values = result[key].float()
        rows = ~torch.isfinite(values.reshape(values.shape[0], -1)).all(1)
        if rows.any():
            bad[key] = rows.nonzero().flatten().tolist()
    if bad:
        classes = sorted({c for rows in bad.values() for c in rows})
        listed = ', '.join(f'{c} ({names[c]})' if names else str(c)
                           for c in classes)
        raise FloatingPointError(
            f'the float32 Gaussians of classes {listed} are not finite '
            f'(in {", ".join(bad)}), with ridge {float(result["ridge"]):.3e} '
            f'and shared jitter {shared_jitter(result):.3g}; nothing saved')


def report(result):
    """Print the per-class voxel counts, jitters and training accuracy, then
    the regularisation and the pooled ID variance per dimension."""
    acc = result.get('train_accuracy')
    print(f"{'class':>18} | {'voxels':>12} | {'jitter':>8} | {'train acc':>9}")
    for c, name in enumerate(result['class_names']):
        a = f'{100 * float(acc[c]):8.2f}%' if acc is not None else '        -'
        print(f"{name:>18} | {int(result['counts'][c]):12d} | "
              f"{float(result['jitter'][c]):8.1e} | {a}")
    pooled = result['pooled_var']
    mean, ridge = float(pooled.mean()), float(result['ridge'])
    zero = (pooled == 0).nonzero().flatten().tolist()
    print(f'ridge {ridge:.3e} = {ridge / mean:.0e} x the mean pooled variance '
          f'{mean:.4g}; shared jitter {shared_jitter(result):.3g}; every '
          f"class gets {float(result['jitter'][0]):.3e} on its diagonal")
    print(f'pooled variance: min {float(pooled.min()):.3g} at dim '
          f'{int(pooled.argmin())}; {len(zero)} dimensions exactly 0: '
          f'{zero[:20]}{" ..." if len(zero) > 20 else ""}; '
          f'{int((pooled < 1e-12 * mean).sum())} below 1e-12 x the mean')


def sums_path(out):
    """Where the pass's running sums go: <out stem>.sums.pth."""
    return osp.splitext(out)[0] + '.sums.pth'


def save_sums(path, stats, correct, total, *, features, fingerprint,
              checkpoint, config, ann_file, frames):
    """torch.save the running sums (count, sum, outer) and the head's
    accuracy counters (correct, total) as plain float64 CPU tensors, with
    their metadata. Refuses to overwrite."""
    if osp.exists(path):
        raise FileExistsError(f'{path} exists')
    os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
    torch.save(
        dict(count=stats.count.double().cpu(), sum=stats.sum.double().cpu(),
             outer=stats.outer.double().cpu(),
             correct=correct.double().cpu(), total=total.double().cpu(),
             features=features, fingerprint=fingerprint,
             checkpoint=checkpoint, config=config, ann_file=ann_file,
             frames=frames), path)


def load_sums(path, features, expected_fingerprint):
    """(GaussianStats on the CPU, the file's dict) from a save_sums file.
    Refuses sums of other features or of other weights."""
    data = torch.load(path, map_location='cpu')
    if data['features'] != features:
        raise ValueError(f'{path} holds sums of {data["features"]} features, '
                         f'not {features}')
    if data['fingerprint'] != expected_fingerprint:
        raise ValueError(
            f'{path} was collected with other weights (fingerprint '
            f'{data["fingerprint"][:12]}, this model '
            f'{expected_fingerprint[:12]})')
    stats = GaussianStats(data['count'].shape[0], data['sum'].shape[1])
    stats.count, stats.sum, stats.outer = (data['count'], data['sum'],
                                           data['outer'])
    return stats, data


def accumulate(model, loader, features, limit):
    """The pass: running sums over the loader's voxels, plus the OCCUQ
    head's per-class accuracy counters for --features head. Returns
    (stats or None without frames, correct, total, frames)."""
    stats = None
    correct = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    total = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    frames, t0 = 0, time.time()
    with torch.no_grad():
        for data in loader:
            if limit and frames >= limit:
                break
            feats, labels, logits = frame_voxels(model, data, features)
            if stats is None:
                stats = GaussianStats(NUM_CLASSES, feats.shape[1],
                                      feats.device)
            stats.update(feats, labels, IGNORE_INDEX)
            if logits is not None:
                valid = labels != IGNORE_INDEX
                y = labels[valid].cpu()  # CPU: CUDA bincount is slow here
                pred = logits[valid, :NUM_CLASSES].argmax(dim=1).cpu()
                total += torch.bincount(y, minlength=NUM_CLASSES).double()
                correct += torch.bincount(y[pred == y],
                                          minlength=NUM_CLASSES).double()
            frames += 1
            if frames % 500 == 0:
                print(f'{frames} frames ({time.time() - t0:.0f} s)', flush=True)
    print(f'{frames} frames in {time.time() - t0:.0f} s', flush=True)
    return stats, correct, total, frames


def fit(config, checkpoint, out, features, ann='dso_infos_train.pkl',
        check=0, limit=0, workers=4, device='cuda:0', from_sums=None):
    """Fit, check and save the Gaussians; returns the saved dict. The pass
    saves its sums to sums_path(out) first; ``from_sums`` finalises such a
    file instead of running the pass."""
    if osp.exists(out):
        raise FileExistsError(f'{out} exists: delete it or choose another --out')
    sums_file = sums_path(out)
    if from_sums is None and osp.exists(sums_file):
        raise FileExistsError(
            f'{sums_file} exists: finalise it with --from-sums {sums_file}, '
            'or delete it to redo the pass')
    cfg, model, loader = build(config, ann, workers)
    if features == 'head' and model.decode_head.occuq_head is None:
        raise ValueError('--features head needs a config whose '
                         'decode_head.occuq_cfg builds the OCCUQ head')
    load_weights(model, checkpoint, features)
    model.to(device).eval()
    fingerprint = model.decode_head.density_fingerprint(features)
    if from_sums is None:
        stats, correct, total, frames = accumulate(model, loader, features,
                                                   limit)
        if stats is None:
            raise ValueError(f'no frames in {ann}')
        sums = dict(features=features, fingerprint=fingerprint,
                    checkpoint=checkpoint, config=config, ann_file=ann,
                    frames=frames)
        save_sums(sums_file, stats, correct, total, **sums)
        print(f'saved the running sums to {sums_file}', flush=True)
    else:
        stats, sums = load_sums(from_sums, features, fingerprint)
        correct, total = sums['correct'], sums['total']
        print(f'loaded the running sums of {sums["frames"]} frames from '
              f'{from_sums}', flush=True)
    result = {k: v.cpu() for k, v in finalize(stats, MIN_COUNT).items()}
    result.update(
        features=features, fingerprint=fingerprint,
        checkpoint=sums['checkpoint'], config=sums['config'],
        ann_file=sums['ann_file'], frames=sums['frames'],
        class_names=list(cfg.get('class_names',
                                 [str(c) for c in range(NUM_CLASSES)])))
    if features == 'head':
        result['train_accuracy'] = correct / total.clamp(min=1)
    report(result)
    check_float32_finite(result)
    if check:
        worst = check_precision(model, loader, features, result, check, device)
        print(f'--check {check}: max |log q float32 GPU - float64 CPU| = '
              f'{worst:.4g} nats')
    os.makedirs(osp.dirname(osp.abspath(out)), exist_ok=True)
    torch.save(result, out)
    print(f'saved {out}')
    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='an OCCUQ config (as test.py)')
    ap.add_argument('checkpoint', help='checkpoint file')
    ap.add_argument('--features', choices=('head', 'pe'), required=True)
    ap.add_argument('--out', required=True,
                    help='output .pth; refuses to overwrite it, and the '
                    'pass refuses to overwrite <out stem>.sums.pth')
    ap.add_argument('--ann', default='dso_infos_train.pkl',
                    help='info pkl of the split to fit on (with --from-sums, '
                    'only the --check frames come from it)')
    ap.add_argument('--check', type=int, default=0, metavar='N',
                    help='compare float32 GPU and float64 CPU log-densities '
                    'on N frames before saving')
    ap.add_argument('--from-sums', metavar='PATH',
                    help='finalise the running sums an earlier pass saved '
                    '(<out stem>.sums.pth) instead of running the pass; '
                    'they must come from the same features and weights')
    ap.add_argument('--limit', type=int, default=0,
                    help='stop the pass after this many frames (0 = all; '
                    'smoke runs)')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    fit(args.config, args.checkpoint, args.out, args.features, ann=args.ann,
        check=args.check, limit=args.limit, workers=args.workers,
        device=args.device, from_sums=args.from_sums)


if __name__ == '__main__':
    main()
