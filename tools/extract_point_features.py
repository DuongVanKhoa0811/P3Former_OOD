#!/usr/bin/env python
"""Sample per-point penultimate features of P3Former for offline analysis.

The auxiliary semantic classifier of _P3FormerHead is a bias-free linear
layer on ``pe_features``: the per-voxel 256-d backbone features after
``pe_conv`` plus the positional embedding ``mpe`` (cartesian + polar). Its
logits, sem_preds = pe_features @ sem_queries^T, are the ones every OOD
score and the logits dumps use. This tool runs the model on one split,
wraps ``decode_head.init_inputs`` on the model instance (the model code is
not changed) to keep ``pe_features``, ``mpe`` and ``sem_preds`` of each
frame, maps them to points with ``point2voxel_map`` and writes a
stratified sample of every frame: up to --per-class ID points of each
mapped class and up to --ood-per-frame OOD points (raw 17 Stop / 28
Others, as _OODPointMetric), each weighted by stratum size / points drawn,
so that weighted means over the samples estimate population means.

One ``f<frame>.npz`` per frame in --out-dir -- feat (float16 [n, 256]), pos
(float16 [n, 256], the mpe part of feat), logits (float16 [n, 24]), label
(int16 mapped train id, 24 for OOD), raw (int16 raw semantic id), ood
(bool), weight (float32), index (int32 point index in the frame),
lidar_path, frame -- plus meta.json. Checks: on every frame feat @ W^T
(W = sem_queries.weight) must reproduce the captured logits (float64
reference; tolerance 1e-2 * max(1, max |z|), room for the TF32 matmuls of
test.py's numerics, which also wrote the logits dumps); with --check-dump
the sampled logits, labels and OOD flags must equal those of a logits dump
at the same points (frames matched by lidar_path).

Run from the repo root (one GPU, ~25 GB; Cetran ~10 min, test ~25 min):
    CUDA_VISIBLE_DEVICES=0 python tools/extract_point_features.py \\
        configs/p3former/p3former_2xb1_3x_dso_ood.py \\
        work_dirs/p3former_2xb1_3x_dso/epoch_36.pth \\
        --ann dso_infos_cetran.pkl \\
        --out-dir work_dirs/p3former_2xb1_3x_dso_ood_dump/features_cetran \\
        --check-dump work_dirs/p3former_2xb1_3x_dso_ood_dump/logits
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

NUM_CLASSES = 24  # DSO train classes; logit channel 24 is the ignore slot
IGNORE_INDEX = 24
OOD_RAW_IDS = (17, 28)  # Stop, Others: never trained
SEG_OFFSET = 2**16
LOGIT_TOL = 1e-2  # |feat @ W^T - logits| <= LOGIT_TOL * max(1, max |z|)
DUMP_TOL = 0.05  # sampled logits vs a logits dump, both float16


class FeatureCapture:
    """Keep the per-voxel tensors of the last ``init_inputs`` call of a
    decode head: ``feat`` (pe_features), ``pos`` (mpe) and ``logits``
    (sem_preds), each a list over the batch. Use as a context manager: the
    wrapper lives on the instance and is removed on exit."""

    def __init__(self, head):
        self.head = head
        self.feat = self.pos = self.logits = None

    def __enter__(self):
        original = self.head.init_inputs

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            _, self.feat, self.pos, self.logits = out
            return out

        self.head.init_inputs = wrapped
        return self

    def __exit__(self, *exc):
        del self.head.init_inputs  # the class method is visible again
        return False


def sample_frame(label, ood, per_class, ood_per_frame, rng):
    """Stratified sample of one frame's valid points: up to ``per_class``
    ID points of every mapped class, then up to ``ood_per_frame`` OOD
    points, uniformly without replacement within each stratum. Returns
    (index int64 [n], weight float32 [n]), weight = stratum size / drawn."""
    strata = [np.flatnonzero((label == c) & ~ood) for c in range(NUM_CLASSES)]
    strata.append(np.flatnonzero(ood))
    caps = [per_class] * NUM_CLASSES + [ood_per_frame]
    index, weight = [], []
    for members, cap in zip(strata, caps):
        n = min(cap, len(members))
        if n == 0:
            continue
        index.append(np.sort(rng.choice(members, n, replace=False)))
        weight.append(np.full(n, len(members) / n, np.float32))
    if not index:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(index).astype(np.int64), np.concatenate(weight)


def frame_ground_truth(data_sample):
    """(mapped label, raw semantic id, ood) per point, as _OODPointMetric."""
    ann = data_sample.eval_ann_info
    raw = np.asarray(ann['pts_instance_mask']) % SEG_OFFSET
    label = np.asarray(ann['pts_semantic_mask'])
    return label, raw, np.isin(raw, OOD_RAW_IDS)


def classifier_weight(head):
    """The semantic classifier's rows of the used classes, [NUM_CLASSES, C]."""
    w = head.sem_queries.weight.detach()
    return w.reshape(w.shape[0], -1)[:NUM_CLASSES]


def check_logits(feat, logits, weight):
    """Raise unless feat @ weight^T (float64) reproduces the captured
    logits; returns the largest |difference|."""
    if not len(feat):
        return 0.0
    ref = feat.double() @ weight.double().t()
    diff = float((ref - logits.double()).abs().max())
    scale = max(1.0, float(logits.abs().max()))
    if diff > LOGIT_TOL * scale:
        raise RuntimeError(
            'the captured features do not reproduce the logits (max |diff| '
            f'{diff:.4g} at scale {scale:.4g}): does init_inputs still return '
            '(queries, pe_features, mpe, sem_preds)?')
    return diff


def frame_record(capture, data_sample, weight, per_class, ood_per_frame, rng,
                 frame):
    """The sampled points of one frame (batch size 1) as a dict of arrays,
    and the logit-reconstruction error."""
    label, raw, ood = frame_ground_truth(data_sample)
    p2v = data_sample.gt_pts_seg.point2voxel_map.long()
    if p2v.shape[0] != label.shape[0]:
        raise ValueError(f'point2voxel_map has {p2v.shape[0]} points, the '
                         f'ground truth {label.shape[0]}')
    index, w = sample_frame(label, ood, per_class, ood_per_frame, rng)
    voxels = p2v[torch.from_numpy(index).to(p2v.device)]
    # detach: the captured tensors may still carry autograd history (e.g. a
    # live classifier weight) even when the caller forgot torch.no_grad()
    feat = capture.feat[0][voxels].detach().float()
    pos = capture.pos[0][voxels].detach().float()
    logits = capture.logits[0][voxels, :NUM_CLASSES].detach().float()
    diff = check_logits(feat.cpu(), logits.cpu(), weight.cpu())
    record = dict(
        feat=feat.half().cpu().numpy(), pos=pos.half().cpu().numpy(),
        logits=logits.half().cpu().numpy(),
        label=label[index].astype(np.int16), raw=raw[index].astype(np.int16),
        ood=ood[index], weight=w, index=index.astype(np.int32),
        lidar_path=str(data_sample.metainfo.get('lidar_path', '')),
        frame=frame)
    return record, diff


def index_dump(dump_dir):
    """lidar_path -> npz file of a logits dump (_OODLogitsDumpMetric)."""
    files = sorted(glob.glob(osp.join(dump_dir, '*.npz')))
    if not files:
        raise FileNotFoundError(f'no .npz dumps in {dump_dir}')
    out = {}
    for path in files:
        with np.load(path) as data:
            out[str(data['lidar_path'])] = path
    return out


def dump_file_for(record, dump, dump_dir):
    """The dump file for ``record['lidar_path']``. Raises KeyError naming
    the frame, the lidar_path and ``dump_dir`` when the path is missing
    from ``dump`` (an ``index_dump`` result)."""
    lidar_path = record['lidar_path']
    if lidar_path not in dump:
        raise KeyError(f'frame {record["frame"]} ({lidar_path}) is not in '
                       f'the dump {dump_dir}')
    return dump[lidar_path]


def compare_with_dump(record, dump_file):
    """Raise unless the record's logits, labels and OOD flags equal the
    dump's at the same points; returns the largest logit |difference|."""
    idx = record['index']
    with np.load(dump_file) as data:
        logits = data['logits'][idx].astype(np.float32)
        mapped, ood = data['mapped'][idx], data['ood'][idx]
    diff = (float(np.abs(logits - record['logits'].astype(np.float32)).max())
            if len(idx) else 0.0)
    same_label = np.array_equal(mapped, record['label'])
    same_ood = np.array_equal(ood, record['ood'])
    if diff > DUMP_TOL or not same_label or not same_ood:
        raise RuntimeError(
            f'frame {record["frame"]} ({record["lidar_path"]}) differs from '
            f'{dump_file}: max logit |diff| {diff:.4g}, labels equal '
            f'{same_label}, OOD flags equal {same_ood}')
    return diff


def build(config, checkpoint, ann=None, workers=4, device='cuda:0'):
    """(model, dataloader) as test.py builds them, with batch size 1."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint

    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(config)  # also imports cfg.custom_imports
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    if ann:
        cfg.test_dataloader.dataset.dataset.ann_file = ann
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = workers
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device).eval()
    return model, Runner.build_dataloader(cfg.test_dataloader)


def extract(config, checkpoint, out_dir, ann=None, per_class=64,
            ood_per_frame=512, seed=0, limit=0, check_dump=None, workers=4,
            device='cuda:0'):
    """Write one sample file per frame and meta.json; returns the meta."""
    os.makedirs(out_dir, exist_ok=True)
    if os.listdir(out_dir):
        raise FileExistsError(f'{out_dir} is not empty: delete its files or '
                              'choose another --out-dir')
    dump = index_dump(check_dump) if check_dump else None
    model, loader = build(config, checkpoint, ann, workers, device)
    weight = classifier_weight(model.decode_head)
    stats = dict(frames=0, points=0, ood_samples=0,
                 samples_per_class=[0] * NUM_CLASSES, max_logit_diff=0.0,
                 max_dump_diff=None if dump is None else 0.0)
    t0 = time.time()
    with FeatureCapture(model.decode_head) as capture, torch.no_grad():
        for frame, data in enumerate(loader):
            if limit and frame >= limit:
                break
            (sample, ) = model.test_step(data)
            record, diff = frame_record(capture, sample, weight, per_class,
                                        ood_per_frame,
                                        np.random.default_rng([seed, frame]),
                                        frame)
            stats['max_logit_diff'] = max(stats['max_logit_diff'], diff)
            if dump is not None:
                dump_file = dump_file_for(record, dump, check_dump)
                stats['max_dump_diff'] = max(
                    stats['max_dump_diff'],
                    compare_with_dump(record, dump_file))
            np.savez(osp.join(out_dir, f'f{frame:06d}.npz'), **record)
            per_class_count = np.bincount(record['label'][~record['ood']],
                                          minlength=NUM_CLASSES)[:NUM_CLASSES]
            stats['samples_per_class'] = [
                a + int(b) for a, b in zip(stats['samples_per_class'],
                                           per_class_count)]
            stats['ood_samples'] += int(record['ood'].sum())
            stats['points'] += int(len(record['index']))
            stats['frames'] += 1
            if stats['frames'] % 100 == 0:
                print(f'{stats["frames"]} frames ({time.time() - t0:.0f} s)',
                      flush=True)
    meta = dict(config=config, checkpoint=checkpoint, ann=ann,
                per_class=per_class, ood_per_frame=ood_per_frame, seed=seed,
                check_dump=check_dump, seconds=round(time.time() - t0),
                **stats)
    with open(osp.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='test config (as test.py)')
    ap.add_argument('checkpoint', help='checkpoint file')
    ap.add_argument('--ann', default=None,
                    help="info pkl of the split (default: the config's)")
    ap.add_argument('--out-dir', required=True,
                    help='empty directory for the per-frame samples')
    ap.add_argument('--per-class', type=int, default=64,
                    help='ID points sampled per mapped class and frame')
    ap.add_argument('--ood-per-frame', type=int, default=512,
                    help='OOD points sampled per frame')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after this many frames (0 = all)')
    ap.add_argument('--check-dump', default=None, metavar='DIR',
                    help='logits dump of the same split to compare with')
    ap.add_argument('--workers', type=int, default=4,
                    help='dataloader worker processes')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    meta = extract(args.config, args.checkpoint, args.out_dir, ann=args.ann,
                   per_class=args.per_class, ood_per_frame=args.ood_per_frame,
                   seed=args.seed, limit=args.limit,
                   check_dump=args.check_dump, workers=args.workers,
                   device=args.device)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
