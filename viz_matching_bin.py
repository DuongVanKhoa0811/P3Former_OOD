"""Export a frame's HD-OOD3D matching stages as point-cloud .bin files.

Writes seven .bin files for one frame, each = the LiDAR sweep plus some
combination of ground-truth boxes and predictions, so the matching funnel can
be stepped through in any point-cloud viewer:

  <fid>_1_all.bin        GT + every prediction in the dump
  <fid>_2_scored.bin     GT + predictions with score > delta
  <fid>_3_matched.bin    GT + predictions kept by greedy_match (Algorithm 1)
  <fid>_4_gt_only.bin    GT alone, no predictions
  <fid>_5_pred_only.bin  every prediction alone, no GT
  <fid>_6_matched_ood.bin   matched predictions alone, split by their TRUE OOD label
  <fid>_7_matched_pred.bin  matched predictions alone, split by the MLP's PREDICTION

.bin holds points, not boxes, so each box is rendered as points traced along
its 12 edges. Output stays (N, 4) float32 like the source sweep; the 4th
channel is a LABEL (not intensity) so a viewer can colour by it:

  0 = raw LiDAR point
  1 = ground-truth box, known class
  2 = ground-truth box, unknown class
  3 = prediction box
  4 = matched prediction, KNOWN     (file 6: true label | file 7: MLP prediction)
  5 = matched prediction, UNKNOWN   (file 6: true label | file 7: MLP prediction)

Files 6 and 7 hold the same boxes and differ only in what 4/5 mean, so diffing
them shows the MLP's errors directly:
  file 6 -- the label the matcher assigns, i.e. the class of the ground-truth box
            the prediction matched (greedy_match's third return). This is exactly
            the y used to train/evaluate the MLP.
  file 7 -- the MLP's own call: p_OOD >= threshold => unknown, where the threshold
            is calibrated at the FPR-95 operating point (TPR=0.95 on the val Open
            subset, the same protocol eval_ood.py reports). At that operating point
            ~45% of KNOWN boxes are flagged unknown by design — it is a
            high-recall point, so expect file 7 to over-flag.

Neither file carries GT boxes, but the raw sweep is identical across all seven,
so loading _4_gt_only alongside _6/_7 overlays them exactly.

NOTE on "all predictions": the dumps were written at the detector's own
POST_PROCESSING.SCORE_THRESH (0.1), so file 1 is every box the detector
emitted, not literally every raw box. Boxes below 0.1 would need the detector
re-run with SCORE_THRESH=0.

Usage (run from tools/):
  python ood/viz_matching_bin.py --frame_id 1000000 --split val
"""
import argparse
import glob
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.ood.feature_extract import build_frame_gt
from pcdet.ood.matching import greedy_match
from pcdet.ood.metrics import threshold_at_tpr
from pcdet.ood.mlp import OodMlp, p_ood_from_logits
from pcdet.utils.box_utils import boxes_to_corners_3d

# corner ids from box_utils.boxes_to_corners_3d: 0-3 bottom face, 4-7 top face
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

LBL_RAW, LBL_GT_KNOWN, LBL_GT_UNKNOWN, LBL_PRED = 0.0, 1.0, 2.0, 3.0
LBL_MATCH_KNOWN, LBL_MATCH_UNKNOWN = 4.0, 5.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--frame_id', required=True, help='e.g. 1000000')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--dump_dir', default=None,
                   help='default ../data/argo2/ood_feats/<split>')
    p.add_argument('--info', default=None,
                   help='default ../data/argo2/argo2_infos_<split>.pkl')
    p.add_argument('--lidar_dir', default='../data/argo2/training/velodyne')
    p.add_argument('--ood_cfg', default='cfgs/argo2_models/hd_ood3d_mlp.yaml')
    p.add_argument('--out_dir', default='../docs/reports/match_viz')
    p.add_argument('--score_thresh', type=float, default=None,
                   help='override delta (default: split-appropriate value from OOD cfg)')
    p.add_argument('--dist_thresh', type=float, default=None)
    p.add_argument('--pts_per_edge', type=int, default=60,
                   help='points traced along each box edge')
    p.add_argument('--pc_range', type=float, nargs=6,
                   default=[-72.0, -72.0, -3.0, 72.0, 72.0, 5.0])
    p.add_argument('--no_crop', action='store_true',
                   help='keep LiDAR points outside POINT_CLOUD_RANGE')
    # --- file 7: the MLP's own known/unknown decision ---
    p.add_argument('--mlp',
                   default='../output/argo2_models/centerpoint_15known/default/ood/mlp.pth',
                   help='trained OOD head; file 7 is skipped if missing')
    p.add_argument('--ood_thresh', type=float, default=None,
                   help='p_OOD decision threshold (default: calibrated at --tpr)')
    p.add_argument('--tpr', type=float, default=0.95,
                   help='operating point for the calibrated threshold (0.95 = FPR-95)')
    p.add_argument('--thresh_dump_dir', default='../data/argo2/ood_feats/val',
                   help='dump the threshold is calibrated on (the val protocol)')
    p.add_argument('--thresh_max_frames', type=int, default=4000,
                   help='frames sampled for calibration; 0 = all of them')
    p.add_argument('--thresh_cache',
                   default='../output/argo2_models/centerpoint_15known/default/ood/'
                           'ood_threshold.json')
    p.add_argument('--recompute_thresh', action='store_true')
    return p.parse_args()


def boxes_to_edge_points(boxes, label, pts_per_edge):
    """(N,7) boxes -> (M,4) points tracing all 12 edges, 4th col = label."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 7)
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), np.float32)
    corners = boxes_to_corners_3d(boxes)
    if hasattr(corners, 'numpy'):                 # tolerate a torch return
        corners = corners.numpy()
    corners = np.asarray(corners, dtype=np.float32)      # (N,8,3)

    t = np.linspace(0.0, 1.0, pts_per_edge, dtype=np.float32)[:, None]
    segs = []
    for c in corners:
        for a, b in EDGES:
            segs.append(c[a][None, :] * (1.0 - t) + c[b][None, :] * t)
    pts = np.concatenate(segs, axis=0).astype(np.float32)
    return np.concatenate([pts, np.full((len(pts), 1), label, np.float32)], axis=1)


def predict_p_ood(model, device, e, p, b):
    """p_OOD for every prediction row of a dump."""
    with torch.no_grad():
        t = [torch.from_numpy(np.asarray(a, dtype=np.float32)).to(device)
             for a in (e, p, b)]
        return p_ood_from_logits(model(*t)).cpu().numpy()


def compute_ood_threshold(model, device, dump_dir, score_thresh, dist_thresh,
                          tpr_target, max_frames):
    """p_OOD threshold at the FPR-95 operating point.

    Mirrors eval_ood.py (Open subset, same matching), so the threshold really is
    the one the reported FPR-95 corresponds to. A single frame cannot define it —
    it needs the val score distribution, hence the separate calibration pass.
    """
    frames = sorted(glob.glob(str(Path(dump_dir) / '*.npz')))
    if max_frames:
        random.Random(0).shuffle(frames)
        frames = frames[:max_frames]
    all_p, all_y = [], []
    for fp in frames:
        d = np.load(fp, allow_pickle=True)
        gtu = d['gt_is_unknown']
        if gtu.shape[0] == 0 or not gtu.any():
            continue                                    # Open subset only
        b, sc, gtb = d['b'], d['score'], d['gt_boxes']
        if b.shape[0] == 0:
            continue
        pi, _, y = greedy_match(b[:, :3], sc, gtb[:, :3], gtu.astype(np.int64),
                                score_thresh, dist_thresh)
        if pi.size == 0:
            continue
        all_p.append(predict_p_ood(model, device, d['e'], d['p'], b)[pi])
        all_y.append(y)
    assert all_p, f'no matched pairs in {dump_dir} to calibrate on'
    p_all, y_all = np.concatenate(all_p), np.concatenate(all_y)
    return (threshold_at_tpr(p_all, y_all, tpr_target),
            len(p_all), int((y_all == 1).sum()))


def resolve_threshold(args, model, device, score_thresh, dist_thresh):
    """Return (threshold, provenance). Cached so calibration runs once."""
    if args.ood_thresh is not None:
        return float(args.ood_thresh), 'given via --ood_thresh'
    key = {'tpr': args.tpr, 'score_thresh': score_thresh,
           'dist_thresh': dist_thresh, 'max_frames': args.thresh_max_frames,
           'mlp': str(Path(args.mlp).resolve()),
           'dump_dir': str(Path(args.thresh_dump_dir).resolve())}
    cache = Path(args.thresh_cache)
    if cache.exists() and not args.recompute_thresh:
        got = json.loads(cache.read_text())
        if all(got.get(k) == v for k, v in key.items()):
            return float(got['threshold']), f'cached in {cache.name}'
    thr, n, n_unk = compute_ood_threshold(
        model, device, args.thresh_dump_dir, score_thresh, dist_thresh,
        args.tpr, args.thresh_max_frames)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {**key, 'threshold': thr, 'n_pairs': n, 'n_unknown': n_unk}, indent=2))
    return thr, f'calibrated on {n} pairs ({n_unk} unknown)'


def main():
    args = parse_args()
    fid = str(args.frame_id)
    dump_dir = Path(args.dump_dir or f'../data/argo2/ood_feats/{args.split}')
    info_path = Path(args.info or f'../data/argo2/argo2_infos_{args.split}.pkl')

    ood = cfg_from_yaml_file(args.ood_cfg, type(cfg)()).OOD
    dist_thresh = args.dist_thresh if args.dist_thresh is not None \
        else float(ood.MATCH.DIST_THRESH)
    if args.score_thresh is not None:
        score_thresh = args.score_thresh
    else:
        score_thresh = float(ood.MATCH.SCORE_THRESH_TRAIN if args.split == 'train'
                             else ood.MATCH.SCORE_THRESH_EVAL)

    # --- raw sweep ---
    lidar_file = Path(args.lidar_dir) / f'{fid}.bin'
    assert lidar_file.exists(), f'no sweep at {lidar_file}'
    pts = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    if not args.no_crop:
        r = args.pc_range
        m = ((pts[:, 0] >= r[0]) & (pts[:, 0] <= r[3]) &
             (pts[:, 1] >= r[1]) & (pts[:, 1] <= r[4]) &
             (pts[:, 2] >= r[2]) & (pts[:, 2] <= r[5]))
        pts = pts[m]
    pts = pts.copy()
    pts[:, 3] = LBL_RAW                      # 4th channel becomes the label

    # --- predictions from the dump ---
    npz = dump_dir / f'{fid}.npz'
    assert npz.exists(), f'no dump at {npz}'
    d = np.load(npz, allow_pickle=True)
    pred_boxes, pred_scores = d['b'], d['score']

    # --- ground truth straight from the info (same filter the matcher sees) ---
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)
    info = None
    for it in infos:
        if it['point_cloud']['velodyne_path'].split('/')[-1].rstrip('.bin') == fid:
            info = it
            break
    assert info is not None, f'frame {fid} not found in {info_path}'
    gt_boxes, gt_labels = build_frame_gt(info, args.pc_range)
    assert np.allclose(gt_boxes, d['gt_boxes']), 'GT disagrees with the dump'

    gt_pts = np.concatenate([
        boxes_to_edge_points(gt_boxes[gt_labels == 0], LBL_GT_KNOWN, args.pts_per_edge),
        boxes_to_edge_points(gt_boxes[gt_labels == 1], LBL_GT_UNKNOWN, args.pts_per_edge),
    ], axis=0)

    # --- the three matching stages ---
    keep_scored = pred_scores > score_thresh
    matched_idx, _, matched_y = greedy_match(
        pred_boxes[:, :3], pred_scores, gt_boxes[:, :3], gt_labels,
        score_thresh, dist_thresh)

    matched_boxes = pred_boxes[matched_idx]
    # (name, [(boxes, label), ...], include GT boxes)
    stages = [
        ('1_all',         [(pred_boxes, LBL_PRED)],              True),
        ('2_scored',      [(pred_boxes[keep_scored], LBL_PRED)], True),
        ('3_matched',     [(matched_boxes, LBL_PRED)],           True),
        ('4_gt_only',     [],                                    True),
        ('5_pred_only',   [(pred_boxes, LBL_PRED)],              False),
        ('6_matched_ood', [(matched_boxes[matched_y == 0], LBL_MATCH_KNOWN),
                           (matched_boxes[matched_y == 1], LBL_MATCH_UNKNOWN)], False),
    ]

    print(f'frame {fid} ({args.split})  delta={score_thresh}  d={dist_thresh} m')
    print(f'  GT: {len(gt_boxes)} boxes '
          f'({int((gt_labels == 0).sum())} known, {int((gt_labels == 1).sum())} unknown)')
    print(f'  matched: {len(matched_idx)} preds '
          f'({int((matched_y == 0).sum())} known, {int((matched_y == 1).sum())} unknown)')

    # --- file 7: the same matched boxes, coloured by the MLP's own decision ---
    mlp_path = Path(args.mlp)
    if not mlp_path.exists():
        print(f'  [file 7 skipped] no MLP at {mlp_path}')
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ck = torch.load(str(mlp_path), weights_only=False)   # torch>=2.6 needs this
        model = OodMlp(proj=tuple(ck['proj']), hidden=tuple(ck['hidden'])).to(device)
        model.load_state_dict(ck['state_dict'])
        model.eval()
        # Calibrate on the val protocol (delta_eval) regardless of --split: FPR-95
        # is defined by that protocol.
        thr, how = resolve_threshold(
            args, model, device, float(ood.MATCH.SCORE_THRESH_EVAL), dist_thresh)
        p_ood = predict_p_ood(model, device, d['e'], d['p'], pred_boxes)[matched_idx]
        as_unknown = p_ood >= thr
        stages.append(('7_matched_pred',
                       [(matched_boxes[~as_unknown], LBL_MATCH_KNOWN),
                        (matched_boxes[as_unknown], LBL_MATCH_UNKNOWN)], False))
        print(f'  MLP: p_OOD >= {thr:.4f} => unknown '
              f'(TPR={args.tpr} operating point, {how})')
        print(f'  predicted: {int((~as_unknown).sum())} known, '
              f'{int(as_unknown.sum())} unknown')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, groups, with_gt in stages:
        parts = [pts]
        if with_gt:
            parts.append(gt_pts)
        for boxes, label in groups:
            parts.append(boxes_to_edge_points(boxes, label, args.pts_per_edge))
        cloud = np.concatenate(parts, axis=0).astype(np.float32)
        out = out_dir / f'{fid}_{name}.bin'
        cloud.tofile(str(out))
        n_pred = sum(len(b) for b, _ in groups)
        print(f'  {name:<14s} GT={"y" if with_gt else "n"}  {n_pred:4d} pred boxes '
              f'-> {out}  ({len(cloud)} pts)')
    print(f'\nlabel channel: {LBL_RAW:.0f}=raw  {LBL_GT_KNOWN:.0f}=GT known  '
          f'{LBL_GT_UNKNOWN:.0f}=GT unknown  {LBL_PRED:.0f}=prediction  '
          f'{LBL_MATCH_KNOWN:.0f}=matched known  {LBL_MATCH_UNKNOWN:.0f}=matched unknown')


if __name__ == '__main__':
    main()
