#!/usr/bin/env python
"""
eval_ood_from_logits.py — uniform OOD-score calculator from saved per-point logits.

ONE script for many panoptic models / any dataset. The ONLY required input is a folder
of saved logits; each model just needs to dump logits the same way:

    <logits-dir>/<seq>/<scan>.npy   with shape [N, C]  (raw class logits, float16/32)

MODES
-----
1) SCORE (default): read logits -> compute every OOD score -> write score folders
       <out-dir>/prediction_<method>/<seq>/<scan>.npy   (one score per point)
   Needs ONLY the logits. `--num-classes` is auto-inferred from the logit shape.
   Group/gn scores need a class hierarchy (from --dataset preset or --hierarchy);
   without one, only the flat scores (msp/maxlogit/energy/entropy) are produced.

2) EVAL (add --data-root): also derive OOD/ID ground truth from the labels and report
   point-level AUROC / FPR95 / AP (2.5-50 m window, >= min-ood OOD / scan).

Scores (higher = more OOD):
flat {msp, maxlogit, energy, entropy, odin},
group {group_msp, group_maxlogit, group_energy, group_entropy, group_odin},
group normalization: {gn_msp, gn_maxlogit, gn_energy, gn_entropy, gn_odin}.
ODIN uses T=--odin-temperature (default 1000) with eps=0, i.e. temperature-scaled MSP.

Examples
--------
# just compute scores for a model (no labels needed):
python eval_ood_from_logits.py --dataset dso \
    --logits-dir /path/modelA/prediction --out-dir /path/modelA/ood

# compute + evaluate on DSO:
python eval_ood_from_logits.py --dataset dso \
    --logits-dir /path/modelA/prediction --out-dir /path/modelA/ood \
    --data-root /mnt/hdd/duynn/datasets/stu/sequences
    
python eval_ood_from_logits.py --dataset dso24 \
  --logits-dir <p3former_logits> --out-dir <ood> \
  --data-root /mnt/hdd/duynn/datasets/stu/sequences \
  --sequences 313 314 315        # (new setup; repeat for 303/311/312 and all-6)
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

PRESETS = {
    "dso24": dict(
        hierarchy=[
           [0, 1, 2, 3, 4],            # vehicle: car,bicycle,motorcycle,truck,bus
           [5, 6],                     # human: person,rider
           [9, 10, 11, 19],            # ground: paved-road,unpaved-road,sidewalk,drain
           [12, 13, 14, 15, 16, 17],   # construction: building,window,perim-barrier,other-barrier,overhead-bridge,gate
           [20, 21, 22],               # nature: terrain,trunks,vegetation
           [7, 8, 18, 23]              # object: traffic-sign,traffic-cone,pole-like-object,obscurant
        ],             
        id_raw_ids=[i for i in range(24)],
        ood_raw_ids=[24],
        feat_dim=5,
    ),
    "stu_kitti": dict(
        hierarchy=[
           [0, 1, 2, 3, 4],   # vehicle: car,bicycle,motorcycle,truck,bus/other-vehicle
           [5, 6, 7],         # human: person,bicyclist,motorcyclist
           [8, 9, 10, 11],    # ground: road,parking,sidewalk,other-ground
           [12, 13],          # construction: building,fence
           [14, 15, 16],      # nature: vegetation,trunk,terrain
           [17, 18]           # object: pole,traffic-sign
        ],
        id_raw_ids=[i for i in range(19)],
        ood_raw_ids=[19],
        feat_dim=4,
    ),
}

# ------------------------------ score maths ----------------------------------
def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / np.clip(e.sum(1, keepdims=True), 1e-30, None)

def _lse(z, axis=1):
    m = z.max(axis, keepdims=True)
    return m.squeeze(axis) + np.log(np.clip(np.exp(z - m).sum(axis), 1e-30, None))

def scores_from_logits(logits, hier, C, odin_T=1000.0):
    """OOD scores (higher = more OOD) from raw logits [N,C]. hier=None -> flat only."""
    z = logits.astype(np.float64); p = _softmax(z); pc = np.clip(p, 1e-12, None)
    pT = _softmax(z / odin_T)
    out = {
        "msp":      1 - p.max(1),
        "maxlogit": -z.max(1),
        "energy":   -_lse(z, 1),
        "entropy":  -(pc * np.log(pc)).sum(1),
        "odin":     1 - pT.max(1),
    }
    if hier:

        gp   = np.stack([p[:, g].sum(1)   for g in hier], 1)   # group prob-sum  (msp/entropy/gn)
        gpT  = np.stack([pT[:, g].sum(1)  for g in hier], 1)   # group prob-sum at T            (odin)
        gmax = np.stack([z[:, g].max(1)   for g in hier], 1)   # MAX logit within group        (maxlogit)
        glse = np.stack([_lse(z[:, g], 1) for g in hier], 1)   # LOGSUMEXP within group         (energy)
        gpc = np.clip(gp, 1e-12, None)
        K = np.array([len(g) for g in hier], dtype=np.float64)

        # Group aggregation = <within-group op> then MAX over groups (consistent with group_msp).
        # maxlogit uses MAX within group, energy uses LOGSUMEXP within group. Neither sums raw logits.
        # NB: group_maxlogit (max-within, max-over) is mathematically identical to flat maxlogit.
        logK = np.log(K); prior = K / C
        Q = np.clip(gp - prior, 0, None)
        QT = np.clip(gpT - prior, 0, None)
        out.update({
            "group_msp":      1 - gp.max(1),
            "group_maxlogit": -gmax.max(1),
            "group_energy":   -glse.max(1),
            "group_entropy":  -(gpc * np.log(gpc)).sum(1),
            "group_odin":     1 - gpT.max(1),
            "gn_msp":         1 - Q.max(1),
            "gn_maxlogit":    -(gmax - logK).max(1),
            "gn_energy":      -(glse - logK).max(1),
            "gn_entropy":     -((Q + 1e-12) * np.log(Q + 1e-12)).sum(1),
            "gn_odin":        1 - QT.max(1),
        })
    return out

# ------------------------------- driver --------------------------------------
def fpr95(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0: return float("nan")
    return float((neg >= np.quantile(pos, 0.05)).mean())

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logits-dir", required=True, help="<seq>/<scan>.npy raw logits [N,C]")
    ap.add_argument("--out-dir", default=None, help="write prediction_<method>/<seq>/<scan>.npy here")
    ap.add_argument("--dataset", choices=list(PRESETS), default=None)
    ap.add_argument("--hierarchy", type=str, help='JSON index-lists, e.g. "[[0,1],[2,3]]"')
    ap.add_argument("--num-classes", type=int, help="default: inferred from logit shape")
    ap.add_argument("--odin-temperature", type=float, default=1000.0,
                    help="ODIN softmax temperature (eps=0, no input perturbation)")
    ap.add_argument("--sequences", nargs="*", default=None)
    # eval-only (optional)
    ap.add_argument("--data-root", default=None, help="enable metrics: <seq>/labels,<seq>/velodyne")
    ap.add_argument("--id-raw-ids", nargs="*", type=int)
    ap.add_argument("--ood-raw-ids", nargs="*", type=int)
    ap.add_argument("--feat-dim", type=int)
    ap.add_argument("--min-dist", type=float, default=2.5)
    ap.add_argument("--max-dist", type=float, default=50.0)
    ap.add_argument("--min-ood", type=int, default=5)
    ap.add_argument("--id-cap", type=int, default=20000, help="ID pts/scan for metrics (0=all)")
    ap.add_argument("--output", default=None, help="metrics JSON")
    args = ap.parse_args()

    if not args.out_dir and not args.data_root:
        sys.exit("nothing to do: give --out-dir (save scores) and/or --data-root (evaluate)")

    cfg = dict(PRESETS.get(args.dataset, {}))
    if args.hierarchy:   cfg["hierarchy"] = json.loads(args.hierarchy)
    if args.id_raw_ids is not None:  cfg["id_raw_ids"] = args.id_raw_ids
    if args.ood_raw_ids is not None: cfg["ood_raw_ids"] = args.ood_raw_ids
    if args.feat_dim is not None:    cfg["feat_dim"] = args.feat_dim
    HIER = cfg.get("hierarchy")
    if HIER is None:
        print("[warn] no hierarchy (no --dataset/--hierarchy) -> flat scores only", file=sys.stderr)

    ldir = args.logits_dir
    seqs = args.sequences or sorted(d for d in os.listdir(ldir)
                                    if d.isdigit() and os.path.isdir(f"{ldir}/{d}"))
    do_eval = args.data_root is not None
    if do_eval:
        for k in ("id_raw_ids", "ood_raw_ids", "feat_dim"):
            if k not in cfg:
                sys.exit(f"--data-root eval needs '{k}' (use --dataset or --{k.replace('_','-')})")
        ID, OOD, FD = set(cfg["id_raw_ids"]), set(cfg["ood_raw_ids"]), cfg["feat_dim"]
        rng = np.random.default_rng(0); buf, gts = None, []; n_scan = 0

    C = args.num_classes
    n_written = 0
    for seq in seqs:
        for f in sorted(glob.glob(f"{ldir}/{seq}/*.npy")):
            scan = os.path.basename(f)[:-4]
            logits = np.load(f)
            if logits.ndim != 2:
                continue
            if C is None:
                C = logits.shape[1]
                print(f"[info] inferred num_classes = {C} from logits")
            if logits.shape[1] != C:
                continue
            sc = scores_from_logits(logits, HIER, C, args.odin_temperature)
            if args.out_dir:
                for m, v in sc.items():
                    od = f"{args.out_dir}/prediction_{m}/{seq}"; os.makedirs(od, exist_ok=True)
                    np.save(f"{od}/{scan}.npy", v.astype(np.float32))
                n_written += 1
            if do_eval:
                lp = f"{args.data_root}/{seq}/labels/{scan}.label"
                bp = f"{args.data_root}/{seq}/velodyne/{scan}.bin"
                if not (os.path.exists(lp) and os.path.exists(bp)): continue
                raw = (np.fromfile(lp, dtype=np.uint32) & 0xFFFF)
                pts = np.fromfile(bp, dtype=np.float32).reshape(-1, FD)[:, :3]
                if not (len(raw) == len(pts) == len(logits)): continue
                d = np.linalg.norm(pts, axis=1)
                y = np.full(len(raw), -1, np.int8)
                y[np.isin(raw, list(ID))] = 0; y[np.isin(raw, list(OOD))] = 1
                keep = (d >= args.min_dist) & (d <= args.max_dist) & (y != -1)
                if int((y[keep] == 1).sum()) < args.min_ood: continue
                yk = y[keep]
                if buf is None: buf = {m: [] for m in sc}
                oi = np.where(yk == 1)[0]; ii = np.where(yk == 0)[0]
                if args.id_cap and len(ii) > args.id_cap:
                    ii = rng.choice(ii, args.id_cap, replace=False)
                sel = np.concatenate([oi, ii])
                for m in sc: buf[m].append(sc[m][keep][sel])
                gts.append(yk[sel]); n_scan += 1

    if args.out_dir:
        print(f"[score] wrote {n_written} scans x {len(sc)} methods -> {args.out_dir}/prediction_*")
    if do_eval:
        if not gts: sys.exit("eval: no scans with both ID and OOD points")
        from sklearn.metrics import roc_auc_score, average_precision_score
        y = np.concatenate(gts)
        print(f"\n[eval] scans={n_scan}  ID={(y==0).sum():,}  OOD={(y==1).sum():,}"
              f"{'  (ID capped /scan)' if args.id_cap else ''}")
        print(f"{'method':16}{'AUROC':>9}{'FPR95':>9}{'AP':>9}")
        res = {}
        for m in buf:
            s = np.concatenate(buf[m]).astype(np.float64)
            a, fp, pr = roc_auc_score(y, s), fpr95(s, y), average_precision_score(y, s)
            res[m] = dict(AUROC=a, FPR95=fp, AP=pr)
            print(f"{m:16}{a:>9.4f}{fp:>9.4f}{pr:>9.4f}")
        if args.output:
            json.dump({"n_scans": n_scan, "metrics": res}, open(args.output, "w"), indent=2)
            print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
