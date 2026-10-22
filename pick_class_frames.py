#!/usr/bin/env python3
"""For every class in class_stats.csv, list N .ply frames that contain it.

Reads the artefacts `analyze_seg_labels.py` already wrote -- the class list from
`class_stats.csv` and the per-frame histogram from `per_frame_counts.npz` -- so
nothing is re-scanned; the whole thing runs in well under a second.

Picking the top-N frames by raw point count alone is a trap: the LiDAR runs at
10 Hz, so the best five frames for a class are usually five consecutive sweeps
of the same object from the same sequence. The default `best` strategy therefore
caps how many frames may come from any one sequence, giving five genuinely
different scenes to look at.

Examples
--------
    # five frames per class, one per sequence, written to class_frames.csv
    python pick_class_frames.py

    # just one class, more frames, allow several from the same sequence
    python pick_class_frames.py --class-id 9 --top 10 --per-seq 3

    # the Cetran run instead of the original 12 sequences
    python pick_class_frames.py --analysis-dir seg_label_analysis_cetran

    # frames spanning the whole range, not just the densest ones
    python pick_class_frames.py --strategy spread
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def slug(text):
    """Sequence name -> filename-safe token: '(2024-06-10) Ubin Route 1' -> '2024-06-10-Ubin-Route-1'."""
    return "-".join(text.replace("(", " ").replace(")", " ").split())


def gather(rows, dest_root, how, overwrite):
    """Copy/link every selected .ply into <dest_root>/<class>/<rank>_<seq>_<frame>.ply.

    Files are grouped per class so a whole class can be flipped through in one
    directory, and prefixed with their rank so the densest frame sorts first.
    Returns (n_transferred, n_skipped, bytes_written).
    """
    done = skipped = written = 0
    for r in rows:
        cdir = dest_root / ("%03d_%s" % (r["class_id"], r["name"]))
        cdir.mkdir(parents=True, exist_ok=True)
        dst = cdir / ("%d_%s_%s.ply" % (r["rank"], slug(r["sequence"]), r["frame"]))
        r["copied_to"] = str(dst)

        src = Path(r["path"])
        if not src.exists():
            r["copied_to"] = ""
            continue
        if dst.exists() or dst.is_symlink():
            if not overwrite:
                skipped += 1
                continue
            dst.unlink()

        if how == "copy":
            shutil.copy2(src, dst)
            written += dst.stat().st_size
        elif how == "hardlink":
            os.link(src, dst)          # same filesystem only, costs no extra space
        else:
            dst.symlink_to(src.resolve())
        done += 1
    return done, skipped, written


def render_table(headers, rows, aligns=None):
    aligns = aligns or [">"] * len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["  ".join("%-*s" % (w, h) if a == "<" else "%*s" % (w, h)
                     for h, a, w in zip(headers, aligns, widths)),
           "  ".join("-" * w for w in widths)]
    for row in rows:
        out.append("  ".join("%-*s" % (w, c) if a == "<" else "%*s" % (w, c)
                             for c, a, w in zip(row, aligns, widths)))
    return "\n".join(out)


def load_classes(csv_path):
    """Class list in the order class_stats.csv presents them."""
    with open(csv_path, newline="") as fh:
        return [(int(r["id"]), r["name"], r.get("color", ""))
                for r in csv.DictReader(fh)]


def pick(counts_col, seq_index, top, strategy, per_seq, min_points, rng):
    """Return frame indices for one class, best first."""
    present = np.nonzero(counts_col >= min_points)[0]
    if present.size == 0:
        return []

    if strategy == "random":
        rng.shuffle(present)
        order = present
    else:  # 'best' and 'spread' both work off the density ranking
        order = present[np.argsort(-counts_col[present], kind="stable")]
        if strategy == "spread" and order.size > top:
            # evenly spaced through the ranking: densest, then progressively
            # more typical frames, down to the sparsest that still qualifies
            order = order[np.linspace(0, order.size - 1, top * 4).astype(int)]

    picked, used = [], Counter()
    for i in order:                       # first pass: honour the per-sequence cap
        s = int(seq_index[i])
        if used[s] >= per_seq:
            continue
        picked.append(int(i))
        used[s] += 1
        if len(picked) == top:
            return picked

    seen = set(picked)                    # second pass: too few sequences, relax it
    for i in order:
        if int(i) not in seen:
            picked.append(int(i))
            if len(picked) == top:
                break
    return picked


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis-dir", type=Path, default=Path("seg_label_analysis"),
                    help="directory holding class_stats.csv and per_frame_counts.npz")
    ap.add_argument("--root", type=Path, default=None,
                    help="Annotation_Final root (default: whatever summary.json recorded)")
    ap.add_argument("--top", type=int, default=5, help="frames per class")
    ap.add_argument("--per-seq", type=int, default=1,
                    help="max frames from any one sequence, before relaxing")
    ap.add_argument("--strategy", choices=("best", "spread", "random"), default="best",
                    help="best: densest frames; spread: across the whole range; random")
    ap.add_argument("--min-points", type=int, default=1,
                    help="ignore frames with fewer than this many points of the class")
    ap.add_argument("--class-id", type=int, nargs="*", default=None,
                    help="restrict to these semantic ids")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV to write (default <analysis-dir>/class_frames.csv)")
    ap.add_argument("--copy", action="store_true",
                    help="also gather the selected .ply files under <analysis-dir>/<copy-dir>,"
                         " one sub-directory per class")
    ap.add_argument("--copy-dir", default="class_frames",
                    help="name of that directory, relative to --analysis-dir")
    ap.add_argument("--link", choices=("copy", "hardlink", "symlink"), default="copy",
                    help="how to place the files: a real copy, or a hardlink/symlink"
                         " (both are instant and use no extra disk)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace files already present in the destination")
    args = ap.parse_args()

    cache = args.analysis_dir / "per_frame_counts.npz"
    stats = args.analysis_dir / "class_stats.csv"
    for p in (cache, stats):
        if not p.exists():
            raise SystemExit("missing %s -- run analyze_seg_labels.py first" % p)

    z = np.load(cache, allow_pickle=False)
    counts, seq_index = z["counts"], z["seq_index"]
    seq_names = [str(s) for s in z["seq_names"]]
    frame_names = [str(s) for s in z["frame_names"]]

    root = args.root
    if root is None:
        summary = args.analysis_dir / "summary.json"
        if not summary.exists():
            raise SystemExit("no --root given and %s is missing" % summary)
        root = Path(json.loads(summary.read_text())["root"])

    classes = load_classes(stats)
    if args.class_id:
        wanted = set(args.class_id)
        classes = [c for c in classes if c[0] in wanted]
        if not classes:
            raise SystemExit("none of %s appear in %s" % (args.class_id, stats))

    rng = np.random.RandomState(args.seed)
    rows, table, missing = [], [], 0

    for cid, name, color in classes:
        idx = pick(counts[:, cid], seq_index, args.top, args.strategy,
                   args.per_seq, args.min_points, rng)
        if not idx:
            table.append([cid, name, "-", "-", "(class never occurs)", ""])
            continue
        for rank, i in enumerate(idx, 1):
            seq = seq_names[int(seq_index[i])]
            frame = frame_names[i]
            path = root / seq / (frame + ".ply")
            if not path.exists():
                missing += 1
            n = int(counts[i, cid])
            rows.append({"class_id": cid, "name": name, "color": color, "rank": rank,
                         "points": n, "pct_of_frame": round(100.0 * n / counts[i].sum(), 4),
                         "sequence": seq, "frame": frame, "path": str(path),
                         "copied_to": ""})
            table.append([cid if rank == 1 else "", name if rank == 1 else "",
                          rank, "{:,}".format(n), seq[:34], frame])

    print(render_table(["id", "name", "#", "points", "sequence", "frame"], table,
                       [">", "<", ">", ">", "<", ">"]))

    dest = None
    if args.copy:
        dest = args.analysis_dir / args.copy_dir
        total = sum(os.path.getsize(r["path"]) for r in rows
                    if os.path.exists(r["path"]))
        print()
        print("gathering %d files (%.2f GiB) into %s via %s ..."
              % (len(rows), total / 2 ** 30, dest, args.link))
        done, skipped, written = gather(rows, dest, args.link, args.overwrite)
        print("  %d placed, %d already there, %.2f GiB written"
              % (done, skipped, written / 2 ** 30))

    out = args.out or (args.analysis_dir / "class_frames.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    txt = out.with_suffix(".paths.txt")
    txt.write_text("\n".join(r["path"] for r in rows) + "\n")

    print()
    print("%d classes x up to %d frames = %d paths  (strategy=%s, per-seq cap=%d)"
          % (len(classes), args.top, len(rows), args.strategy, args.per_seq))
    if missing:
        print("WARNING: %d of those .ply files do not exist on disk" % missing)
    print("wrote %s" % out)
    print("wrote %s  (bare path list)" % txt)
    if dest is not None:
        print("gathered %d class directories under %s" % (len(classes), dest))


if __name__ == "__main__":
    main()
