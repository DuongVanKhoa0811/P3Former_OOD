#!/usr/bin/env python3
"""Analyse the 3D segmentation labels of the DSO dataset.

Each annotated LiDAR sweep is one binary-little-endian .ply under

    <root>/<sequence>/<frame>.ply

with a fixed 27-byte-per-point record:

    float x, y, z, intensity, device_id     (20 B)
    uchar  semantic                         ( 1 B)   <- semantic class id
    ushort instance                         ( 2 B)   <- instance id, 0 = no instance
    uchar  isVisible                        ( 1 B)
    uchar  red, green, blue                 ( 3 B)   <- colour code of the semantic label

The script answers two questions:

  * how many distinct semantic classes appear in the labels, and
  * how many points each class contributes to an average frame.

Only the `semantic` (and optionally `instance`) column is used, but the whole
file has to be pulled off disk, so the scan is I/O bound and runs in a process
pool.  Per-frame histograms are cached to an .npz so the (expensive) scan can be
run once and the (cheap) reporting re-run with `--reuse`.

Examples
--------
    # quick look: every 20th frame
    python analyze_seg_labels.py --stride 20

    # full scan of all 11.5k frames, with panoptic instance counts
    python analyze_seg_labels.py --instances --out ./seg_label_analysis

    # re-print the report from the cached scan
    python analyze_seg_labels.py --reuse --out ./seg_label_analysis
    
    python3 tools/analyze_seg_labels.py --instances --geometry --workers 48 --out ./seg_label_analysis
    python3 tools/analyze_seg_labels.py --reuse --out ./seg_label_analysis   # re-report, no rescan

    # sequence sets: each --set has its own default --out (and therefore its own
    # cache), so the runs stay fully independent of each other
    #   --set orig    the 12 original sequences   -> ./seg_label_analysis         (default)
    #   --set cetran  the 3 "Cetran" sequences    -> ./seg_label_analysis_cetran
    #   --set all     every sequence under root   -> ./seg_label_analysis_all
    python3 tools/analyze_seg_labels.py --set cetran --instances --geometry --plot
    python3 tools/analyze_seg_labels.py --set cetran --reuse --plot   # re-report Cetran, no rescan
    python3 tools/analyze_seg_labels.py --set orig --reuse            # re-report the original 12
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None


DEFAULT_ROOT = Path("/mnt/ssd/khoadv/projects/OOD_PanSeg_3D/data/DSO_Dataset/Annotation_Final")

# `semantic` is a uchar, so ids live in [0, 255].
N_IDS = 256

# Names for ids 1-7 were harvested from the <label>/<semanticId> pairs in the
# dataset's own 2D/3D bounding-box XMLs.  Id 27 and 255 are named in the vendor's README/Code/annotation.py.
# The remaining ids are "stuff" classes that never receive a box, so the shipped
# data carries no name for them -- they are reported as class_NN together with
# their RGB colour code, which can be matched against the colour-coded 2D
# semantic images.  Override with --names-json '{"8": "road", ...}'.
CLASS_NAMES = {
    0: "unlabeled", # 
    
    1: "person",
    2: "rider",
    3: "car",
    4: "truck",
    5: "bus",
    6: "motorcycle",
    7: "bicycle",
    
    27: "obscurant", # 
    255: "unlabelled", # 
}

# Ids that are void/ignore rather than real classes.
DEFAULT_IGNORE_IDS = (0, 255)

# Named sequence sets, selected with --set.  Each is a pair of case-insensitive
# substring filters over the sequence directory names; None means "no filter".
# An explicit --sequences / --exclude-sequences on the command line replaces the
# corresponding preset filter.
SEQUENCE_SETS = {
    "orig":   {"include": None,        "exclude": ("cetran",)},
    "cetran": {"include": ("cetran",), "exclude": None},
    "all":    {"include": None,        "exclude": None},
}

# Default --out per set: separate directories, so each set keeps its own
# per_frame_counts.npz cache and the runs never mix.
DEFAULT_OUT_BY_SET = {
    "orig":   "seg_label_analysis",
    "cetran": "seg_label_analysis_cetran",
    "all":    "seg_label_analysis_all",
}


# --------------------------------------------------------------------------- #
# PLY reading
# --------------------------------------------------------------------------- #

_PLY_TYPES = {
    b"int8": "i1", b"char": "i1",
    b"uint8": "u1", b"uchar": "u1",
    b"int16": "i2", b"short": "i2",
    b"uint16": "u2", b"ushort": "u2",
    b"int32": "i4", b"int": "i4",
    b"uint32": "u4", b"uint": "u4",
    b"float32": "f4", b"float": "f4",
    b"float64": "f8", b"double": "f8",
}

_ENDIAN = {"binary_big_endian": ">", "binary_little_endian": "<"}


def read_ply_header(fh):
    """Parse a binary PLY header. Returns (n_vertices, vertex_dtype, data_offset)."""
    if b"ply" not in fh.readline():
        raise ValueError("not a PLY file")

    fmt = fh.readline().split()[1].decode()
    if fmt not in _ENDIAN:
        raise ValueError("expected a binary PLY, got format %r" % fmt)
    endian = _ENDIAN[fmt]

    n_vertices = None
    props = []
    element = None
    while True:
        line = fh.readline()
        if not line:
            raise ValueError("header ended without end_header")
        tok = line.split()
        if not tok or tok[0] == b"comment":
            continue
        if tok[0] == b"end_header":
            break
        if tok[0] == b"element":
            element = tok[1]
            if element == b"vertex":
                n_vertices = int(tok[2])
        elif tok[0] == b"property" and element == b"vertex":
            if tok[1] == b"list":
                raise ValueError("list properties on vertex elements are unsupported")
            props.append((tok[2].decode(), endian + _PLY_TYPES[tok[1]]))

    if n_vertices is None:
        raise ValueError("header has no vertex element")
    return n_vertices, np.dtype(props), fh.tell()


def load_frame(path):
    """Read one annotated sweep into a structured array, validating the file size."""
    with open(path, "rb") as fh:
        n, dtype, offset = read_ply_header(fh)
        expected = offset + n * dtype.itemsize
        actual = os.path.getsize(path)
        if actual != expected:
            raise ValueError(
                "size mismatch: header implies %d B (%d points x %d B + %d B header), "
                "file is %d B" % (expected, n, dtype.itemsize, offset, actual)
            )
        data = np.fromfile(fh, dtype=dtype, count=n)
    if data.shape[0] != n:
        raise ValueError("short read: got %d of %d points" % (data.shape[0], n))
    for field in ("semantic", "red", "green", "blue"):
        if field not in data.dtype.names:
            raise ValueError("missing %r property" % field)
    return data


# --------------------------------------------------------------------------- #
# Per-frame scan (runs in worker processes)
# --------------------------------------------------------------------------- #

# Per-process colour cache: after the first few frames a worker has seen every
# class it will ever see, so the lookup below costs nothing for the rest of the run.
_PALETTE = {}


def scan_frame(task):
    """Histogram one frame's semantic ids. Returns a plain dict (pickled back)."""
    idx, path, want_instances, want_geometry = task
    try:
        data = load_frame(path)
    except Exception as exc:  # a single bad file must not kill the whole scan
        return {"idx": idx, "error": "%s: %s" % (type(exc).__name__, exc)}

    sem = np.asarray(data["semantic"], dtype=np.intp)
    counts = np.bincount(sem, minlength=N_IDS)

    geo = None
    if want_geometry:
        z = data["z"].astype(np.float64)
        rng = np.hypot(data["x"].astype(np.float64), data["y"].astype(np.float64))
        inten = data["intensity"].astype(np.float64)
        geo = {
            "sum_z": np.bincount(sem, weights=z, minlength=N_IDS),
            "sumsq_z": np.bincount(sem, weights=z * z, minlength=N_IDS),
            "sum_range": np.bincount(sem, weights=rng, minlength=N_IDS),
            "sum_intensity": np.bincount(sem, weights=inten, minlength=N_IDS),
        }

    inst_counts = None
    if want_instances:
        # Distinct non-zero instance ids per class. semantic is 8 bit and
        # instance is 16 bit, so (semantic << 16) | instance is a lossless key.
        key = (sem.astype(np.int64) << 16) | np.asarray(data["instance"], dtype=np.int64)
        uniq = np.unique(key)
        uniq = uniq[(uniq & 0xFFFF) != 0]
        inst_counts = np.bincount((uniq >> 16).astype(np.intp), minlength=N_IDS)

    new_colors = {}
    for cid in np.nonzero(counts)[0]:
        cid = int(cid)
        if cid not in _PALETTE:
            i = int(np.argmax(sem == cid))
            rgb = (int(data["red"][i]), int(data["green"][i]), int(data["blue"][i]))
            _PALETTE[cid] = rgb
            new_colors[cid] = rgb

    return {
        "idx": idx,
        "n_points": int(sem.shape[0]),
        "counts": counts.astype(np.int64),
        "inst_counts": None if inst_counts is None else inst_counts.astype(np.int64),
        "geo": geo,
        "colors": new_colors,
    }


# --------------------------------------------------------------------------- #
# Scan driver
# --------------------------------------------------------------------------- #

def discover_frames(root, sequences=None, exclude=None, stride=1, max_frames=None):
    """Return (seq_names, frame_names, seq_index, paths) for the frames to scan.

    `sequences` keeps only directories whose name contains one of the given
    substrings; `exclude` then drops directories matching one of its substrings.
    Both are case-insensitive; None disables the filter.
    """
    seq_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if sequences:
        wanted = [s.lower() for s in sequences]
        seq_dirs = [d for d in seq_dirs if any(w in d.name.lower() for w in wanted)]
    if exclude:
        banned = [s.lower() for s in exclude]
        seq_dirs = [d for d in seq_dirs if not any(b in d.name.lower() for b in banned)]
    if not seq_dirs:
        raise SystemExit("no sequence directories found under %s" % root)

    seq_names, frame_names, seq_index, paths = [], [], [], []
    for si, d in enumerate(seq_dirs):
        files = sorted(d.glob("*.ply"))[::stride]
        if max_frames:
            files = files[:max_frames]
        if not files:
            print("warning: no .ply files in %s" % d, file=sys.stderr)
        seq_names.append(d.name)
        for f in files:
            frame_names.append(f.stem)
            seq_index.append(si)
            paths.append(str(f))
    if not paths:
        raise SystemExit("no .ply files found under %s" % root)
    return seq_names, frame_names, np.asarray(seq_index, dtype=np.int32), paths


GEO_KEYS = ("sum_z", "sumsq_z", "sum_range", "sum_intensity")


def run_scan(paths, workers, want_instances, want_geometry):
    """Scan every frame in a process pool.

    Returns (counts, n_points, inst, geo, palette, errors).  `counts` and `inst`
    are per frame; `geo` accumulates dataset-wide sums per class.
    """
    n = len(paths)
    counts = np.zeros((n, N_IDS), dtype=np.int64)
    inst = np.zeros((n, N_IDS), dtype=np.int64) if want_instances else None
    geo = {k: np.zeros(N_IDS, dtype=np.float64) for k in GEO_KEYS} if want_geometry else None
    n_points = np.zeros(n, dtype=np.int64)
    palette = {}
    errors = []

    tasks = [(i, p, want_instances, want_geometry) for i, p in enumerate(paths)]
    chunksize = max(1, min(16, n // (workers * 8) or 1))
    
    import ipdb; ipdb.set_trace()
    scan_frame(tasks[0])
    
    with ProcessPoolExecutor(max_workers=workers) as pool:
        stream = pool.map(scan_frame, tasks, chunksize=chunksize)
        if tqdm is not None:
            stream = tqdm(stream, total=n, unit="frame", dynamic_ncols=True)
        else:
            stream = _plain_progress(stream, n)
        for res in stream:
            i = res["idx"]
            if "error" in res:
                errors.append((i, paths[i], res["error"]))
                continue
            counts[i] = res["counts"]
            n_points[i] = res["n_points"]
            if inst is not None:
                inst[i] = res["inst_counts"]
            if geo is not None:
                for k in GEO_KEYS:
                    geo[k] += res["geo"][k]
            for cid, rgb in res["colors"].items():
                palette.setdefault(cid, rgb)

    return counts, n_points, inst, geo, palette, errors


def _plain_progress(stream, total, every=200):
    t0 = time.time()
    for i, item in enumerate(stream, 1):
        if i % every == 0 or i == total:
            el = time.time() - t0
            eta = el / i * (total - i)
            print("  %d/%d frames  %.0fs elapsed  ~%.0fs left" % (i, total, el, eta),
                  file=sys.stderr, flush=True)
        yield item


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

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


def class_name(cid, names):
    return names.get(cid, "class_%02d" % cid)


def class_rows(counts, inst, ids, names, palette):
    """Per-class statistics over a set of frames (counts is (n_frames, N_IDS))."""
    n_frames = counts.shape[0]
    grand_total = int(counts.sum())
    rows = []
    for cid in ids:
        col = counts[:, cid]
        total = int(col.sum())
        present = int(np.count_nonzero(col))
        vals = col[col > 0]
        rgb = palette.get(cid)
        row = {
            "id": cid,
            "name": class_name(cid, names),
            "color": "" if rgb is None else "%d,%d,%d" % rgb,
            "total_points": total,
            "pct_of_all_points": 100.0 * total / grand_total if grand_total else 0.0,
            "frames_present": present,
            "pct_frames_present": 100.0 * present / n_frames if n_frames else 0.0,
            "avg_points_per_frame": total / n_frames if n_frames else 0.0,
            "avg_points_per_present_frame": total / present if present else 0.0,
            "median_points_per_present_frame": float(np.median(vals)) if present else 0.0,
            "max_points_in_a_frame": int(col.max()) if n_frames else 0,
        }
        if inst is not None:
            icol = inst[:, cid]
            itotal = int(icol.sum())
            ipresent = int(np.count_nonzero(icol))
            row["total_instances"] = itotal
            row["avg_instances_per_frame"] = itotal / n_frames if n_frames else 0.0
            row["avg_instances_per_present_frame"] = itotal / ipresent if ipresent else 0.0
            row["avg_points_per_instance"] = total / itotal if itotal else 0.0
        rows.append(row)
    return rows


def print_geometry(counts, geo, ids, names):
    """Height/range/intensity profile per class.

    The dataset ships names only for the 7 box classes, so this is the practical
    way to work out what the remaining ids are: ground-like classes sit at a
    constant low z with tiny spread, tall structure has a large z spread, and so on.
    """
    rows = []
    for cid in ids:
        n = int(counts[:, cid].sum())
        if n == 0:
            continue
        mean_z = geo["sum_z"][cid] / n
        var_z = max(0.0, geo["sumsq_z"][cid] / n - mean_z * mean_z)
        rows.append([cid, class_name(cid, names),
                     "%.2f" % mean_z, "%.2f" % var_z ** 0.5,
                     "%.1f" % (geo["sum_range"][cid] / n),
                     "%.1f" % (geo["sum_intensity"][cid] / n)])
    print()
    print("Geometry profile per class (all points pooled; z is IMU-frame height)")
    print(render_table(["id", "name", "mean z", "std z", "mean range", "mean intens"],
                       rows, [">", "<", ">", ">", ">", ">"]))


def print_report(rows, title, want_instances):
    headers = ["id", "name", "rgb", "avg pts/frame", "avg pts/frame*",
               "median*", "max", "frames", "%frames", "total pts", "%pts"]
    aligns = [">", "<", ">", ">", ">", ">", ">", ">", ">", ">", ">"]
    if want_instances:
        headers += ["avg inst/frame", "pts/inst"]
        aligns += [">", ">"]

    table = []
    for r in rows:
        line = [
            r["id"], r["name"], r["color"],
            "%.1f" % r["avg_points_per_frame"],
            "%.1f" % r["avg_points_per_present_frame"],
            "%.0f" % r["median_points_per_present_frame"],
            "{:,}".format(r["max_points_in_a_frame"]),
            "{:,}".format(r["frames_present"]),
            "%.1f" % r["pct_frames_present"],
            "{:,}".format(r["total_points"]),
            "%.3f" % r["pct_of_all_points"],
        ]
        if want_instances:
            line += ["%.2f" % r["avg_instances_per_frame"],
                     "%.0f" % r["avg_points_per_instance"] if r["total_instances"] else "-"]
        table.append(line)

    print()
    print(title)
    print(render_table(headers, table, aligns))
    print("  avg pts/frame  = total points of the class / ALL frames scanned")
    print("  avg pts/frame* = total points of the class / frames where the class occurs")
    print("  median*        = median points per frame, over frames where the class occurs")


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in r.items()})


def plot_class_stats(rows, png_path, n_frames):
    """Bar chart of total_points (and total_instances, when counted) per class.

    One bar per class from the `rows` produced by class_rows(), heaviest first,
    each bar wearing the dataset's own RGB colour code for that class.  The y
    axis is logarithmic because the class totals span ~4 orders of magnitude
    (class_20 at 576k points vs class_24 at 1.6B) -- on a linear axis every
    minority class would be an invisible sliver.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    rows = sorted(rows, key=lambda r: -r["total_points"])
    if not rows:
        print("nothing to plot")
        return
    names = [r["name"] for r in rows]
    colors = []
    for r in rows:
        try:
            colors.append(tuple(int(c) / 255.0 for c in r["color"].split(",")))
        except ValueError:
            colors.append((0.6, 0.6, 0.6))

    def compact(v):
        for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
            if v >= div:
                return ("%.1f" if v < 10 * div else "%.0f") % (v / div) + suf
        return "%d" % int(v)

    metrics = [("total_points", "total points per class (log scale)")]
    if "total_instances" in rows[0]:
        metrics.append(("total_instances",
                        "total distinct instances per class (log scale)"))

    fig, axes = plt.subplots(len(metrics), 1,
                             figsize=(13.0, 4.3 * len(metrics)), squeeze=False)
    fig.suptitle("DSO 3D segmentation labels - class totals over {:,} frames"
                 .format(n_frames), x=0.02, ha="left", fontsize=13, color="#0b0b0b")
    xs = list(range(len(rows)))
    for ax, (key, title) in zip(axes[:, 0], metrics):
        vals = [r[key] for r in rows]
        bars = ax.bar(xs, vals, width=0.72, color=colors, log=True)
        ax.bar_label(bars, labels=[compact(v) for v in vals],
                     fontsize=7, color="#52514e", padding=2)
        positive = [v for v in vals if v > 0]
        if positive:
            ax.set_ylim(min(positive) / 2.0, max(positive) * 4.0)
        ax.set_title(title, loc="left", fontsize=10.5, color="#52514e", pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, fontsize=8.5, rotation=45, ha="right")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))
        ax.tick_params(colors="#52514e", labelsize=8.5, length=0)
        ax.grid(axis="y", which="major", color="#e1e0d9", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print("wrote %s" % png_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Analyse DSO dataset 3D segmentation labels (.ply).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="directory holding one subdirectory per sequence")
    p.add_argument("--set", choices=sorted(SEQUENCE_SETS), default="orig",
                   help="named sequence set to analyse: 'orig' = the 12 original "
                        "sequences, 'cetran' = the 3 Cetran sequences, 'all' = "
                        "everything under root; each set defaults to its own --out")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory for CSV/JSON/npz (default depends on "
                        "--set: %s)" % ", ".join(
                            "%s -> ./%s" % (k, v) for k, v in DEFAULT_OUT_BY_SET.items()))
    p.add_argument("--sequences", nargs="+", default=None,
                   help="only scan sequences whose name contains one of these "
                        "substrings (replaces the --set include filter)")
    p.add_argument("--exclude-sequences", nargs="+", default=None,
                   help="skip sequences whose name contains one of these "
                        "substrings (replaces the --set exclude filter)")
    p.add_argument("--stride", type=int, default=1,
                   help="scan every Nth frame of each sequence")
    p.add_argument("--max-frames", type=int, default=None,
                   help="cap the number of frames scanned per sequence")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8),
                   help="worker processes (the scan is disk bound)")
    p.add_argument("--instances", action="store_true",
                   help="also count distinct instance ids per class (panoptic stats)")
    p.add_argument("--geometry", action="store_true",
                   help="also profile height/range/intensity per class -- useful for "
                        "identifying the stuff classes that ship without a name")
    p.add_argument("--per-sequence", action="store_true",
                   help="also print a full class table for every sequence")
    p.add_argument("--ignore-ids", type=int, nargs="*", default=list(DEFAULT_IGNORE_IDS),
                   help="ids treated as void rather than as classes")
    p.add_argument("--names-json", type=Path, default=None,
                   help="JSON file mapping semantic id -> class name, merged over the built-ins")
    p.add_argument("--reuse", action="store_true",
                   help="reload per-frame counts from <out>/per_frame_counts.npz instead of rescanning")
    p.add_argument("--plot", action="store_true",
                   help="also draw <out>/class_stats.png: total points (and total "
                        "instances, when counted) per class, log scale, bars in "
                        "the dataset's own class colours")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    preset = SEQUENCE_SETS[args.set]
    include = args.sequences if args.sequences is not None else preset["include"]
    exclude = (args.exclude_sequences if args.exclude_sequences is not None
               else preset["exclude"])
    if args.out is None:
        args.out = Path(DEFAULT_OUT_BY_SET[args.set])
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "per_frame_counts.npz"

    names = dict(CLASS_NAMES)
    if args.names_json:
        names.update({int(k): v for k, v in json.loads(args.names_json.read_text()).items()})

    errors = []
    if args.reuse:
        if not cache.exists():
            raise SystemExit("--reuse given but %s does not exist" % cache)
        z = np.load(cache, allow_pickle=False)
        counts, n_points, seq_index = z["counts"], z["n_points"], z["seq_index"]
        seq_names = [str(s) for s in z["seq_names"]]
        frame_names = [str(s) for s in z["frame_names"]]
        inst = z["inst_counts"] if "inst_counts" in z.files else None
        geo = {k: z["geo_" + k] for k in GEO_KEYS} if "geo_sum_z" in z.files else None
        pal = z["palette"]
        palette = {i: tuple(int(v) for v in pal[i]) for i in range(N_IDS) if pal[i][0] >= 0}
        args.instances = inst is not None
        args.geometry = geo is not None
        print("loaded cached scan of %d frames from %s" % (counts.shape[0], cache))
    else:
        if not args.root.is_dir():
            raise SystemExit("root %s is not a directory" % args.root)
        seq_names, frame_names, seq_index, paths = discover_frames(
            args.root, include, exclude, args.stride, args.max_frames)
        total_bytes = sum(os.path.getsize(p) for p in paths)
        print("root      : %s" % args.root)
        print("set       : %s (include: %s, exclude: %s)"
              % (args.set, ", ".join(include) if include else "-",
                 ", ".join(exclude) if exclude else "-"))
        print("sequences : %d" % len(seq_names))
        print("frames    : %d  (%.1f GiB to read)" % (len(paths), total_bytes / 2**30))
        print("workers   : %d" % args.workers)
        print()

        t0 = time.time()
        counts, n_points, inst, geo, palette, errors = run_scan(
            paths, args.workers, args.instances, args.geometry)
        elapsed = time.time() - t0
        print("scanned in %.1fs (%.0f frames/s, %.0f MiB/s)"
              % (elapsed, len(paths) / elapsed, total_bytes / 2**20 / elapsed))

        if errors:
            ok = np.ones(len(paths), dtype=bool)
            for i, _, _ in errors:
                ok[i] = False
            counts, n_points, seq_index = counts[ok], n_points[ok], seq_index[ok]
            if inst is not None:
                inst = inst[ok]
            frame_names = [f for f, keep in zip(frame_names, ok) if keep]

        pal = np.full((N_IDS, 3), -1, dtype=np.int16)
        for cid, rgb in palette.items():
            pal[cid] = rgb
        payload = dict(counts=counts, n_points=n_points, seq_index=seq_index,
                       seq_names=np.array(seq_names), frame_names=np.array(frame_names),
                       palette=pal)
        if inst is not None:
            payload["inst_counts"] = inst
        if geo is not None:
            payload.update({"geo_" + k: v for k, v in geo.items()})
        np.savez_compressed(cache, **payload)

    # ---- summary -------------------------------------------------------- #
    n_frames = counts.shape[0]
    total_points = int(counts.sum())
    present_ids = [int(c) for c in np.nonzero(counts.sum(axis=0))[0]]
    ignore = sorted(set(args.ignore_ids) & set(present_ids))
    class_ids = [c for c in present_ids if c not in set(args.ignore_ids)]

    print()
    print("=" * 78)
    print("DSO 3D segmentation labels")
    print("=" * 78)
    print("sequences scanned    : %d" % len(seq_names))
    print("frames scanned       : {:,}".format(n_frames))
    print("points labelled      : {:,}".format(total_points))
    print("points per frame     : {:,.0f} mean, {:,} min, {:,} max"
          .format(n_points.mean(), int(n_points.min()), int(n_points.max())))
    print("ids present          : %d  (%s)"
          % (len(present_ids), ", ".join(str(c) for c in present_ids)))
    print("void / ignore ids    : %s" % (", ".join("%d (%s)" % (c, class_name(c, names))
                                                   for c in ignore) or "none"))
    print("=> SEMANTIC CLASSES  : %d" % len(class_ids))
    if errors:
        print("unreadable frames    : %d (see %s)" % (len(errors), args.out / "errors.txt"))

    rows = class_rows(counts, inst, class_ids + ignore, names, palette)
    print_report(rows, "Per-class statistics over all %s scanned frames"
                 % "{:,}".format(n_frames), args.instances)
    if geo is not None:
        print_geometry(counts, geo, class_ids + ignore, names)

    # ---- per sequence ---------------------------------------------------- #
    seq_rows = []
    for si, sname in enumerate(seq_names):
        m = seq_index == si
        if not m.any():
            continue
        sub, sub_inst = counts[m], (inst[m] if inst is not None else None)
        for r in class_rows(sub, sub_inst, class_ids + ignore, names, palette):
            r = dict(r)
            r["sequence"] = sname
            r["frames_in_sequence"] = int(m.sum())
            seq_rows.append(r)
        if args.per_sequence:
            print_report([r for r in seq_rows if r["sequence"] == sname],
                         "%s  (%d frames)" % (sname, int(m.sum())), args.instances)

    print()
    print("Per-sequence overview")
    ov = []
    for si, sname in enumerate(seq_names):
        m = seq_index == si
        if not m.any():
            continue
        sub = counts[m]
        n_cls = int(np.count_nonzero([sub[:, c].sum() for c in class_ids]))
        ov.append([sname, "{:,}".format(int(m.sum())), "{:,}".format(int(sub.sum())),
                   "{:,.0f}".format(sub.sum() / m.sum()), n_cls])
    print(render_table(["sequence", "frames", "points", "pts/frame", "classes"],
                       ov, ["<", ">", ">", ">", ">"]))

    # ---- files ----------------------------------------------------------- #
    fields = list(rows[0].keys())
    write_csv(args.out / "class_stats.csv", rows, fields)
    write_csv(args.out / "class_stats_by_sequence.csv", seq_rows,
              ["sequence", "frames_in_sequence"] + fields)
    if args.plot:
        plot_class_stats([r for r in rows if r["id"] not in set(args.ignore_ids)],
                         args.out / "class_stats.png", n_frames)
    summary = {
        "root": str(args.root),
        "sequence_set": args.set,
        "include_filter": list(include) if include else None,
        "exclude_filter": list(exclude) if exclude else None,
        "sequences": seq_names,
        "frames_scanned": n_frames,
        "stride": args.stride,
        "total_points": total_points,
        "ids_present": present_ids,
        "ignore_ids": ignore,
        "n_semantic_classes": len(class_ids),
        "class_ids": class_ids,
        "class_names": {str(c): class_name(c, names) for c in present_ids},
        "class_colors": {str(c): list(palette[c]) for c in present_ids if c in palette},
        "avg_points_per_frame": {class_name(c, names): r["avg_points_per_frame"]
                                 for c, r in zip(class_ids + ignore, rows)},
    }
    if geo is not None:
        tot = counts.sum(axis=0)
        summary["geometry"] = {
            str(c): {
                "mean_z": geo["sum_z"][c] / tot[c],
                "std_z": max(0.0, geo["sumsq_z"][c] / tot[c]
                             - (geo["sum_z"][c] / tot[c]) ** 2) ** 0.5,
                "mean_range": geo["sum_range"][c] / tot[c],
                "mean_intensity": geo["sum_intensity"][c] / tot[c],
            }
            for c in present_ids if tot[c]
        }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    if errors:
        (args.out / "errors.txt").write_text(
            "\n".join("%s\t%s" % (p, e) for _, p, e in errors) + "\n")

    print()
    print("wrote %s/{class_stats.csv, class_stats_by_sequence.csv, summary.json, "
          "per_frame_counts.npz}" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
