# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

The Mac side of a LiDAR out-of-distribution (OOD) detection research project. Datasets, GPU jobs and the real training repos live on a server (`/mnt/ssd/khoadv/projects/`); this folder holds copies of the scripts plus results downloaded for viewing in Open3D. Claude does not run anything on the server. For server-side work:

1. Write the exact commands for the user (see Commands).
2. Say what to download and where it goes in this folder.
3. Continue from the downloaded results.

Two independent pipelines:

**A. DSO dataset, semantic-label analysis** (DSO LiDAR sweeps carry per-point panoptic labels)
1. `analyze_seg_labels.py` runs on the server (as `tools/analyze_seg_labels.py` in `OOD_PanSeg_3D`).
   - It scans every `Annotation_Final/<sequence>/<frame>.ply` in a process pool and caches per-frame class histograms in `<out>/per_frame_counts.npz`.
   - It writes `class_stats.csv`, `class_stats_by_sequence.csv` and `summary.json`, plus `class_stats.png` with `--plot`.
   - `--set orig|cetran|all` picks the 12 original sequences, the 3 Cetran ones, or both. Each set has its own default `--out`, so caches never mix.
   - `--reuse` re-reports from the cache without rescanning.
2. `pick_class_frames.py` (server) reads `per_frame_counts.npz`, `class_stats.csv` and `summary.json` instead of rescanning.
   - It picks N frames per class, with at most `--per-seq` per sequence, because 10 Hz sweeps make the densest frames near-duplicates.
   - It writes `class_frames.csv`. `--copy` gathers the files into `class_frames/<NNN>_<name>/<rank>_<seq-slug>_<frame>.ply`.
3. `class_frames/` and `class_frames.csv` are downloaded here. `view_class_frames.py` browses them in one persistent Open3D window (N/P frame, ]/[ class, C colour mode, H highlight).

**B. HD-OOD3D on Argoverse 2, box-matching visualisation**
- `viz_matching_bin.py` runs on the server inside the OpenPCDet-based HD-OOD3D repo (`tools/ood/`, needs `pcdet` + `torch`). The model there is CenterPoint trained on 15 known classes, plus an OOD MLP head. For one frame the script writes seven `.bin` files, one per matching stage, with boxes traced as points.
- They are downloaded to `bb/` and viewed with `read_bin.py`. Each file is `(N, 4)` float32, and the 4th column is a **label**, not intensity:

  | Label | Meaning |
  |---|---|
  | 0 | raw point |
  | 1 | GT known |
  | 2 | GT unknown |
  | 3 | prediction |
  | 4 | matched known |
  | 5 | matched unknown |
- `_6_matched_ood` and `_7_matched_pred` hold the same boxes. Labels 4/5 mean the true OOD label in file 6 and the MLP's call at the FPR-95 threshold in file 7, so comparing the two shows the MLP's errors.
- `LABEL_COLORS` in `read_bin.py` must stay in step with these labels.

## Data formats

- **DSO sweep `.ply`:** binary little-endian, packed 27-byte record:
  - float32 `x y z intensity device_id`
  - uint8 `semantic`
  - uint16 `instance` (0 = none)
  - uint8 `isVisible`
  - uint8 `red green blue` (the label's colour)

  Loaders reject any file whose size isn't exactly header + n × 27.
- **`semantic` ids:** only 1–7 (person, rider, car, truck, bus, motorcycle, bicycle) and 27 (obscurant) have names. 0 and 255 are void. Every other id is an unnamed "stuff" class, reported as `class_NN` with its RGB code. `analyze_seg_labels.py --geometry` (height/range/intensity per class) is how to work out what they are.

## Architecture notes

- Scripts are standalone siblings, not a package, so each can be dropped into a server repo on its own. Shared helpers are therefore copied, not imported, and a change to one copy must go into the other:
  - the PLY reader (`read_ply_header`, `load_frame`) and `CLASS_NAMES`: `analyze_seg_labels.py` ↔ `view_class_frames.py`;
  - `render_table`: `analyze_seg_labels.py` ↔ `pick_class_frames.py`;
  - the viewpoint save/restore helpers: `read_bin.py` ↔ `view_class_frames.py`.
- `analyze_seg_labels.py` and `viz_matching_bin.py` are copies of server files, and nothing syncs them. After editing one, remind the user to copy it to the server. Keep its server-relative default paths intact.
- Stages read the previous stage's files by column/key name, so renaming an output column breaks the next stage:
  - `per_frame_counts.npz`, `class_stats.csv` and `summary.json` feed `pick_class_frames.py`;
  - `class_frames.csv` feeds `view_class_frames.py`.
- `class_frames.csv` stores server paths. The viewer only takes class names and colours from it, and finds the files through `class_frames/`.

## Conventions

- **Python 3.9** (the Mac's system interpreter): no `match`, no `X | Y` unions evaluated at runtime, no 3.10+ stdlib APIs.
- **Dependencies:** numpy is the only hard dependency. Import open3d, matplotlib and tqdm inside the functions that use them, so `--no-view` and listing modes work without open3d.
- **Open3D:** colour arrays must be `(N, 3)` float64 in [0, 1] and C-contiguous, or `Vector3dVector` can segfault. Restore a saved viewpoint after `add_geometry`, because `add_geometry` auto-fits the view.
- **Colormaps:** use `matplotlib.colormaps["viridis"]`. `cm.get_cmap` no longer exists in matplotlib ≥ 3.9.
- **Tests:** plain-assert scripts (`test_<module>.py` with a `main()` runner) on synthetic data written to a temp dir, never the real data folders.

## Commands

On the Mac, from the repo root, with the system `python3` (no venv, no pytest):

```bash
python3 test_view_class_frames.py                                      # all tests
python3 -c "import test_view_class_frames as t; t.test_colorizers()"   # one test
python3 view_class_frames.py                                           # list classes in class_frames/
python3 view_class_frames.py car --no-view                             # per-frame class stats, no window
python3 read_bin.py bb/1000000_6_matched_ood.bin --no-view             # label breakdown of one .bin
# after touching the viewer or the PLY reader: all 140 frames must still load
for d in class_frames/*/; do python3 view_class_frames.py "$(basename "$d")" --no-view >/dev/null || echo "FAIL $d"; done; echo sweep done
```

Without `--no-view`, both viewers open a window that blocks until the user closes it, so leave visual checks to the user.

On the server, for the user to run:

```bash
python3 tools/analyze_seg_labels.py --set orig --instances --geometry --plot  # full scan (--stride 20: quick look)
python3 tools/analyze_seg_labels.py --set orig --reuse                        # re-report from the cache
python pick_class_frames.py --analysis-dir seg_label_analysis --copy          # then download class_frames/ + class_frames.csv
python ood/viz_matching_bin.py --frame_id 1000000 --split val                 # from HD-OOD3D's tools/; download the .bin files to bb/
```

## Repo layout and git

- This folder is the `mac-visualization` branch of the **public** repo `github.com/DuongVanKhoa0811/P3Former_OOD`.
  - It's an orphan branch: it shares no history with the P3Former model branches (`main`, `dso-dataset`, `ood-baselines/*`), so never merge or rebase between them.
  - The server runs `git fetch origin mac-visualization` to read these scripts, which keeps its outputs in the formats the viewers expect.
  - Because the repo is public, never commit credentials, data, or anything from `CLAUDE.local.md`.
- Git tracks code, docs and small configs only. These are git-ignored, so git holds no copy of them:
  - `Cetran/`: three camera runs (folders `1`–`5` of PNGs plus a `multicam_timestamp.txt`) and their zips, ~51 GB.
  - `class_frames/`: 28 classes × 5 frames from the `orig` set, each frame with a `*_camN_panels.png`, ~1.7 GB.
  - `bb/`, `trash/`, all PDFs and all binary formats.
- Never run `git clean -x` or `git clean -X` here. `settings.json` makes `rm` and `git clean` always ask first, and blocks Claude's edits inside the data folders.
- Paper PDFs are grouped by topic in `Baselines/` (OOD baselines), `OD3DPapers/`, `PanopticSegmentation3DPapers/`, `RelatedPapers/` (incl. HD-OOD3D) and `Datasets/`.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` hold the design and plan behind each larger feature.
- Machine-specific notes (interpreter quirks, server details) live in the git-ignored `CLAUDE.local.md` at the repo root.
