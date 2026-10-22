# Class-frame point-cloud viewer — design

Date: 2026-08-07
Status: approved (interactive-viewer option chosen over static gallery / both)

## Problem

`pick_class_frames.py` gathered up to 5 representative `.ply` frames per semantic
class into `class_frames/<NNN>_<name>/<rank>_<seq>_<frame>.ply` (28 classes,
140 files, ~450k points each), with per-class metadata in `class_frames.csv`.
There is no way to look at them. `read_bin.py` shows the desired UX (Open3D
viewer, remembered camera pose) but reads a different format (.bin with a label
channel) and opens one file per run — unusable for flipping through 140 files.

## Deliverable

One script, `view_class_frames.py`, at the project root. Python 3.9,
numpy + open3d (matplotlib optional, for viridis gradients only).

## CLI

```
python3 view_class_frames.py car          # class by name (case-insensitive substring)
python3 view_class_frames.py 3            # class by id
python3 view_class_frames.py 003_car      # class by directory name
python3 view_class_frames.py path/to.ply  # a single file directly
python3 view_class_frames.py              # no arg: list classes + frame counts, exit
```

Name resolution: an exact name match wins; otherwise a unique substring match
is accepted; multiple substring matches print the candidates and exit 2.

Options: `--frames-dir` (default `class_frames` beside the script), `--csv`
(default `class_frames.csv` beside the script), `--color
{semantic,highlight,intensity,height}` (default `semantic`), `--point-size N`,
`--viewpoint FILE` (default `view_class_frames_viewpoint.json` beside the
script), `--reset-view`, `--no-view` (per-frame stats only, no window).

## Data loading

Reuse the binary-PLY reader from `analyze_seg_labels.py` (`read_ply_header` +
`load_frame`, copied in — the tools are siblings, not a package). Fields used:
`x y z` (geometry), `intensity`, `semantic`, `red green blue` (the dataset's own
label colors). File-size validation kept.

Class metadata: `class_frames.csv` provides id → name and id → RGB. If the CSV
is missing, fall back to names parsed from `class_frames/` directory names and
take the highlight color from the target-class points' own RGB (mode value).

## Color modes

- `semantic` (default): per-point `red,green,blue / 255` — the dataset palette.
- `highlight`: points of the target class in the class color at full
  brightness; all other points dim grey (0.35). Only meaningful when a target
  class is known (opened via class, not bare path) — otherwise skipped in the
  cycle.
- `intensity`, `height`: viridis over robust-normalized (2nd–98th pct)
  intensity / z. Without matplotlib these modes fall back to a grey ramp.

## Viewer

`o3d.visualization.VisualizerWithKeyCallback`, one persistent window.

Keys: `N`/`P` next/prev frame within class (rank order, wraps), `]`/`[`
next/prev class (jumps to that class's first frame), `C` cycle color mode,
`H` toggle highlight against the previous mode. GLFW keycodes for letters and
brackets.

Camera: pose restored from the viewpoint JSON at startup (after the first
`add_geometry`, which auto-fits); on frame/class switch the point buffer is
swapped with `update_geometry` and the camera pose is captured before and
re-applied after, so the view never jumps while flipping. Pose saved back to
the JSON when the window closes. Same convert_to/from_pinhole pattern as
`read_bin.py`, `allow_arbitrary=True`.

Terminal output per displayed frame: class name/id, rank, sequence, frame id,
point count, then a table of semantic ids present (id, name, points, % of
frame) with the target class marked `<<`. The window title is static (legacy
Open3D cannot retitle a live window); the terminal line printed on every
switch identifies the current class/rank/sequence/frame.

## Error handling

- Unresolvable class arg → print the class list and exit 2.
- Unreadable/truncated `.ply` while flipping → print the error, skip to the
  next frame (never crash the window).
- Missing CSV → warn once, use fallbacks above.
- Missing open3d → stats still work with `--no-view`; otherwise exit with the
  pip hint (same behavior as read_bin.py).

## Testing

Pure parts get a standalone `test_view_class_frames.py` (plain asserts, no
pytest dependency, runnable as `python3 test_view_class_frames.py`): build a tiny synthetic binary PLY in tmp, then
assert (a) loader round-trips counts/fields, (b) class resolution (id, name,
dirname, ambiguous, miss), (c) each colorizer returns (N,3) float64 in [0,1],
C-contiguous, with highlight marking exactly the target-class rows. GUI paths
are exercised manually by the user (`python3 view_class_frames.py car`).

## Out of scope (YAGNI)

Batch PNG snapshots, instance coloring, bounding boxes, multi-frame overlays,
web viewer. The static-gallery option was explicitly not chosen.
