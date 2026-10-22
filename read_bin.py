#!/usr/bin/env python3
"""Open and visualize a matching-stage point-cloud .bin file in 3D.

These .bin files are written by viz_matching_bin.py as (N, 4) float32, where
the 4th channel is a discrete LABEL (not intensity):

    0 = raw LiDAR point            -> grey
    1 = GT box, known class        -> green
    2 = GT box, unknown class      -> yellow
    3 = prediction box             -> orange
    4 = matched prediction, KNOWN  -> blue
    5 = matched prediction, UNKNOWN-> pink

Labels 4/5 carry a different meaning per file, so the same colors let you diff
the two directly:
    *_6_matched_ood.bin   4/5 = the TRUE OOD label (what the matcher assigns)
    *_7_matched_pred.bin  4/5 = the MLP's own prediction
A box that is blue in file 6 but pink in file 7 is an MLP false positive.

This script loads the file into an (N, 4) numpy array, prints a per-label
breakdown, and opens an interactive Open3D viewer with points colored by label.

The camera pose (position/angle/zoom) is remembered between runs: on close it
is written to read_bin_viewpoint.json, and on the next open it is restored, so
you set up a good view once and keep it. Use --reset-view to start fresh.

Usage:
    python read_bin.py                          # default file, colored by label
    python read_bin.py 1000000_1_all.bin        # a specific file
    python read_bin.py 1000000_1_all.bin --color height   # color by z instead
    python read_bin.py 1000000_1_all.bin --reset-view     # ignore saved camera
    python read_bin.py 1000000_1_all.bin --no-view        # stats only

Dependencies:
    pip install numpy open3d matplotlib
"""

import argparse
import os

import numpy as np

DEFAULT_BIN = "/Users/anhlee/Downloads/Projects/DSO_Ocularis/OOD_OD_3D/1000000_1_all.bin"

# Discrete label (4th channel) -> RGB in [0, 1], matching viz_matching_bin.py.
LABEL_COLORS = {
    0.0: (0.60, 0.60, 0.60),   # grey   - raw LiDAR point
    1.0: (0.00, 0.80, 0.00),   # green  - GT box, known class
    2.0: (1.00, 0.90, 0.00),   # yellow - GT box, unknown class
    3.0: (1.00, 0.55, 0.00),   # orange - prediction box
    4.0: (0.10, 0.45, 1.00),   # blue   - matched prediction, KNOWN
    5.0: (1.00, 0.45, 0.75),   # pink   - matched prediction, UNKNOWN
}
LABEL_NAMES = {
    0.0: "raw LiDAR",
    1.0: "GT known",
    2.0: "GT unknown",
    3.0: "prediction",
    4.0: "matched known",
    5.0: "matched unknown",
}
# Any label not listed above is drawn in cyan so it stands out as unexpected.
# (Cyan rather than magenta, which would read as a shade of the label-5 pink.)
FALLBACK_COLOR = (0.0, 1.0, 1.0)

# Viewer window size and where the camera viewpoint is remembered between runs.
WINDOW_W, WINDOW_H = 1280, 800
VIEWPOINT_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "read_bin_viewpoint.json")


def load_point_cloud(path, num_features=4):
    """Load a raw .bin file into an (N, num_features) float32 array.

    The matching-stage exports use num_features=4 -> (x, y, z, label).
    """
    points = np.fromfile(path, dtype=np.float32)
    if points.size % num_features != 0:
        raise ValueError(
            f"{path}: {points.size} float32 values are not divisible by "
            f"num_features={num_features}; the file may use a different layout."
        )
    return points.reshape(-1, num_features)


def summarize(points):
    """Print shape, per-column stats, and the per-label breakdown."""
    print(f"Points loaded : {points.shape[0]:,}")
    print(f"Features/point: {points.shape[1]} (x, y, z, label)")
    print(f"Dtype         : {points.dtype}\n")

    header = f"{'col':<10}{'min':>12}{'max':>12}{'mean':>12}{'std':>12}"
    print(header)
    print("-" * len(header))
    names = ("x", "y", "z", "label")
    for i, name in enumerate(names[: points.shape[1]]):
        col = points[:, i]
        print(f"{name:<10}{col.min():>12.3f}{col.max():>12.3f}"
              f"{col.mean():>12.3f}{col.std():>12.3f}")

    if points.shape[1] >= 4:
        print("\nLabel breakdown:")
        values, counts = np.unique(points[:, 3], return_counts=True)
        for value, count in zip(values, counts):
            name = LABEL_NAMES.get(float(value), "unknown label")
            print(f"  {value:>5.1f}  {name:<16s} {count:>10,d}")


def _normalize(values):
    """Scale a 1-D array to [0, 1] for colormapping (robust to constants)."""
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def colorize_by_label(points):
    """Map the 4th-channel label of each point to a fixed RGB color.

    Returns an (N, 3) C-contiguous float64 array. Points whose label is not in
    LABEL_COLORS get FALLBACK_COLOR (magenta).
    """
    labels = points[:, 3]
    colors = np.tile(FALLBACK_COLOR, (points.shape[0], 1)).astype(np.float64)
    for value, rgb in LABEL_COLORS.items():
        colors[labels == value] = rgb
    # Open3D's Vector3dVector segfaults on non-contiguous input, so make sure.
    return np.ascontiguousarray(colors, dtype=np.float64)


def colorize_continuous(points, color_by):
    """Color points along a viridis gradient over height (z) or the 4th column."""
    try:
        import matplotlib.cm as cm
    except ImportError:
        return None  # caller falls back to uniform gray
    source = points[:, 2] if color_by == "height" else points[:, 3]
    rgb = cm.get_cmap("viridis")(_normalize(source))[:, :3]
    # The colormap returns a non-contiguous RGBA slice; make it contiguous.
    return np.ascontiguousarray(rgb, dtype=np.float64)


def colorize(points, color_by="label"):
    """Return an (N, 3) C-contiguous RGB array in [0, 1], or None if unavailable."""
    if color_by == "label":
        return colorize_by_label(points)
    return colorize_continuous(points, color_by)


def _print_legend(points, color_by):
    """Print which labels are present and the color each maps to."""
    if color_by != "label":
        return
    present = set(np.unique(points[:, 3]).tolist())
    for value in sorted(LABEL_COLORS):
        if value in present:
            r, g, b = LABEL_COLORS[value]
            print(f"  {LABEL_NAMES[value]:<16s} label {value:.0f} "
                  f"-> RGB({r:.2f}, {g:.2f}, {b:.2f})")
    if present & {4.0, 5.0}:
        # 4/5 mean the true label in file 6 but the MLP's call in file 7.
        print("  (labels 4/5 = TRUE OOD label in *_6_matched_ood.bin, "
              "MLP prediction in *_7_matched_pred.bin)")


def _restore_viewpoint(o3d, view_control, viewpoint_file):
    """Apply a previously saved camera pose to the live window, if one exists."""
    if not viewpoint_file or not os.path.isfile(viewpoint_file):
        return
    try:
        param = o3d.io.read_pinhole_camera_parameters(viewpoint_file)
        # allow_arbitrary=True tolerates a window size that differs from the
        # one the viewpoint was saved at (e.g. if you resized before closing).
        ok = view_control.convert_from_pinhole_camera_parameters(param, True)
        print(f"Restored viewpoint from {viewpoint_file}" if ok else
              "[viewpoint] saved file did not apply cleanly; using default view")
    except Exception as exc:  # never let a bad file break the viewer
        print(f"[viewpoint] could not restore ({exc}); using default view")


def _save_viewpoint(o3d, view_control, viewpoint_file):
    """Write the current camera pose so the next run can restore it."""
    if not viewpoint_file:
        return
    try:
        param = view_control.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(viewpoint_file, param)
        print(f"Saved viewpoint to {viewpoint_file}")
    except Exception as exc:
        print(f"[viewpoint] could not save ({exc})")


def visualize(points, color_by="label", show_axes=False,
              viewpoint_file=VIEWPOINT_FILE, restore_view=True):
    """Open an interactive Open3D viewer, restoring/saving the camera pose.

    The camera position/angle/zoom is loaded from viewpoint_file on open (if it
    exists) and written back when the window is closed, so the view you set up
    is remembered across runs.
    """
    try:
        import open3d as o3d
    except ImportError:
        print("\n[3D view skipped] Open3D is not installed "
              "(pip install open3d).")
        return

    pcd = o3d.geometry.PointCloud()
    xyz = np.ascontiguousarray(points[:, :3], dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz)

    colors = colorize(points, color_by)
    if colors is None:
        # Fall back to a flat gray rather than let Open3D tint by height.
        colors = np.ascontiguousarray(
            np.tile([0.6, 0.6, 0.6], (points.shape[0], 1)), dtype=np.float64)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Use the Visualizer class (not draw_geometries) so that run() returns when
    # the window is closed, giving us a chance to read back the camera pose.
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Matching-stage point cloud",
                      width=WINDOW_W, height=WINDOW_H)
    vis.add_geometry(pcd)
    if show_axes:
        # A coordinate frame at the sensor origin for spatial reference.
        # Note: on some Open3D builds this mesh helper is unstable, so it is
        # opt-in via --axes rather than shown by default.
        vis.add_geometry(
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0))

    view_control = vis.get_view_control()
    if restore_view:
        # Apply AFTER add_geometry, which auto-fits (and would override) the view.
        _restore_viewpoint(o3d, view_control, viewpoint_file)

    print(f"\nOpening Open3D viewer (colored by {color_by}). "
          "Move to a good angle, then close the window to remember it.")
    _print_legend(points, color_by)

    vis.run()  # blocks until the user closes the window

    _save_viewpoint(o3d, view_control, viewpoint_file)
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", nargs="?", default=DEFAULT_BIN,
                        help="Path to the .bin point-cloud file.")
    parser.add_argument("--features", type=int, default=4,
                        help="Floats per point (default: 4 -> x, y, z, label).")
    parser.add_argument("--color", choices=("label", "height", "intensity"),
                        default="label",
                        help="How to color points (default: label -> the fixed "
                             "grey/green/yellow/orange scheme).")
    parser.add_argument("--axes", action="store_true",
                        help="Draw a coordinate frame at the sensor origin.")
    parser.add_argument("--viewpoint", default=VIEWPOINT_FILE,
                        help="JSON file the camera pose is loaded from and saved "
                             "to (default: %(default)s).")
    parser.add_argument("--reset-view", action="store_true",
                        help="Ignore the saved viewpoint this run (it is still "
                             "re-saved when you close the window).")
    parser.add_argument("--no-view", action="store_true",
                        help="Print statistics only; do not open the 3D window.")
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        raise SystemExit(f"File not found: {args.path}")

    points = load_point_cloud(args.path, num_features=args.features)
    summarize(points)

    if not args.no_view:
        visualize(points, color_by=args.color, show_axes=args.axes,
                  viewpoint_file=args.viewpoint, restore_view=not args.reset_view)


if __name__ == "__main__":
    main()
