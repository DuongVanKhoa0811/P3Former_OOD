#!/usr/bin/env python3
"""Browse the per-class .ply frames gathered by pick_class_frames.py in 3D.

pick_class_frames.py --copy places up to N annotated LiDAR sweeps per semantic
class under

    class_frames/<NNN>_<name>/<rank>_<sequence>_<frame>.ply

Each sweep is a binary-little-endian PLY with a 27-byte record per point:
float x, y, z, intensity, device_id; uchar semantic; ushort instance;
uchar isVisible; uchar red, green, blue (the dataset's own label colour).

This script opens ONE persistent Open3D window and lets you flip through the
frames without ever leaving it:

    N / P   next / previous frame of the current class (rank order, wraps)
    ] / [   next / previous class (first frame of it)
    C       cycle colour mode: semantic -> highlight -> intensity -> height
    H       toggle highlight of the class the frame was picked for

Colour modes
    semantic   every point wears its own red/green/blue from the file
    highlight  the target class in its class colour, everything else dim grey
    intensity  viridis over the LiDAR return intensity
    height     viridis over z

The camera pose is restored from view_class_frames_viewpoint.json on start and
saved back when the window closes; while flipping frames the pose is kept, so
one good angle serves all frames. --reset-view ignores the saved pose once.

Usage
    python3 view_class_frames.py              # list the classes, then exit
    python3 view_class_frames.py car          # by name (or unique substring)
    python3 view_class_frames.py 3            # by semantic id
    python3 view_class_frames.py 003_car      # by directory name
    python3 view_class_frames.py some.ply     # one file, no class context
    python3 view_class_frames.py car --no-view    # per-frame stats only

Dependencies: numpy required; open3d required only to open the window;
matplotlib optional (gradient modes fall back to grey without it).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# PLY reading (same format handling as analyze_seg_labels.py)
# ---------------------------------------------------------------------------

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
                "file is %d B" % (expected, n, dtype.itemsize, offset, actual))
        data = np.fromfile(fh, dtype=dtype, count=n)
    if data.shape[0] != n:
        raise ValueError("short read: got %d of %d points" % (data.shape[0], n))
    for field in ("x", "y", "z", "semantic", "red", "green", "blue"):
        if field not in data.dtype.names:
            raise ValueError("missing %r property" % field)
    return data


# ---------------------------------------------------------------------------
# Class catalog
# ---------------------------------------------------------------------------

# Fallback names for ids the CSV may not cover (from analyze_seg_labels.py).
CLASS_NAMES = {
    0: "unlabeled", 1: "person", 2: "rider", 3: "car", 4: "truck", 5: "bus",
    6: "motorcycle", 7: "bicycle", 27: "obscurant", 255: "unlabelled",
}

_CLASS_DIR_RE = re.compile(r"^(\d{1,3})_(.+)$")


def _rank_key(path):
    """Sort '<rank>_<seq>_<frame>.ply' numerically by rank, then by name."""
    head = path.name.split("_", 1)[0]
    return (int(head), path.name) if head.isdigit() else (10 ** 9, path.name)


def discover_classes(frames_dir):
    """One entry per '<NNN>_<name>' directory that holds at least one .ply."""
    classes = []
    for d in sorted(Path(frames_dir).iterdir()):
        if not d.is_dir():
            continue
        mt = _CLASS_DIR_RE.match(d.name)
        if not mt:
            continue
        files = sorted(d.glob("*.ply"), key=_rank_key)
        if not files:
            print("warning: %s has no .ply files, skipping" % d.name, file=sys.stderr)
            continue
        classes.append({"id": int(mt.group(1)), "name": mt.group(2),
                        "dirname": d.name, "color": None, "files": files})
    return classes


def load_csv_meta(csv_path):
    """class_frames.csv -> {class_id: {'name': ..., 'color': (r,g,b) | None}}."""
    meta = {}
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            cid = int(r["class_id"])
            if cid in meta:
                continue
            color = None
            raw = (r.get("color") or "").strip()
            if raw:
                try:
                    color = tuple(int(c) for c in raw.split(","))
                except ValueError:
                    color = None
            meta[cid] = {"name": r["name"], "color": color}
    return meta


def apply_meta(classes, meta):
    for c in classes:
        mm = meta.get(c["id"])
        if mm:
            c["name"] = mm["name"] or c["name"]
            c["color"] = mm["color"]


def build_names(classes, meta):
    names = dict(CLASS_NAMES)
    for cid, mm in meta.items():
        names[cid] = mm["name"]
    for c in classes:
        names[c["id"]] = c["name"]
    return names


def resolve_class(arg, classes):
    """id, exact name/dirname, or unique substring -> class entry; ValueError otherwise."""
    if arg.isdigit():
        cid = int(arg)
        for c in classes:
            if c["id"] == cid:
                return c
        raise ValueError("no class with id %d" % cid)
    low = arg.lower()
    for c in classes:
        if c["dirname"].lower() == low or c["name"].lower() == low:
            return c
    hits = [c for c in classes
            if low in c["name"].lower() or low in c["dirname"].lower()]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError("no class matches %r" % arg)
    raise ValueError("%r is ambiguous: %s" % (arg, ", ".join(c["dirname"] for c in hits)))


def render_class_list(classes):
    lines = ["%4s  %-16s %7s  %s" % ("id", "name", "frames", "color"),
             "%4s  %-16s %7s  %s" % ("-" * 4, "-" * 16, "-" * 7, "-" * 11)]
    for c in classes:
        color = "%d,%d,%d" % c["color"] if c["color"] else "-"
        lines.append("%4d  %-16s %7d  %s" % (c["id"], c["name"], len(c["files"]), color))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Colour modes
# ---------------------------------------------------------------------------

MODES = ("semantic", "highlight", "intensity", "height")
GREY = 0.35   # brightness of non-target points in highlight mode


def colors_semantic(data):
    """Each point's own red/green/blue from the file -> (N, 3) in [0, 1]."""
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1)
    return np.ascontiguousarray(rgb / 255.0, dtype=np.float64)


def class_color_from_points(data, target_id):
    """Most common RGB among the target class's points, or None if absent."""
    mask = data["semantic"] == target_id
    if not mask.any():
        return None
    packed = ((data["red"][mask].astype(np.int64) << 16)
              | (data["green"][mask].astype(np.int64) << 8)
              | data["blue"][mask].astype(np.int64))
    vals, counts = np.unique(packed, return_counts=True)
    v = int(vals[np.argmax(counts)])
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)


def colors_highlight(data, target_id, target_rgb=None):
    """Target class in its colour, everything else a flat dim grey."""
    colors = np.full((data.shape[0], 3), GREY, dtype=np.float64)
    if target_rgb is None:
        target_rgb = class_color_from_points(data, target_id)
    if target_rgb is not None:
        colors[data["semantic"] == target_id] = np.asarray(target_rgb, dtype=np.float64) / 255.0
    return np.ascontiguousarray(colors)


def _normalize(values):
    """Robustly scale to [0, 1] (2nd-98th percentile window)."""
    lo, hi = np.percentile(values, 2), np.percentile(values, 98)
    if hi <= lo:
        return np.zeros(values.shape[0], dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def colors_gradient(values):
    """Viridis over a 1-D column; grey ramp if matplotlib is unavailable."""
    t = _normalize(np.asarray(values, dtype=np.float64))
    try:
        from matplotlib import colormaps           # cm.get_cmap is gone in mpl 3.9
        rgb = colormaps["viridis"](t)[:, :3]
    except ImportError:
        rgb = np.stack([t, t, t], axis=1)
    return np.ascontiguousarray(rgb, dtype=np.float64)


def colorize(data, mode, target_id=None, target_rgb=None):
    if mode == "highlight" and target_id is not None:
        return colors_highlight(data, target_id, target_rgb)
    if mode == "intensity":
        return colors_gradient(data["intensity"])
    if mode == "height":
        return colors_gradient(data["z"])
    return colors_semantic(data)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def summarize_frame(data, names, target_id=None):
    """Per-class breakdown of one frame, densest first, target marked '<<'."""
    sem = np.asarray(data["semantic"], dtype=np.intp)
    total = sem.shape[0]
    ids, counts = np.unique(sem, return_counts=True)
    lines = ["  %3s  %-16s %11s  %6s" % ("id", "name", "points", "%")]
    for i in np.argsort(-counts):
        cid, n = int(ids[i]), int(counts[i])
        mark = "  <<" if cid == target_id else ""
        lines.append("  %3d  %-16s %11s  %5.1f%%%s" % (
            cid, names.get(cid, "class_%02d" % cid),
            "{:,}".format(n), 100.0 * n / total, mark))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frame browser (GUI state)
# ---------------------------------------------------------------------------

class FrameBrowser:
    """Current class/frame/mode + the Open3D geometry the window displays."""

    def __init__(self, classes, names, class_idx=0, mode="semantic"):
        self.classes = classes
        self.names = names
        self.ci = class_idx
        self.fi = 0
        self.mode = mode
        self.last_plain = "semantic" if mode == "highlight" else mode
        self.step = 1          # direction N/P moved last, for skipping bad files
        self.data = None       # structured array of the current frame
        self.pcd = None        # o3d.geometry.PointCloud, created in first_geometry

    # -- state helpers ------------------------------------------------------
    @property
    def cls(self):
        return self.classes[self.ci]

    @property
    def path(self):
        return self.cls["files"][self.fi]

    def _load_current(self):
        """Load self.path, skipping unreadable files in the direction of travel."""
        for _ in range(len(self.cls["files"])):
            try:
                return load_frame(self.path)
            except Exception as exc:
                print("skipping %s: %s" % (self.path.name, exc), file=sys.stderr)
                self.fi = (self.fi + self.step) % len(self.cls["files"])
        print("no readable frames in %s" % self.cls["dirname"], file=sys.stderr)
        return None

    def _announce(self):
        c = self.cls
        head = "[%s  frame %d/%d]  %s" % (
            c["dirname"], self.fi + 1, len(c["files"]), self.path.name)
        print("\n%s" % head)
        print("  %s points   mode=%s" % ("{:,}".format(self.data.shape[0]), self.mode))
        print(summarize_frame(self.data, self.names, target_id=c["id"]))

    def _colors(self):
        return colorize(self.data, self.mode,
                        target_id=self.cls["id"], target_rgb=self.cls["color"])

    # -- geometry -----------------------------------------------------------
    def _fill(self):
        import open3d as o3d
        xyz = np.stack([self.data["x"], self.data["y"], self.data["z"]], axis=1)
        self.pcd.points = o3d.utility.Vector3dVector(
            np.ascontiguousarray(xyz, dtype=np.float64))
        self.pcd.colors = o3d.utility.Vector3dVector(self._colors())

    def first_geometry(self, o3d):
        self.pcd = o3d.geometry.PointCloud()
        self.data = self._load_current()
        if self.data is None:
            raise SystemExit(1)
        self._fill()
        self._announce()
        return self.pcd

    def _switch(self, vis):
        """Swap the point buffer, keeping the camera pose explicitly."""
        ctl = vis.get_view_control()
        cam = ctl.convert_to_pinhole_camera_parameters()
        data = self._load_current()
        if data is not None:
            self.data = data
            self._fill()
            self._announce()
            vis.update_geometry(self.pcd)
        ctl.convert_from_pinhole_camera_parameters(cam, True)
        vis.update_renderer()
        return False

    def _recolor(self, vis):
        import open3d as o3d
        self.pcd.colors = o3d.utility.Vector3dVector(self._colors())
        vis.update_geometry(self.pcd)
        vis.update_renderer()
        print("mode = %s" % self.mode)
        return False

    # -- key handlers (each: fn(vis) -> False) ------------------------------
    def next_frame(self, vis):
        self.step, self.fi = 1, (self.fi + 1) % len(self.cls["files"])
        return self._switch(vis)

    def prev_frame(self, vis):
        self.step, self.fi = -1, (self.fi - 1) % len(self.cls["files"])
        return self._switch(vis)

    def next_class(self, vis):
        self.ci = (self.ci + 1) % len(self.classes)
        self.step, self.fi = 1, 0
        return self._switch(vis)

    def prev_class(self, vis):
        self.ci = (self.ci - 1) % len(self.classes)
        self.step, self.fi = 1, 0
        return self._switch(vis)

    def cycle_mode(self, vis):
        i = MODES.index(self.mode)
        while True:
            i = (i + 1) % len(MODES)
            if MODES[i] == "highlight" and self.cls["id"] is None:
                continue                       # single-file mode: no target class
            break
        self.mode = MODES[i]
        if self.mode != "highlight":
            self.last_plain = self.mode
        return self._recolor(vis)

    def toggle_highlight(self, vis):
        if self.cls["id"] is None:
            print("highlight needs a class context (opened via bare .ply path)")
            return False
        if self.mode == "highlight":
            self.mode = self.last_plain
        else:
            self.last_plain = self.mode
            self.mode = "highlight"
        return self._recolor(vis)

    # -- no-GUI path ---------------------------------------------------------
    def print_stats(self):
        """--no-view: stats for every frame of the current class."""
        for self.fi in range(len(self.cls["files"])):
            try:
                self.data = load_frame(self.path)
            except Exception as exc:
                print("skipping %s: %s" % (self.path.name, exc), file=sys.stderr)
                continue
            self._announce()
        return 0


# ---------------------------------------------------------------------------
# Viewer window
# ---------------------------------------------------------------------------

WINDOW_W, WINDOW_H = 1280, 800


def _restore_viewpoint(o3d, view_control, viewpoint_file):
    if not viewpoint_file or not Path(viewpoint_file).is_file():
        return
    try:
        param = o3d.io.read_pinhole_camera_parameters(str(viewpoint_file))
        ok = view_control.convert_from_pinhole_camera_parameters(param, True)
        print("Restored viewpoint from %s" % viewpoint_file if ok else
              "[viewpoint] saved file did not apply cleanly; using default view")
    except Exception as exc:
        print("[viewpoint] could not restore (%s); using default view" % exc)


def _save_viewpoint(o3d, view_control, viewpoint_file):
    if not viewpoint_file:
        return
    try:
        param = view_control.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(str(viewpoint_file), param)
        print("Saved viewpoint to %s" % viewpoint_file)
    except Exception as exc:
        print("[viewpoint] could not save (%s)" % exc)


def run_viewer(browser, args):
    try:
        import open3d as o3d
    except ImportError:
        raise SystemExit("open3d is required for the 3D window (pip install open3d); "
                         "use --no-view for stats only")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="DSO class frames -- N/P frame, ]/[ class, C colour, H highlight",
        width=WINDOW_W, height=WINDOW_H)
    vis.add_geometry(browser.first_geometry(o3d))
    vis.get_render_option().point_size = args.point_size

    if not args.reset_view:
        # after add_geometry, which auto-fits (and would override) the view
        _restore_viewpoint(o3d, vis.get_view_control(), args.viewpoint)

    for key, fn in ((ord("N"), browser.next_frame), (ord("P"), browser.prev_frame),
                    (ord("]"), browser.next_class), (ord("["), browser.prev_class),
                    (ord("C"), browser.cycle_mode), (ord("H"), browser.toggle_highlight)):
        vis.register_key_callback(key, fn)

    print("\nKeys: N/P frame, ]/[ class, C colour mode, H highlight, "
          "mouse to orbit; close the window to save the viewpoint.")
    vis.run()
    _save_viewpoint(o3d, vis.get_view_control(), args.viewpoint)
    vis.destroy_window()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", nargs="?", default=None,
                   help="class id, name, unique name substring, class directory, "
                        "or a .ply path; omit to list the classes")
    p.add_argument("--frames-dir", type=Path, default=here / "class_frames",
                   help="directory of <NNN>_<name> class folders (default: %(default)s)")
    p.add_argument("--csv", type=Path, default=here / "class_frames.csv",
                   help="class_frames.csv for names/colours (default: %(default)s)")
    p.add_argument("--color", choices=MODES, default="semantic",
                   help="initial colour mode (default: %(default)s)")
    p.add_argument("--point-size", type=float, default=2.0,
                   help="rendered point size in pixels (default: %(default)s)")
    p.add_argument("--viewpoint", type=Path,
                   default=here / "view_class_frames_viewpoint.json",
                   help="camera pose file (default: %(default)s)")
    p.add_argument("--reset-view", action="store_true",
                   help="ignore the saved viewpoint this run")
    p.add_argument("--no-view", action="store_true",
                   help="print per-frame stats only; no window")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    meta = {}
    if args.csv.is_file():
        meta = load_csv_meta(args.csv)
    else:
        print("note: %s not found; names come from directory names, highlight "
              "colours from the points themselves" % args.csv, file=sys.stderr)

    if args.target and args.target.lower().endswith(".ply"):
        path = Path(args.target)
        if not path.is_file():
            raise SystemExit("file not found: %s" % path)
        classes = [{"id": None, "name": path.stem, "dirname": path.name,
                    "color": None, "files": [path]}]
        class_idx = 0
    else:
        if not args.frames_dir.is_dir():
            raise SystemExit("%s is not a directory -- run pick_class_frames.py "
                             "--copy first, or pass --frames-dir" % args.frames_dir)
        classes = discover_classes(args.frames_dir)
        if not classes:
            raise SystemExit("no <NNN>_<name> class directories under %s" % args.frames_dir)
        apply_meta(classes, meta)
        if args.target is None:
            print(render_class_list(classes))
            return 0
        try:
            class_idx = classes.index(resolve_class(args.target, classes))
        except ValueError as exc:
            print("error: %s\n" % exc, file=sys.stderr)
            print(render_class_list(classes), file=sys.stderr)
            return 2

    names = build_names(classes, meta)

    browser = FrameBrowser(classes, names, class_idx, args.color)
    if args.no_view:
        return browser.print_stats()
    run_viewer(browser, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
