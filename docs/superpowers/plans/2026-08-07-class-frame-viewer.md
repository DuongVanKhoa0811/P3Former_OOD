# Class-Frame Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** An interactive Open3D browser (`view_class_frames.py`) for the per-class `.ply` frames gathered by `pick_class_frames.py`, with key-driven frame/class navigation and semantic/highlight coloring.

**Architecture:** One script at the project root, plus one standalone test file. Non-GUI logic (PLY loading, class catalog/resolution, colorizers, stats table) is pure-function and tested against synthetic PLYs; the GUI layer (`FrameBrowser` + `run_viewer`) is thin glue over those functions, verified on real data via `--no-view` and a manual launch.

**Tech Stack:** Python 3.9 (`/usr/bin/python3`), numpy 2.0, open3d 0.18 (GUI only), matplotlib 3.9 (optional, gradient modes only — NOTE: `matplotlib.cm.get_cmap` was removed in 3.9, use `matplotlib.colormaps["viridis"]`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-07-class-frame-viewer-design.md`.
- Working dir: `/Users/anhlee/Downloads/Projects/DSO_Ocularis/OOD_OD_3D` — **not a git repo**, so plan tasks have no commit steps.
- Python 3.9 syntax only (no match, no `X | Y` type unions at runtime).
- `--no-view` and the no-arg listing must work without open3d installed (lazy import).
- Tests are plain asserts, run with `python3 test_view_class_frames.py`, no pytest.
- All color arrays handed to Open3D: shape (N, 3), float64, values in [0, 1], C-contiguous.
- Data layout ground truth: binary-little-endian PLY, 27 B/point, fields `x y z intensity device_id` (float32), `semantic` (uchar), `instance` (ushort), `isVisible` (uchar), `red green blue` (uchar).

---

### Task 1: PLY loader + synthetic-PLY test fixture

**Files:**
- Create: `test_view_class_frames.py` (fixture + loader tests; later tasks append to it)
- Create: `view_class_frames.py` (module docstring, imports, PLY reader)

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `read_ply_header(fh) -> (n_vertices, np.dtype, data_offset)`, `load_frame(path) -> np.ndarray` (structured, fields as in Global Constraints); test helpers `VERTEX` (packed dtype) and `make_ply(path, sem, rgb, z=None, intensity=None)` used by every later test.

- [x] **Step 1: Write the failing test file**

```python
#!/usr/bin/env python3
"""Tests for the non-GUI parts of view_class_frames.py (plain asserts, no pytest).

Run:  python3 test_view_class_frames.py
"""
import tempfile
from pathlib import Path

import numpy as np

import view_class_frames as m

# Packed on-disk record: 27 bytes, matching the DSO annotation .ply layout.
VERTEX = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("intensity", "<f4"), ("device_id", "<f4"),
    ("semantic", "u1"), ("instance", "<u2"), ("isVisible", "u1"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
])
assert VERTEX.itemsize == 27, "test dtype must be packed to 27 B like the real files"

HEADER = (
    "ply\n"
    "format binary_little_endian 1.0\n"
    "element vertex %d\n"
    "property float x\n"
    "property float y\n"
    "property float z\n"
    "property float intensity\n"
    "property float device_id\n"
    "property uchar semantic\n"
    "property ushort instance\n"
    "property uchar isVisible\n"
    "property uchar red\n"
    "property uchar green\n"
    "property uchar blue\n"
    "end_header\n"
)


def make_data(sem, rgb, z=None, intensity=None):
    """Structured in-memory frame; sem is a list of ids, rgb a per-point (r,g,b) list."""
    n = len(sem)
    arr = np.zeros(n, dtype=VERTEX)
    arr["x"] = np.arange(n, dtype=np.float32)
    arr["y"] = 1.0
    arr["z"] = np.linspace(0.0, 5.0, n) if z is None else z
    arr["intensity"] = np.arange(n, dtype=np.float32) if intensity is None else intensity
    arr["semantic"] = sem
    arr["red"] = [c[0] for c in rgb]
    arr["green"] = [c[1] for c in rgb]
    arr["blue"] = [c[2] for c in rgb]
    return arr


def make_ply(path, sem, rgb, z=None, intensity=None):
    arr = make_data(sem, rgb, z=z, intensity=intensity)
    with open(path, "wb") as fh:
        fh.write((HEADER % len(sem)).encode("ascii"))
        fh.write(arr.tobytes())
    return arr


def test_loader():
    sem = [3, 3, 8, 8, 8, 1]
    rgb = [(0, 0, 142)] * 2 + [(100, 60, 20)] * 3 + [(216, 82, 24)]
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "1_seq_000.ply"
        src = make_ply(p, sem, rgb)
        data = m.load_frame(p)
        assert data.shape[0] == 6
        for f in ("x", "y", "z", "intensity", "semantic", "red", "green", "blue"):
            assert f in data.dtype.names, f
            assert np.array_equal(data[f], src[f]), f
        # a truncated/padded file must be rejected, not silently mis-read
        with open(p, "ab") as fh:
            fh.write(b"\x00")
        try:
            m.load_frame(p)
            raise AssertionError("size mismatch not detected")
        except ValueError:
            pass


def main():
    tests = [test_loader]
    for fn in tests:
        fn()
        print("ok  %s" % fn.__name__)
    print("all tests passed")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run it to verify it fails**

Run: `cd /Users/anhlee/Downloads/Projects/DSO_Ocularis/OOD_OD_3D && python3 test_view_class_frames.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'view_class_frames'`

- [x] **Step 3: Create `view_class_frames.py` with the PLY reader**

Module head + reader, adapted verbatim from `analyze_seg_labels.py` (proven against this dataset), minus the field checks we don't need loosened — keep them:

```python
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
    import os
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
```

(Note: `import os` moves to the top-level import block in the real file.)

- [x] **Step 4: Run the test to verify it passes**

Run: `python3 test_view_class_frames.py`
Expected: `ok  test_loader` then `all tests passed`

---

### Task 2: Class catalog, CSV metadata, class resolution

**Files:**
- Modify: `view_class_frames.py` (append catalog section)
- Modify: `test_view_class_frames.py` (append tests, register in `main()`)

**Interfaces:**
- Consumes: nothing from Task 1 (independent section).
- Produces: `CLASS_NAMES` (dict id->name fallback), `discover_classes(frames_dir: Path) -> list[dict]` with keys `id:int, name:str, dirname:str, color:tuple|None, files:list[Path]` (rank-sorted); `load_csv_meta(csv_path) -> dict[int, dict(name, color)]`; `apply_meta(classes, meta) -> None`; `resolve_class(arg: str, classes) -> dict` (raises `ValueError` on miss/ambiguity); `build_names(classes, meta) -> dict[int,str]`; `render_class_list(classes) -> str`.

- [x] **Step 1: Append the failing tests**

```python
def test_catalog_and_resolution():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        frames = root / "class_frames"
        layout = {
            "003_car": ["1_seqA_000.ply", "2_seqB_001.ply", "10_seqC_002.ply"],
            "006_motorcycle": ["1_seqA_003.ply"],
            "007_bicycle": ["1_seqA_004.ply"],
            "008_class_08": ["1_seqA_005.ply"],
        }
        for d, files in layout.items():
            (frames / d).mkdir(parents=True)
            for f in files:
                (frames / d / f).touch()
        (frames / "009_class_09").mkdir()          # empty -> skipped
        (frames / "notes").mkdir()                  # not a class dir -> ignored

        classes = m.discover_classes(frames)
        assert [c["id"] for c in classes] == [3, 6, 7, 8]
        car = classes[0]
        assert car["name"] == "car" and car["dirname"] == "003_car"
        # rank order must be numeric: 1, 2, 10 (not 1, 10, 2)
        assert [f.name.split("_")[0] for f in car["files"]] == ["1", "2", "10"]

        csv_path = root / "class_frames.csv"
        csv_path.write_text(
            "class_id,name,color,rank,points,pct_of_frame,sequence,frame,path,copied_to\n"
            '3,car,"0,0,142",1,10,1.0,seqA,000,/x/000.ply,\n'
            '3,car,"0,0,142",2,9,1.0,seqB,001,/x/001.ply,\n'
            '8,class_08,"100,60,20",1,5,0.5,seqA,005,/x/005.ply,\n')
        meta = m.load_csv_meta(csv_path)
        assert meta[3]["color"] == (0, 0, 142) and meta[8]["name"] == "class_08"
        m.apply_meta(classes, meta)
        assert classes[0]["color"] == (0, 0, 142)
        assert classes[3]["color"] == (100, 60, 20)

        assert m.resolve_class("3", classes) is car          # by id
        assert m.resolve_class("car", classes) is car        # exact name
        assert m.resolve_class("003_car", classes) is car    # dirname
        assert m.resolve_class("class_08", classes)["id"] == 8
        assert m.resolve_class("moto", classes)["id"] == 6   # unique substring
        for bad in ("cycle", "zzz", "99"):                   # ambiguous / miss / bad id
            try:
                m.resolve_class(bad, classes)
                raise AssertionError("resolve_class(%r) should have raised" % bad)
            except ValueError as exc:
                if bad == "cycle":
                    assert "motorcycle" in str(exc) and "bicycle" in str(exc)

        names = m.build_names(classes, meta)
        assert names[3] == "car" and names[1] == "person"    # 1 from CLASS_NAMES fallback
        listing = m.render_class_list(classes)
        assert "003_car" in listing and "3" in listing.split()
```

Register in `main()`: `tests = [test_loader, test_catalog_and_resolution]`.

- [x] **Step 2: Run to verify it fails**

Run: `python3 test_view_class_frames.py`
Expected: FAIL with `AttributeError: module 'view_class_frames' has no attribute 'discover_classes'`

- [x] **Step 3: Append the implementation**

```python
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
```

- [x] **Step 4: Run to verify it passes**

Run: `python3 test_view_class_frames.py`
Expected: `ok  test_loader`, `ok  test_catalog_and_resolution`, `all tests passed`

---

### Task 3: Colorizers and per-frame stats table

**Files:**
- Modify: `view_class_frames.py` (append color + stats section)
- Modify: `test_view_class_frames.py` (append tests, register in `main()`)

**Interfaces:**
- Consumes: structured arrays as produced by `load_frame` / test `make_data`.
- Produces: `MODES = ("semantic", "highlight", "intensity", "height")`; `colorize(data, mode, target_id=None, target_rgb=None) -> np.ndarray (N,3) float64 C-contiguous in [0,1]`; `class_color_from_points(data, target_id) -> (r,g,b)|None`; `summarize_frame(data, names, target_id=None) -> str`; `GREY = 0.35`.

- [x] **Step 1: Append the failing tests**

```python
def _check_colors(c, n):
    assert c.shape == (n, 3) and c.dtype == np.float64
    assert c.flags["C_CONTIGUOUS"]
    assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0


def test_colorizers():
    sem = [3, 3, 8, 8, 8]
    rgb = [(0, 0, 142)] * 2 + [(100, 60, 20)] * 3
    data = make_data(sem, rgb)

    c = m.colorize(data, "semantic")
    _check_colors(c, 5)
    assert np.allclose(c[0], np.array([0, 0, 142]) / 255.0)
    assert np.allclose(c[4], np.array([100, 60, 20]) / 255.0)

    for target_rgb in ((0, 0, 142), None):       # explicit CSV colour, then derived
        c = m.colorize(data, "highlight", target_id=3, target_rgb=target_rgb)
        _check_colors(c, 5)
        assert np.allclose(c[:2], np.tile(np.array([0, 0, 142]) / 255.0, (2, 1)))
        assert np.allclose(c[2:], m.GREY)
    assert np.allclose(m.colorize(data, "highlight", target_id=99), m.GREY)

    assert m.class_color_from_points(data, 8) == (100, 60, 20)
    assert m.class_color_from_points(data, 99) is None

    for mode in ("intensity", "height"):
        _check_colors(m.colorize(data, mode), 5)
    flat = make_data(sem, rgb, z=np.zeros(5), intensity=np.zeros(5))
    for mode in ("intensity", "height"):          # constant column: no NaNs, all zero
        c = m.colorize(flat, mode)
        assert np.isfinite(c).all()


def test_summarize():
    sem = [3, 3, 8, 8, 8]
    rgb = [(0, 0, 142)] * 2 + [(100, 60, 20)] * 3
    data = make_data(sem, rgb)
    s = m.summarize_frame(data, {3: "car", 8: "class_08"}, target_id=3)
    assert "car" in s and "class_08" in s
    assert "60.0%" in s and "40.0%" in s
    car_line = [ln for ln in s.splitlines() if "car" in ln and "class_08" not in ln][0]
    assert car_line.rstrip().endswith("<<")
    # densest class first
    assert s.index("class_08") < s.index("car")
```

Register both in `main()`'s test list.

- [x] **Step 2: Run to verify it fails**

Run: `python3 test_view_class_frames.py`
Expected: FAIL with `AttributeError: ... no attribute 'colorize'`

- [x] **Step 3: Append the implementation**

```python
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
```

- [x] **Step 4: Run to verify it passes**

Run: `python3 test_view_class_frames.py`
Expected: all four `ok` lines, `all tests passed`

---

### Task 4: FrameBrowser, viewer window, CLI

**Files:**
- Modify: `view_class_frames.py` (append browser/viewer/CLI sections)

**Interfaces:**
- Consumes: everything above.
- Produces: `FrameBrowser(classes, names, class_idx, mode)` with `.print_stats()`, `.first_geometry(o3d)`, key handlers `next_frame/prev_frame/next_class/prev_class/cycle_mode/toggle_highlight` (all `fn(vis) -> False`); `run_viewer(browser, args)`; `parse_args(argv)`; `main(argv) -> int`.

No unit tests (GUI + argparse glue); verified in Task 5 against real data. Keep every method under ~20 lines.

- [x] **Step 1: Append the implementation**

```python
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
    if classes[class_idx]["id"] is not None:
        names[classes[class_idx]["id"]] = classes[class_idx]["name"]

    browser = FrameBrowser(classes, names, class_idx, args.color)
    if args.no_view:
        return browser.print_stats()
    run_viewer(browser, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

(Also: move Task 1's `import os` from inside `load_frame` up to the module import block.)

- [x] **Step 2: Re-run the full test suite**

Run: `python3 test_view_class_frames.py`
Expected: all `ok` lines, `all tests passed` (nothing GUI is imported at module scope, so tests still run without opening a window)

- [x] **Step 3: Syntax/entry sanity without GUI**

Run: `python3 view_class_frames.py --help`
Expected: usage text listing target, --frames-dir, --csv, --color, --point-size, --viewpoint, --reset-view, --no-view

---

### Task 5: Real-data verification and handoff

**Files:**
- Modify: none (verification only)

**Interfaces:**
- Consumes: the finished script + the real `class_frames/` tree and `class_frames.csv`.
- Produces: evidence the tool works; launch instructions for the user.

- [x] **Step 1: Class listing against real data**

Run: `python3 view_class_frames.py`
Expected: 28 rows (ids 0-28 range plus 255), names person/rider/car/... and class_NN, 5 frames each, colors filled from the CSV

- [x] **Step 2: Stats on a named class**

Run: `python3 view_class_frames.py car --no-view`
Expected: 5 frames, each with a per-class table; the `car` row marked `<<`; no warnings

- [x] **Step 3: Stats on an unnamed stuff class and by-id resolution**

Run: `python3 view_class_frames.py 8 --no-view && python3 view_class_frames.py class_20 --no-view`
Expected: same shape of output, `class_08` / `class_20` rows marked `<<`

- [x] **Step 4: Error paths**

Run: `python3 view_class_frames.py cycle; python3 view_class_frames.py zzz`
Expected: exit 2 each, "ambiguous" naming motorcycle+bicycle for the first, "no class matches" plus the class list for the second

- [x] **Step 5: Whole-tree sweep (every class loads)**

Run: `for d in class_frames/*/; do python3 view_class_frames.py "$(basename "$d")" --no-view >/dev/null || echo "FAIL $d"; done; echo sweep done`
Expected: only `sweep done` (no FAIL lines) — proves all 140 files parse

- [x] **Step 6: Hand off the interactive check**

Tell the user to run `python3 view_class_frames.py car` locally, orbit to a good angle, flip with N/]/C/H, close, and confirm the viewpoint JSON appeared.
