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
        assert "003_car" in listing or "car" in listing
        assert "3" in listing.split()


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
    for mode in ("intensity", "height"):          # constant column: no NaNs
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


def main():
    tests = [test_loader, test_catalog_and_resolution, test_colorizers, test_summarize]
    for fn in tests:
        fn()
        print("ok  %s" % fn.__name__)
    print("all tests passed")


if __name__ == "__main__":
    main()
