# Divided Mass and OOD–Class Resemblance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain why some two-group splits beat the flat OOD scores, using the divided mass measured from the logit dumps. Then test, in P3Former's penultimate feature space, whether the best split puts the classes that OOD points resemble on one side.

**Architecture:** One extended sweep tool, five new offline tools and one new GPU tool, all under `tools/`. They write TSV, markdown and figures under `work_dirs/p3former_2xb1_3x_dso_ood_dump/`.
- **Part 1** reuses the logit dumps and the sweep machinery of `tools/sweep_bipartitions.py` to histogram the divided mass m = min(P_A, P_B) of 503 splits.
- **Part 2** samples per-point `pe_features` with a runtime wrapper around `_P3FormerHead.init_inputs`, with no model change. It then measures kNN neighbourhood shares against a class-balanced bank of the same split's ID points.

**Tech Stack:** Python 3.8, numpy 1.24, torch 1.10.1 (CUDA; TF32 off for the analyses), scipy 1.10 (`rankdata`), scikit-learn 1.3.2 (PCA, t-SNE, metrics in tests), matplotlib 3.5, mmengine 0.7.4 / mmdet3d 1.1.0 (extraction only).

**Spec:** `docs/superpowers/specs/2026-09-29-divided-mass-resemblance-design.md`

## Global Constraints

- **Branch and worktree.** Work on branch `ood-baselines/grouping` in this checkout (`/home/khoadv/projects/OOD_PanSeg_3D/P3Former_OOD`). Do not use a worktree: `data/`, `checkpoint/` and `work_dirs/` are untracked local resources.
- **Interpreter.** `ENVPY=/home/khoadv/miniconda3/envs/p3former/bin/python`; bare `python` is the wrong env. Run everything from the repo root.
- **Python 3.8 syntax only.** No `X | Y` unions and no `list[str]`.
- **No model or config changes.** Nothing under `p3former/`, `configs/`, `datasets/` or `evaluation/` is modified.
- **Class names** come from `sweep_bipartitions.CLASSES` (24 DSO train ids; the ignore id is 24). OOD ground truth is raw 17 (Stop) or 28 (Others), exactly as `_OODPointMetric`.
- **Improvement**: mean ΔAUROC + mean ΔAP − mean ΔFPR@95 over `group_msp`, `group_energy` and `group_entropy`. This is `summarise(rows, exclude={'maxlogit', 'odin'})`, family `group`, then `_improvement(...)`. Robust means improvement > 0 on all three sets.
- **Divided-mass grid**: δ ∈ (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3); headline δ = 0.05. Bins run 0, then log-spaced at 200 per decade from 1e-15 to 0.5, with every δ an exact edge.
- **Check tolerances**: histogram metrics against the sweep logs within 0.1 (AUROC, AP) and 0.5 (FPR@95), in percentage points; the same split in the random and the single-class sweep within 0.02. A check beyond tolerance is reported as FAIL and never hidden.
- **TF32**: every GPU analysis (divided mass, kNN) turns TF32 off through `sweep_bipartitions.require_torch()`. The feature extraction keeps the TF32 setting `test.py` uses, so its logits match the dumps.
- **Storage and jobs**: large outputs go to `/mnt/sandisk/khoadv/P3Former_OOD/p3former_2xb1_3x_dso_ood_dump/` and are symlinked into `work_dirs/`. Jobs longer than a few minutes run detached: `nohup setsid bash -c "<cmd> > <log> 2>&1" < /dev/null > /dev/null 2>&1 &`. Check `nvidia-smi` first, run one GPU job per GPU, and allow ~25 GB per extraction.
- **Figure style**: serif Times text; `pdf.fonttype` = 42; Okabe–Ito gain `#0072B2` and drop `#D55E00`; bubble area ∝ |improvement|; each figure written as PDF plus a 300-dpi PNG.
- **Tests** follow the repo convention: pytest functions, an `if __name__ == '__main__':` runner printing `... passed`, and `sys.path` inserts of the repo root and `tools/`. `$ENVPY -m pytest -q tests` must pass after every task.
- **Commits**: one per task, staging explicit paths only (never `git add -A`), with messages ending in `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never push.

## Review Focus

Five realistic failure modes that no task's main-line test exercises. Each line names the input and the expected behaviour; the test that pins it is added to the owning task.

1. **Frames without OOD points, and classes absent from a frame**, are routine in the test split. Sampling must skip the empty strata without crashing, and the weights must still sum to the stratum sizes. Pinned in Task 4, `test_sample_frame`.
2. **Zero divided counts**: at δ = 0.3 many single-class splits divide no ID point. The expected results are selectivity `inf`, a finite log ratio and precision 100. The bubble charts draw the point on the axis floor with an open marker, and Spearman skips NaN. Pinned in Task 2 `test_zero_divided_counts` and Task 3 `test_figures_are_written`.
3. **Running `divided_mass.py` before the single-class sweeps, or on sweeps with different seeds**, must fail. It raises a `FileNotFoundError` naming `sweep_bipartitions.py`, or a `ValueError` naming `--seed`, never a `KeyError` from deep inside. Pinned in Task 2, `test_missing_inputs_are_explained`.
4. **`--subsets` without `--out-dir`** would overwrite the random sweep's `bipartitions/`, so the CLI refuses it. Pinned in Task 1, `test_subsets_requires_out_dir`.
5. **`--check-dump` must fail loudly.** A frame missing from the dump or a label mismatch raises, never a silent skip. A non-empty `--out-dir` is refused. Pinned in Task 4, `test_dump_comparison` and `test_out_dir_must_be_empty`.

## File Structure

| File | Responsibility |
| --- | --- |
| `tools/sweep_bipartitions.py` (modify) | `--subsets` explicit split lists; `prefetch_frames` made public |
| `tools/divided_mass.py` (create) | Part 1: histograms of m and u, divided statistics, joins with the sweep logs, checks, robust splits, Spearman, TSV and markdown I/O helpers |
| `tools/plot_divided_mass.py` (create) | Part 1 figures and the shared bubble-chart and style helpers |
| `tools/extract_point_features.py` (create) | Part 2a: GPU pass writing stratified per-frame feature samples |
| `tools/ood_class_resemblance.py` (create) | Part 2b: kNN profiles, per-split measures, alignment ρ |
| `tools/plot_ood_class_resemblance.py` (create) | Part 2c figures |
| `tools/plot_feature_tsne.py` (create) | Part 2d t-SNE figure and the fixed `CLASS_COLOURS` |
| `tests/test_sweep_bipartitions.py` (modify) | `--subsets` tests |
| `tests/test_divided_mass.py`, `tests/test_plot_divided_mass.py`, `tests/test_extract_point_features.py`, `tests/test_ood_class_resemblance.py`, `tests/test_plot_ood_class_resemblance.py`, `tests/test_plot_feature_tsne.py` (create) | Unit and smoke tests |
| `DOCs.md`, `.claude/CLAUDE.md`, `.claude/rules/offline-tools.md`, `.claude/rules/testing.md` (modify) | Results log, progress log, tool list, test count |

---

### Task 1: `--subsets` in the bipartition sweep

**Files:**
- Modify: `tools/sweep_bipartitions.py`
- Test: `tests/test_sweep_bipartitions.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `singleton_subsets() -> List[Tuple[int, ...]]`: `[(0,), ..., (23,)]`.
  - `load_subsets(spec: str) -> List[Tuple[int, ...]]`: canonical subsets. `spec` is `'singletons'` or a JSON path whose items are id lists or `{"A": [...]}` dicts. Raises `ValueError` on duplicates, an invalid split or an empty list.
  - `sweep(..., subsets=None)`: scores the explicit list when given.
  - `prefetch_frames(files, depth=8)`: the old `_prefetch`, renamed.
  - CLI `--subsets SPEC`, which requires `--out-dir`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_sweep_bipartitions.py`, adding `import json` and `import subprocess` to its imports:

```python
def test_load_subsets():
    assert sb.load_subsets('singletons') == [(c, ) for c in range(sb.NUM_CLASSES)]
    assert sb.singleton_subsets() == sb.load_subsets('singletons')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'subsets.json')
        with open(path, 'w') as fh:  # id lists and partitions.json items
            json.dump([[3], list(range(5, 24)), {'A': [16], 'B': []}], fh)
        assert sb.load_subsets(path) == [(3, ), (0, 1, 2, 3, 4), (16, )]
        for bad in ([[3], [3]], [[]], [list(range(24))], []):
            with open(path, 'w') as fh:
                json.dump(bad, fh)
            try:
                sb.load_subsets(path)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{bad} accepted')
    print('test_load_subsets passed')


def test_sweep_explicit_subsets():
    with tempfile.TemporaryDirectory() as tmp:
        dump_dir = os.path.join(tmp, 'logits')
        os.makedirs(dump_dir)
        _write_dump(dump_dir)
        kwargs = dict(bins=2**14, workers=1, chunk=4, top=2)
        singles = sb.sweep(dump_dir, os.path.join(tmp, 'singletons'),
                           subsets=sb.singleton_subsets(), **kwargs)
        assert set(singles) == set(sb.FLAT_KEYS) | {
            f's{c}_{k}' for c in range(sb.NUM_CLASSES) for k in sb.HIER_KEYS}
        log = os.path.join(tmp, 'singletons', 'bipartitions.log')
        assert offline_sweep_logs([log]) == [log]  # still a sweep log
        with open(log) as fh:
            assert '24 explicit two-group partitions' in fh.read()
        # a split scored in another block agrees with its single-class row
        mixed = sb.sweep(dump_dir, os.path.join(tmp, 'mixed'),
                         subsets=[(0, 1, 2, 3, 4), (7, ), (3, )], **kwargs)
        for c in (3, 7):
            for key in sb.HIER_KEYS:
                for k in ('auroc', 'ap', 'fpr95'):
                    assert abs(mixed[f's{c}_{key}'][k]
                               - singles[f's{c}_{key}'][k]) < 2e-3, (c, key, k)
    print('test_sweep_explicit_subsets passed')


def test_subsets_requires_out_dir():
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [sys.executable,
             os.path.join(_REPO_ROOT, 'tools', 'sweep_bipartitions.py'), tmp,
             '--subsets', 'singletons'],
            capture_output=True, text=True, cwd=_REPO_ROOT)
        assert proc.returncode == 2 and '--out-dir' in proc.stderr, proc.stderr
    print('test_subsets_requires_out_dir passed')
```

Register them in the `__main__` block before `print('ALL TESTS PASSED')`:

```python
    test_load_subsets()
    test_sweep_explicit_subsets()
    test_subsets_requires_out_dir()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$ENVPY -m pytest -q tests/test_sweep_bipartitions.py`
Expected: FAIL with `AttributeError: module 'sweep_bipartitions' has no attribute 'load_subsets'`.

- [ ] **Step 3: Implement.** In `tools/sweep_bipartitions.py`:

(a) Rename `def _prefetch(files, depth=8):` to `def prefetch_frames(files, depth=8):` and update its two call sites in `_torch_passes`:
`enumerate(_prefetch(files))` → `enumerate(prefetch_frames(files))` (both occurrences).

(b) After `sample_bipartitions`, add:

```python
def singleton_subsets():
    """The 24 single-class splits {c} | rest, as canonical subsets."""
    return [canonical([c]) for c in range(NUM_CLASSES)]


def load_subsets(spec):
    """Explicit bipartitions for ``--subsets``: ``'singletons'``, or the path
    to a JSON list whose items are class-id lists or ``{"A": [...]}`` dicts
    (the ``partitions.json`` a sweep writes). Canonical subsets in the given
    order; a split listed twice is an error."""
    if spec == 'singletons':
        return singleton_subsets()
    with open(spec) as fh:
        items = json.load(fh)
    subsets, seen = [], set()
    for item in items:
        subset = canonical(item['A'] if isinstance(item, dict) else item)
        if subset in seen:
            raise ValueError(f'{spec}: partition {partition_name(subset)} '
                             'is listed twice')
        seen.add(subset)
        subsets.append(subset)
    if not subsets:
        raise ValueError(f'{spec}: no partitions')
    return subsets
```

(c) In `sweep`, change the signature, docstring and split selection:

```python
def sweep(dump_dirs, out_dir, num=500, seed=0, bins=2**16, workers=8,
          chunk=64, top=10, rank_exclude=('odin', ), backend='numpy',
          device='cuda:0', bin_eps=BIN_EPS, subsets=None):
    """Score ``num`` random bipartitions -- or the explicit ``subsets``, when
    given (``num`` and ``seed`` are then unused) -- over the frames of one
    or several dump directories (evaluated together as one split)."""
```

and replace the block from `subsets = sample_bipartitions(num, seed)` down to its `print(...)` with:

```python
    if subsets is None:
        subsets = sample_bipartitions(num, seed)
        what = f'{num} two-group partitions (seed {seed})'
    else:
        subsets = [canonical(s) for s in subsets]
        what = f'{len(subsets)} explicit two-group partitions'
    names = [partition_name(s) for s in subsets]
    a_mask = subsets_to_mask(subsets)
    sizes = np.bincount(a_mask.sum(axis=1), minlength=NUM_CLASSES // 2 + 1)
    print(f'{len(files)} frames in {dump_dir}; {what}, smaller-group sizes '
          f'1..{NUM_CLASSES // 2}: ' + ' '.join(str(n) for n in sizes[1:]))
```

and pass `what` to `write_log`:
`write_log(log_path, rows, n_id, n_ood, out_dir, dump_dir, what, bins, bin_eps)`.

(d) `write_log`: replace the `num, seed` parameters with `what` and the first comment line with:

```python
def write_log(path, rows, n_id, n_ood, out_dir, dump_dir, what, bins,
              bin_eps):
    with open(path, 'w') as fh:
        fh.write(f"work_dir = '{out_dir}'\n")
        fh.write(f'# offline bipartition sweep: {what} of the {NUM_CLASSES} '
                 f'classes, scored from {dump_dir} with {bins} histogram bins '
                 + (f'log-spaced towards both ends (eps {bin_eps:g})'
                    if bin_eps > 0 else 'of equal width') + '\n')
```

(the rest of `write_log` unchanged; the header text of a random sweep is identical to before).

(e) `print_ranking`: guard the vehicle-vs-rest lines:

```python
    reference = partition_name(canonical(REFERENCE))
    if reference in composition:  # always there in a random sweep
        for key in ('group_msp', 'gn_msp'):
            m = rows[f'{reference}_{key}']
            print(f'reference {reference} (vehicle vs rest) {key}: '
                  f'{m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}')
```

(f) CLI: after the `--seed` argument add

```python
    ap.add_argument('--subsets', default=None, metavar='SPEC',
                    help="score an explicit list instead of --num random "
                    "partitions: 'singletons' (the 24 single-class splits) "
                    'or a JSON list of class-id lists (a partitions.json '
                    'works too); needs --out-dir')
```

and in `main()`, right after `args = ap.parse_args()`:

```python
    if args.subsets and args.out_dir is None:
        ap.error("--subsets needs --out-dir: the default is the random "
                 "sweep's directory, which it would overwrite")
```

and pass `subsets=load_subsets(args.subsets) if args.subsets else None` in the `sweep(...)` call.

(g) Module docstring: after the paragraph beginning "Several dump directories are evaluated as one split", insert:

```
``--subsets`` replaces the random sample with an explicit list:
``singletons`` (the 24 single-class splits {c} | rest, all of which
tools/divided_mass.py needs) or a JSON file of class-id lists (a sweep's
``partitions.json`` works too). ``--out-dir`` is then required, so that the
random sweep's outputs are never overwritten.
```

and add to its run commands:

```
    # the 24 single-class splits
    python tools/sweep_bipartitions.py --backend torch --device cuda:0 \\
        work_dirs/p3former_2xb1_3x_dso_ood_dump/logits --subsets singletons \\
        --out-dir work_dirs/p3former_2xb1_3x_dso_ood_dump/singletons
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$ENVPY -m pytest -q tests/test_sweep_bipartitions.py && $ENVPY tests/test_sweep_bipartitions.py`
Expected: every test passes, ending in `ALL TESTS PASSED`.

- [ ] **Step 5: Full suite, then commit**

Run: `$ENVPY -m pytest -q tests` — expected: all pass.

```bash
git add tools/sweep_bipartitions.py tests/test_sweep_bipartitions.py
git commit -m "$(cat <<'EOF'
Score explicit split lists in the bipartition sweep (--subsets)

'singletons' scores the 24 single-class splits, of which the random
sweep sampled only 21; a JSON list works too. --out-dir is required with
--subsets so the random sweep's outputs are never overwritten.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `tools/divided_mass.py`

**Files:**
- Create: `tools/divided_mass.py`
- Test: `tests/test_divided_mass.py`

**Interfaces:**
- Consumes (Task 1): `sb.singleton_subsets`, `sb.prefetch_frames`, `sb.require_torch`, `sb.torch_counts`, `sb.load_frame`, `sb.canonical`, `sb.partition_name`, `sb.subsets_to_mask`, `sb.CLASSES`, `sb.sweep` (tests); `summarize_hierarchy_ablation.parse_logs`, `summarise`, `_improvement`; `evaluation.functional.ood_eval.metrics_from_histograms`.
- Produces (used by Tasks 3, 5, 6, 7):
  - Constants: `ROOT`, `SETS` (keys `cetran`, `test`, `test_cetran`, each holding `label`), `DELTAS`, `HEADLINE`, `STATISTICS`, `TARGETS`.
  - Helpers: `delta_key(d) -> str` (`'0.05'`), `read_tsv(path) -> List[OrderedDict]`, `write_tsv(path, rows, header=None)`, `md_table(header, rows) -> List[str]`, `fmt(v, digits=2) -> str`, `pct(v) -> str`, `share(n, total)`, `ratio(a, b)`, `spearman(x, y) -> (rho, n)`, `resolve_sets(root)`, `default_subsets(sets)`, `run(sets, subsets, out_dir, backend, device, chunk, deltas) -> dict(tables, checks, robust)`.
  - `<set>.tsv` columns:
    - `name, size_A, group_A, n_id, n_ood`;
    - then for each δ key `k`: `ood_div_n@k, id_div_n@k, ood_div@k, id_div@k, precision@k, ood_retention@k, id_retention@k, selectivity@k, log_ratio@k`. The retentions are divided / flat-uncertain in %, and selectivity is their ratio;
    - then `delta95, id_div@delta95, gmsp_auroc, gmsp_ap, gmsp_fpr95, d_auroc, d_ap, d_fpr95, improvement, hist_auroc, hist_ap, hist_fpr95, robust`.
  - `flat.tsv` columns: `set, n_id, n_ood`, per δ `ood_unc@k, id_unc@k, precision@k`, then `u95, id_unc@u95, msp_auroc, msp_ap, msp_fpr95, hist_auroc, hist_ap, hist_fpr95`.
  - `rho.tsv` columns: `set, population, delta, statistic, target, rho, n`.
  - `robust.tsv` columns: `name, size_A, group_A, improvement_<set>..., worst`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_divided_mass.py`:

```python
"""Tests for tools/divided_mass.py.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_divided_mass.py
"""
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import divided_mass as dm  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402

RNG = np.random.RandomState(0)


def _logits(n, scale=4.0):
    return (scale * RNG.randn(n, sb.NUM_CLASSES)).astype(np.float32)


def _write_dump(dump_dir, num_frames=3, n=4000, start=0):
    """Frames as _OODLogitsDumpMetric writes them; OOD points flatter."""
    os.makedirs(dump_dir, exist_ok=True)
    for i in range(num_frames):
        z = _logits(n)
        ood = RNG.rand(n) < 0.1
        z[ood] *= 0.3
        np.savez(os.path.join(dump_dir, f'r0_{start + i:06d}.npz'),
                 logits=z.astype(np.float16), ood=ood,
                 valid=RNG.rand(n) < 0.95, mapped=np.zeros(n, np.int16),
                 lidar_path=f'f{start + i}')


def _files(d):
    return sorted(os.path.join(d, f) for f in os.listdir(d))


def _fake_root(tmp):
    """A --root with two dumps and every sweep log divided_mass.py reads."""
    root = os.path.join(tmp, 'root')
    _write_dump(os.path.join(root, 'logits'), num_frames=2)
    _write_dump(os.path.join(root, 'logits_test'), num_frames=3, start=2)
    kwargs = dict(bins=2**16, workers=1, chunk=8, top=1)
    for spec in dm.SETS.values():
        dumps = [os.path.join(root, d) for d in spec['dumps']]
        sb.sweep(dumps, os.path.join(root, spec['sweep']), num=8, seed=3,
                 **kwargs)
        sb.sweep(dumps, os.path.join(root, spec['singletons']),
                 subsets=sb.singleton_subsets(), **kwargs)
    return root


def test_bin_edges():
    edges = dm.bin_edges()
    assert edges[0] == 0.0 and edges[1] == dm.M_MIN and edges[-1] == 0.5
    assert np.all(np.diff(edges) > 0)
    for d in dm.DELTAS:
        i = dm.edge_index(edges, d)
        assert edges[i] == d
        # a value exactly at delta opens the bin that starts at delta
        assert dm.bin_index(np.array([d]), edges)[0] == i
    assert list(dm.bin_index(np.array([0.0, 1e-20, 0.5]), edges)) == [
        0, 0, len(edges) - 2]
    for bad in (lambda: dm.edge_index(edges, 0.123456),
                lambda: dm.bin_edges(deltas=(0.6, ))):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid edge accepted')
    print('test_bin_edges passed')


def test_masses_have_no_cancellation():
    z = np.zeros((3, sb.NUM_CLASSES), np.float32)
    z[0, 0] = 30.0  # confident: 1 - max p cancels to 0 in float32
    z[1, 5] = 12.0
    z[2] = _logits(1)[0]
    p = dm.softmax(z)
    p64 = np.exp(z.astype(np.float64) - z.max(axis=1, keepdims=True))
    p64 /= p64.sum(axis=1, keepdims=True)
    u_exact = np.sort(p64, axis=1)[:, :-1].sum(axis=1)
    u = dm.flat_uncertainty(p)
    assert np.allclose(u, u_exact, rtol=1e-5, atol=0), (u, u_exact)
    assert 0 < u[0] < 1e-11  # 23 * exp(-30) = 2.2e-12, not 0
    subsets = [(0, ), (5, 6), tuple(range(12))]
    m = dm.divided_mass(p, sb.subsets_to_mask(subsets))
    for j, subset in enumerate(subsets):
        rest = [c for c in range(sb.NUM_CLASSES) if c not in subset]
        want = np.minimum(p64[:, list(subset)].sum(axis=1),
                          p64[:, rest].sum(axis=1))
        assert np.allclose(m[j], want, rtol=1e-5, atol=0), (subset, m[j], want)
    print('test_masses_have_no_cancellation passed')


def test_divided_counts_match_direct_counts():
    with tempfile.TemporaryDirectory() as tmp:
        _write_dump(tmp)
        files = _files(tmp)
        subsets = [(3, ), (0, 1, 2, 3, 4), tuple(range(12)), (16, )]
        a_mask = sb.subsets_to_mask(subsets)
        edges = dm.bin_edges()
        hist = dm.directory_histograms(files, a_mask, edges, chunk=3)
        frames = [sb.load_frame(f) for f in files]
        z = np.concatenate([f[0] for f in frames])
        ood = np.concatenate([f[1] for f in frames])
        assert list(hist['counts']) == [int((~ood).sum()), int(ood.sum())]
        p = dm.softmax(z)
        m = dm.divided_mass(p, a_mask).astype(np.float64)
        u = dm.flat_uncertainty(p).astype(np.float64)
        rows = dm.divided_stats(hist, hist['counts'], edges)
        flat = dm.flat_stats(hist, hist['counts'], edges)
        for d in dm.DELTAS:
            key = dm.delta_key(d)
            assert math.isclose(flat[f'ood_unc@{key}'],
                                100.0 * (u[ood] >= d).mean())
            for s in range(len(subsets)):
                assert rows[s][f'ood_div_n@{key}'] == int((m[s][ood] >= d).sum())
                assert rows[s][f'id_div_n@{key}'] == int((m[s][~ood] >= d).sum())
                # the divided points are among the flat-uncertain ones
                assert rows[s][f'ood_div@{key}'] <= flat[f'ood_unc@{key}'] + 1e-9
    print('test_divided_counts_match_direct_counts passed')


def test_histogram_metrics_match_exact():
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 roc_curve)
    rng = np.random.RandomState(1)
    m_id = 10.0**rng.uniform(-12, -1, 200000)
    m_ood = 10.0**rng.uniform(-6, np.log10(0.5), 5000)
    edges = dm.bin_edges()
    nbins = len(edges) - 1
    hist_id = np.bincount(dm.bin_index(m_id, edges), minlength=nbins)
    hist_ood = np.bincount(dm.bin_index(m_ood, edges), minlength=nbins)
    auroc, ap, fpr95 = dm.histogram_metrics(hist_ood, hist_id)
    scores = np.concatenate([m_id, m_ood])
    labels = np.r_[np.zeros(len(m_id)), np.ones(len(m_ood))]
    fpr, tpr, _ = roc_curve(labels, scores)
    assert abs(auroc - 100.0 * roc_auc_score(labels, scores)) < 0.05
    assert abs(ap - 100.0 * average_precision_score(labels, scores)) < 0.1
    assert abs(fpr95 - 100.0 * fpr[np.searchsorted(tpr, 0.95)]) < 0.5
    print('test_histogram_metrics_match_exact passed')


def test_several_directories_sum_and_backends_agree():
    with tempfile.TemporaryDirectory() as tmp:
        a, b, both = (os.path.join(tmp, d) for d in ('a', 'b', 'both'))
        _write_dump(a, num_frames=2)
        _write_dump(b, num_frames=2, start=2)
        os.makedirs(both)
        for d in (a, b):
            for f in _files(d):
                shutil.copy(f, both)
        a_mask = sb.subsets_to_mask([(3, ), (0, 1, 2, 3, 4), (16, 17)])
        edges = dm.bin_edges()
        summed = dm.sum_histograms([
            dm.directory_histograms(_files(a), a_mask, edges),
            dm.directory_histograms(_files(b), a_mask, edges)])
        whole = dm.directory_histograms(_files(both), a_mask, edges)
        for key in ('m', 'u', 'counts'):
            assert np.array_equal(summed[key], whole[key]), key
        # the torch backend (CPU tensors here) agrees up to last-ulp binning
        torch_hist = dm.directory_histograms(_files(both), a_mask, edges,
                                             backend='torch', device='cpu',
                                             chunk=2)
        assert np.array_equal(torch_hist['counts'], whole['counts'])
        for key in ('m', 'u'):
            diff = np.abs(torch_hist[key] - whole[key]).sum()
            assert diff <= 1e-3 * whole[key].sum(), (key, diff)
        try:
            dm.directory_histograms(_files(a), a_mask, edges, backend='cuda')
        except ValueError:
            pass
        else:
            raise AssertionError("backend 'cuda' accepted")
    print('test_several_directories_sum_and_backends_agree passed')


def test_zero_divided_counts():
    edges = dm.bin_edges()
    nbins = len(edges) - 1
    at = lambda v: dm.bin_index(np.array([v]), edges)[0]  # noqa: E731
    hist = dict(m=np.zeros((1, 2, nbins), np.int64),
                u=np.zeros((2, nbins), np.int64))
    hist['m'][0, 0, at(1e-7)] = 1000  # no ID point divided at >= 1e-6
    hist['m'][0, 1, at(0.35)] = 5
    hist['m'][0, 1, at(1e-3)] = 5
    hist['u'][0, at(0.4)] = 1000  # u >= m for every point
    hist['u'][1, at(0.35)] = 10
    (row, ) = dm.divided_stats(hist, np.array([1000, 10]), edges)
    assert row['id_div_n@0.3'] == 0 and row['ood_div_n@0.3'] == 5
    assert row['selectivity@0.3'] == float('inf')
    assert row['precision@0.3'] == 100.0
    assert row['ood_retention@0.3'] == 50.0 and row['id_retention@0.3'] == 0.0
    assert math.isfinite(row['log_ratio@0.3'])
    assert row['delta95'] == 1e-3 and row['id_div@delta95'] == 0.0
    rho, n = dm.spearman([1.0, float('nan'), 3.0, 2.0],
                         [1.0, 5.0, float('inf'), 2.0])
    assert n == 3 and abs(rho - 1.0) < 1e-12
    assert math.isnan(dm.spearman([1, 1, 1], [1, 2, 3])[0])
    print('test_zero_divided_counts passed')


def test_tsv_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 't.tsv')
        rows = [dict(name='s16', size_A=1, group_A='car, bicycle', x=0.05,
                     y=float('inf'), z=float('nan'), w=12345678.9)]
        dm.write_tsv(path, rows)
        (back, ) = dm.read_tsv(path)
        assert back['name'] == 's16' and back['size_A'] == 1
        assert back['group_A'] == 'car, bicycle' and back['x'] == 0.05
        assert back['y'] == float('inf') and math.isnan(back['z'])
        assert abs(back['w'] - 12345678.9) / 12345678.9 < 1e-5
        dm.write_tsv(path, [], header=['name', 'worst'])
        assert dm.read_tsv(path) == []
    print('test_tsv_round_trip passed')


def test_run_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_root(tmp)
        sets = dm.resolve_sets(root)
        subsets = dm.default_subsets(sets)
        assert subsets[:24] == sb.singleton_subsets()
        assert len(set(subsets)) == len(subsets)
        out = os.path.join(tmp, 'divided_mass')
        result = dm.run(sets, subsets, out)
        for name in ('histograms.npz', 'cetran.tsv', 'test.tsv',
                     'test_cetran.tsv', 'flat.tsv', 'robust.tsv', 'rho.tsv',
                     'summary.md'):
            assert os.path.exists(os.path.join(out, name)), name
        with np.load(os.path.join(out, 'histograms.npz')) as h:
            assert np.array_equal(h['test_cetran_m'], h['cetran_m'] + h['test_m'])
            assert list(h['names']) == [sb.partition_name(s) for s in subsets]
        rows = dm.read_tsv(os.path.join(out, 'test_cetran.tsv'))
        assert [r['name'] for r in rows] == [sb.partition_name(s) for s in subsets]
        improvements = {key: {r['name']: r['improvement'] for r in table}
                        for key, table in result['tables'].items()}
        robust = {r['name'] for r in dm.read_tsv(os.path.join(out, 'robust.tsv'))}
        for r in rows:
            want = all(imp[r['name']] > 0 for imp in improvements.values())
            assert r['robust'] == int(want) and (r['name'] in robust) == want
        rho = dm.read_tsv(os.path.join(out, 'rho.tsv'))
        assert len(rho) == (3 * 2 * len(dm.DELTAS) * len(dm.TARGETS)
                            * len(dm.STATISTICS))
        agreement = [c for c in result['checks'] if 'both sweep logs' in c['check']]
        assert len(agreement) == 3 and all(c['status'] == 'PASS' for c in agreement)
        assert all(c['status'] in ('PASS', 'FAIL') for c in result['checks'])
        with open(os.path.join(out, 'summary.md')) as fh:
            text = fh.read()
        for section in ('## Consistency checks', '## Flat MSP reference',
                        '## Single-class splits, Cetran', '## Robust splits',
                        '## Spearman rho'):
            assert section in text, section
    print('test_run_end_to_end passed')


def test_missing_inputs_are_explained():
    with tempfile.TemporaryDirectory() as tmp:
        root = _fake_root(tmp)
        sets = dm.resolve_sets(root)
        # a sweep of the test set with another list of partitions: refused
        path = os.path.join(root, 'bipartitions_test', 'partitions.json')
        with open(path) as fh:
            parts = json.load(fh)
        with open(path, 'w') as fh:
            json.dump(parts[:-1], fh)
        try:
            dm.default_subsets(sets)
        except ValueError as err:
            assert '--seed' in str(err)
        else:
            raise AssertionError('different partitions accepted')
        # no single-class sweep yet: the error names the tool to run
        shutil.rmtree(os.path.join(root, 'singletons_test'))
        try:
            dm.sweep_rows(sets['test']['sweep_log'],
                          sets['test']['singletons_log'])
        except FileNotFoundError as err:
            assert 'sweep_bipartitions.py' in str(err)
        else:
            raise AssertionError('missing single-class log accepted')
        # a split the logs do not hold: KeyError naming it
        rows = dm.sweep_rows(sets['cetran']['sweep_log'],
                             sets['cetran']['singletons_log'])
        try:
            dm.split_metrics(rows, ['s99'])
        except KeyError as err:
            assert 's99' in str(err)
        else:
            raise AssertionError('missing split accepted')
    print('test_missing_inputs_are_explained passed')


if __name__ == '__main__':
    test_bin_edges()
    test_masses_have_no_cancellation()
    test_divided_counts_match_direct_counts()
    test_histogram_metrics_match_exact()
    test_several_directories_sum_and_backends_agree()
    test_zero_divided_counts()
    test_tsv_round_trip()
    test_run_end_to_end()
    test_missing_inputs_are_explained()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$ENVPY -m pytest -q tests/test_divided_mass.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'divided_mass'`.

- [ ] **Step 3: Implement** — create `tools/divided_mass.py`:

```python
#!/usr/bin/env python
"""Divided mass of two-group splits, measured from the logit dumps.

For a split A | B of the 24 DSO classes and a point with softmax p, the
divided mass is m = min(P_A, P_B), P_g being the softmax mass of the
classes of g. Group MSP of the split is m - 1, so a point is flagged only
when its mass is divided across the boundary. At a threshold delta a point
is *divided* when m >= delta (delta = 0.05: 0.05 <= P_A <= 0.95), and the
OOD / ID divided shares are Group MSP's TPR / FPR at that threshold. Flat
MSP gets the same treatment through u = 1 - max_c p_c: the flat-uncertain
points (u >= delta) include the divided ones of every split, since u >= m
for every point.

Per evaluation set (``SETS``: Cetran, Test, Test + Cetran) the tool
histograms m for every split -- the 24 single-class splits and the random
sweep's partitions -- and u, over bins log-spaced from 1e-15 to 0.5 whose
edges include every reported delta (``DELTAS``). Per split and delta it
reports the OOD / ID divided counts and shares, the divided precision
OOD / (OOD + ID divided), the selectivity (OOD retention over ID retention,
retention = divided / flat-uncertain) and the log OOD/ID divided ratio;
per split also delta95 -- the largest bin edge with at least 95% of the OOD
points divided -- and the ID divided share there (about Group MSP's
FPR@95). It joins each split's Group MSP metrics, their delta against flat
MSP and ``improvement`` from the sweep logs, flags the robust splits
(improvement > 0 on every set) and computes the Spearman correlations of
the divided statistics with the metric deltas. The model is never run.

Inputs under --root (default work_dirs/p3former_2xb1_3x_dso_ood_dump): the
logits dumps ``logits`` (Cetran) and ``logits_test`` (test), the random
sweeps ``bipartitions{,_test,_test_cetran}`` and the single-class sweeps
``singletons{,_test,_test_cetran}`` of tools/sweep_bipartitions.py. Each
dump directory is read once; Test + Cetran sums the histograms of both.
Outputs in --out-dir (default <root>/divided_mass):
    histograms.npz   bin edges, split masks and every histogram
    <set>.tsv        one row per split, for cetran, test and test_cetran
    flat.tsv         the flat-MSP reference, one row per set
    robust.tsv       the splits with improvement > 0 on every set
    rho.tsv          Spearman correlations, long format
    summary.md       consistency checks (FAIL rows included), tables, rho

Run from the repo root after the sweeps (about 15 min on a GPU):
    python tools/divided_mass.py --backend torch --device cuda:0
    python tools/plot_divided_mass.py
"""
import argparse
import csv
import glob
import json
import math
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import sweep_bipartitions as sb  # noqa: E402
from evaluation.functional.ood_eval import metrics_from_histograms  # noqa: E402
from summarize_hierarchy_ablation import (_improvement, parse_logs,  # noqa: E402
                                          summarise)

ROOT = 'work_dirs/p3former_2xb1_3x_dso_ood_dump'
# Evaluation sets: logits dump directories and sweep output directories,
# relative to --root. Test + Cetran is the two dumps together (exactly
# dso_infos_test_cetran.pkl).
SETS = OrderedDict([
    ('cetran', dict(label='Cetran', dumps=['logits'],
                    sweep='bipartitions', singletons='singletons')),
    ('test', dict(label='Test', dumps=['logits_test'],
                  sweep='bipartitions_test', singletons='singletons_test')),
    ('test_cetran', dict(label='Test + Cetran',
                         dumps=['logits_test', 'logits'],
                         sweep='bipartitions_test_cetran',
                         singletons='singletons_test_cetran')),
])
DELTAS = (1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3)
HEADLINE = 0.05  # the delta of the summary tables and the bubble charts
M_MIN = 1e-15  # first log-spaced edge; m below it shares the bin [0, M_MIN)
BINS_PER_DECADE = 200
# improvement = mean dAUROC + mean dAP - mean dFPR@95 over Group MSP, Group
# Energy and Group Entropy (summarize_hierarchy_ablation.py --family group
# --exclude odin; MaxLogit is always dropped)
EXCLUDE = ('maxlogit', 'odin')
GROUP_KEYS = ('group_msp', 'group_energy', 'group_entropy')
# Tolerances in percentage points: Group / flat MSP recomputed from the
# histograms against the sweep logs, and the same split in the random and
# the single-class sweep (blocks of another shape may round the last ulp
# differently).
CHECK_TOL = OrderedDict([('auroc', 0.1), ('ap', 0.1), ('fpr95', 0.5)])
SINGLETON_TOL = 0.02
STATISTICS = ('ood_div', 'id_div', 'log_ratio', 'precision')
TARGETS = ('d_auroc', 'd_ap', 'd_fpr95', 'improvement')


# --------------------------------------------------------------------- bins
def bin_edges(deltas=DELTAS, m_min=M_MIN, per_decade=BINS_PER_DECADE):
    """Ascending bin edges: 0, then ``per_decade`` log-spaced edges per
    decade from ``m_min`` to 0.5, with every delta inserted as an exact edge.
    Bin i is [edges[i], edges[i + 1]); m = 0.5 falls into the last one."""
    for d in deltas:
        if not m_min < d < 0.5:
            raise ValueError(f'delta {d} outside ({m_min}, 0.5)')
    steps = int(round(math.log10(0.5 / m_min) * per_decade))
    grid = np.logspace(math.log10(m_min), math.log10(0.5), steps + 1)
    grid[0], grid[-1] = m_min, 0.5
    return np.unique(np.concatenate([[0.0], grid,
                                     np.asarray(deltas, np.float64)]))


def edge_index(edges, value):
    """Index of the bin edge equal to ``value``."""
    i = int(np.searchsorted(edges, value))
    if i >= len(edges) or edges[i] != value:
        raise ValueError(f'{value} is not a bin edge')
    return i


def bin_index(values, edges):
    """Bin of every value: i with edges[i] <= value < edges[i + 1], the last
    bin closed on the right."""
    b = np.searchsorted(edges, np.asarray(values, np.float64),
                        side='right') - 1
    return np.clip(b, 0, len(edges) - 2)


def delta_key(delta):
    """Column suffix of a threshold: 0.05 -> '0.05', 1e-4 -> '0.0001'."""
    return f'{delta:g}'


# ------------------------------------------------------------ point masses
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def flat_uncertainty(p):
    """u = 1 - max_c p_c, as the sum of the other classes' probabilities:
    exact for confident points, where 1 - max cancels."""
    q = p.copy()
    q[np.arange(len(q)), p.argmax(axis=1)] = 0.0
    return q.sum(axis=1)


def divided_mass(p, a_mask):
    """m = min(P_A, P_B) [c, N] of the splits in ``a_mask`` [c, C]; both
    group masses are sums over their own classes (never 1 - P_A)."""
    M = a_mask.astype(p.dtype)
    return np.minimum(M @ p.T, (1.0 - M) @ p.T)


# --------------------------------------------------------------- histograms
def directory_histograms(files, a_mask, edges, backend='numpy',
                         device='cuda:0', chunk=64):
    """Histograms of m (every split) and u over the frames in ``files``:
    dict(m=[S, 2, B], u=[2, B], counts=[2]) of int64, index 0 = ID points,
    1 = OOD points."""
    if backend == 'torch':
        return _torch_directory_histograms(files, a_mask, edges, device,
                                           chunk)
    if backend != 'numpy':
        raise ValueError(f"backend must be 'numpy' or 'torch', got {backend}")
    num, nbins = a_mask.shape[0], len(edges) - 1
    m_hist = np.zeros((num, 2, nbins), np.int64)
    u_hist = np.zeros((2, nbins), np.int64)
    counts = np.zeros(2, np.int64)
    for z, ood in sb.prefetch_frames(files):
        if not z.shape[0]:
            continue
        p = softmax(z)
        counts += [int((~ood).sum()), int(ood.sum())]
        u_bins = bin_index(flat_uncertainty(p), edges)
        for lab, sel in ((0, ~ood), (1, ood)):
            u_hist[lab] += np.bincount(u_bins[sel], minlength=nbins)
        for j0 in range(0, num, chunk):
            m_bins = bin_index(divided_mass(p, a_mask[j0:j0 + chunk]), edges)
            for j in range(m_bins.shape[0]):
                for lab, sel in ((0, ~ood), (1, ood)):
                    m_hist[j0 + j, lab] += np.bincount(m_bins[j, sel],
                                                       minlength=nbins)
    return dict(m=m_hist, u=u_hist, counts=counts)


def _torch_bin_index(values, edges_t):
    torch = sb.require_torch()
    b = torch.bucketize(values.double(), edges_t, right=True) - 1
    return b.clamp_(0, edges_t.numel() - 2)


def _torch_directory_histograms(files, a_mask, edges, device, chunk):
    """:func:`directory_histograms` on a torch device (TF32 off)."""
    torch = sb.require_torch()
    dev = torch.device(device)
    mask = torch.from_numpy(a_mask).to(dev)
    edges_t = torch.from_numpy(edges).to(dev)
    num, nbins = a_mask.shape[0], len(edges) - 1
    m_hist = torch.zeros((num, 2, nbins), dtype=torch.int64, device=dev)
    u_hist = torch.zeros((2, nbins), dtype=torch.int64, device=dev)
    counts = np.zeros(2, np.int64)
    for z, ood in sb.prefetch_frames(files):
        if not z.shape[0]:
            continue
        order = np.argsort(ood, kind='stable')  # ID points first, OOD last
        n_id = int(ood.size - ood.sum())
        counts += [n_id, ood.size - n_id]
        p = torch.softmax(torch.from_numpy(z[order]).to(dev), dim=1)
        q = p.clone()
        q[torch.arange(q.shape[0], device=dev), p.argmax(dim=1)] = 0.0
        u_bins = _torch_bin_index(q.sum(dim=1), edges_t)
        u_hist[0] += sb.torch_counts(u_bins[:n_id], nbins)
        u_hist[1] += sb.torch_counts(u_bins[n_id:], nbins)
        for j0 in range(0, num, chunk):
            M = mask[j0:j0 + chunk].float()
            c = M.shape[0]
            b = _torch_bin_index(torch.minimum(M @ p.T, (1.0 - M) @ p.T),
                                 edges_t)
            b += (torch.arange(c, device=dev) * nbins)[:, None]
            m_hist[j0:j0 + c, 0] += sb.torch_counts(
                b[:, :n_id].reshape(-1), c * nbins).view(c, nbins)
            m_hist[j0:j0 + c, 1] += sb.torch_counts(
                b[:, n_id:].reshape(-1), c * nbins).view(c, nbins)
    return dict(m=m_hist.cpu().numpy(), u=u_hist.cpu().numpy(),
                counts=counts)


def sum_histograms(parts):
    """Element-wise sum of several :func:`directory_histograms` results."""
    return {key: sum(part[key] for part in parts) for key in parts[0]}


# --------------------------------------------------------------- statistics
def share(n, total):
    """``n`` as a percentage of ``total`` (NaN when the total is 0)."""
    return 100.0 * n / total if total else float('nan')


def ratio(a, b):
    """a / b, with inf for a > 0 = b and NaN for 0 / 0."""
    if b:
        return a / b
    return float('inf') if a else float('nan')


def tail_counts(hist):
    """tail[..., i] = count of values in bin i or above (>= edges[i])."""
    return np.cumsum(hist[..., ::-1], axis=-1)[..., ::-1]


def depth95(tail_ood, n_ood):
    """Largest bin index whose tail holds >= 95% of the OOD points."""
    return int(np.flatnonzero(tail_ood >= 0.95 * n_ood)[-1])


def divided_stats(hist, counts, edges, deltas=DELTAS):
    """Divided statistics of every split, one OrderedDict each: per delta the
    divided counts and shares (%), the divided precision (%), the OOD / ID
    retention against flat MSP (divided / flat-uncertain, %), their ratio
    (the selectivity) and the log OOD/ID divided ratio (+0.5 on each count),
    then delta95 and the ID divided share there.

    hist: dict(m=[S, 2, B], u=[2, B]) of one set; counts: (n_id, n_ood)."""
    n_id, n_ood = (int(c) for c in counts)
    tail, u_tail = tail_counts(hist['m']), tail_counts(hist['u'])
    idx = [edge_index(edges, d) for d in deltas]
    rows = []
    for s in range(tail.shape[0]):
        row = OrderedDict()
        for d, i in zip(deltas, idx):
            key = delta_key(d)
            id_div, ood_div = int(tail[s, 0, i]), int(tail[s, 1, i])
            id_unc, ood_unc = int(u_tail[0, i]), int(u_tail[1, i])
            row[f'ood_div_n@{key}'] = ood_div
            row[f'id_div_n@{key}'] = id_div
            row[f'ood_div@{key}'] = share(ood_div, n_ood)
            row[f'id_div@{key}'] = share(id_div, n_id)
            row[f'precision@{key}'] = share(ood_div, ood_div + id_div)
            row[f'ood_retention@{key}'] = share(ood_div, ood_unc)
            row[f'id_retention@{key}'] = share(id_div, id_unc)
            row[f'selectivity@{key}'] = ratio(ood_div * id_unc,
                                              ood_unc * id_div)
            row[f'log_ratio@{key}'] = math.log10(
                ((ood_div + 0.5) / n_ood) / ((id_div + 0.5) / n_id))
        i95 = depth95(tail[s, 1], n_ood)
        row['delta95'] = float(edges[i95])
        row['id_div@delta95'] = share(int(tail[s, 0, i95]), n_id)
        rows.append(row)
    return rows


def flat_stats(hist, counts, edges, deltas=DELTAS):
    """The flat-MSP reference of one set: per delta the uncertain shares
    (u >= delta) and their precision, then u95 and the ID share there."""
    n_id, n_ood = (int(c) for c in counts)
    u_tail = tail_counts(hist['u'])
    row = OrderedDict()
    for d in deltas:
        i, key = edge_index(edges, d), delta_key(d)
        id_unc, ood_unc = int(u_tail[0, i]), int(u_tail[1, i])
        row[f'ood_unc@{key}'] = share(ood_unc, n_ood)
        row[f'id_unc@{key}'] = share(id_unc, n_id)
        row[f'precision@{key}'] = share(ood_unc, ood_unc + id_unc)
    i95 = depth95(u_tail[1], n_ood)
    row['u95'] = float(edges[i95])
    row['id_unc@u95'] = share(int(u_tail[0, i95]), n_id)
    return row


def histogram_metrics(hist_ood, hist_id):
    """AUROC / AP / FPR@95 in percent of the score whose bins are given in
    ascending order (higher = more OOD)."""
    m = metrics_from_histograms(hist_ood, hist_id)
    return tuple(100.0 * m[k] for k in ('auroc', 'ap', 'fpr95'))


def spearman(x, y):
    """(rho, n) over the pairs without NaN (inf ranks last); rho is NaN with
    fewer than 3 pairs or a constant side."""
    from scipy.stats import rankdata
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    keep = ~(np.isnan(x) | np.isnan(y))
    n = int(keep.sum())
    if n < 3:
        return float('nan'), n
    rx, ry = rankdata(x[keep]), rankdata(y[keep])
    if rx.std() == 0 or ry.std() == 0:
        return float('nan'), n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def correlations(table, deltas=DELTAS):
    """Spearman rho of every divided statistic at every delta against every
    target, over the single-class splits ('single') and all splits
    ('all')."""
    out = []
    for population, rows in (('single', [r for r in table if r['size_A'] == 1]),
                             ('all', table)):
        for d in deltas:
            key = delta_key(d)
            for target in TARGETS:
                y = [r[target] for r in rows]
                for stat in STATISTICS:
                    rho, n = spearman([r[f'{stat}@{key}'] for r in rows], y)
                    out.append(OrderedDict([
                        ('population', population), ('delta', key),
                        ('statistic', stat), ('target', target),
                        ('rho', rho), ('n', n)]))
    return out


# -------------------------------------------------------------- sweep logs
def sweep_rows(sweep_log, singletons_log):
    """Metric rows of one set. Where both logs hold a split the random sweep
    wins (parse_logs keeps the last file's row)."""
    for path in (sweep_log, singletons_log):
        if not osp.exists(path):
            raise FileNotFoundError(
                f'{path} is missing: run tools/sweep_bipartitions.py for this '
                'set first (the random sweep, and --subsets singletons)')
    return parse_logs([singletons_log, sweep_log])


def split_metrics(rows, names):
    """Per split: Group MSP AUROC / AP / FPR@95, their delta against flat MSP
    and the improvement (Group family, EXCLUDE dropped)."""
    _, summary = summarise(rows, exclude=set(EXCLUDE))
    missing = [n for n in names if f'{n}_group_msp' not in rows
               or 'group' not in summary.get(n, {})]
    if missing:
        raise KeyError(f'{len(missing)} splits are missing from the sweep '
                       f'logs, e.g. {missing[:3]}')
    msp = rows['msp']
    out = OrderedDict()
    for name in names:
        g = rows[f'{name}_group_msp']
        out[name] = OrderedDict([
            ('gmsp_auroc', g[0]), ('gmsp_ap', g[1]), ('gmsp_fpr95', g[2]),
            ('d_auroc', g[0] - msp[0]), ('d_ap', g[1] - msp[1]),
            ('d_fpr95', g[2] - msp[2]),
            ('improvement', _improvement(summary[name]['group']['mean'])),
        ])
    return out


def log_agreement(sweep_log, singletons_log):
    """(number of splits in both logs, largest |difference| of their Group
    rows over AUROC / AP / FPR@95)."""
    a, b = parse_logs([sweep_log]), parse_logs([singletons_log])
    common = [k for k in a if k in b and k.endswith(GROUP_KEYS)]
    worst = max((abs(x - y) for k in common for x, y in zip(a[k], b[k])),
                default=0.0)
    return len({k.rsplit('_group_', 1)[0] for k in common}), worst


# ------------------------------------------------------------------ splits
def resolve_sets(root, sets=SETS):
    """``SETS`` with paths under ``root``: label, dumps, sweep_log,
    singletons_log and the random sweep's partitions.json."""
    return OrderedDict(
        (key, dict(label=spec['label'],
                   dumps=[osp.join(root, d) for d in spec['dumps']],
                   sweep_log=osp.join(root, spec['sweep'], 'bipartitions.log'),
                   singletons_log=osp.join(root, spec['singletons'],
                                           'bipartitions.log'),
                   partitions=osp.join(root, spec['sweep'],
                                       'partitions.json')))
        for key, spec in sets.items())


def default_subsets(sets):
    """The 24 single-class splits, then the random sweep's partitions, which
    must be the same on every set."""
    lists = []
    for spec in sets.values():
        with open(spec['partitions']) as fh:
            lists.append([sb.canonical(item['A']) for item in json.load(fh)])
    if any(other != lists[0] for other in lists[1:]):
        raise ValueError('the random sweeps of the sets hold different '
                         'partitions: rerun them with the same --num / --seed')
    subsets = sb.singleton_subsets()
    for subset in lists[0]:
        if subset not in subsets:
            subsets.append(subset)
    return subsets


def split_table(names, subsets, hist, edges, metrics, deltas=DELTAS):
    """One row per split: composition, point counts, divided statistics,
    the sweep's metrics and the Group MSP metrics recomputed from the m
    histograms (hist_*, for the consistency check)."""
    n_id, n_ood = (int(c) for c in hist['counts'])
    stats = divided_stats(hist, hist['counts'], edges, deltas)
    rows = []
    for s, (name, subset) in enumerate(zip(names, subsets)):
        row = OrderedDict([
            ('name', name), ('size_A', len(subset)),
            ('group_A', ', '.join(sb.CLASSES[c] for c in subset)),
            ('n_id', n_id), ('n_ood', n_ood)])
        row.update(stats[s])
        row.update(metrics[name])
        auroc, ap, fpr95 = histogram_metrics(hist['m'][s, 1], hist['m'][s, 0])
        row.update([('hist_auroc', auroc), ('hist_ap', ap),
                    ('hist_fpr95', fpr95)])
        rows.append(row)
    return rows


def flat_row(label, hist, edges, rows, deltas=DELTAS):
    """The flat-MSP row of one set, with the sweep's msp row and the same
    metrics recomputed from the u histograms."""
    row = OrderedDict([('set', label), ('n_id', int(hist['counts'][0])),
                       ('n_ood', int(hist['counts'][1]))])
    row.update(flat_stats(hist, hist['counts'], edges, deltas))
    msp = rows['msp']
    auroc, ap, fpr95 = histogram_metrics(hist['u'][1], hist['u'][0])
    row.update([('msp_auroc', msp[0]), ('msp_ap', msp[1]),
                ('msp_fpr95', msp[2]), ('hist_auroc', auroc),
                ('hist_ap', ap), ('hist_fpr95', fpr95)])
    return row


def consistency_checks(label, table, flat, agreement):
    """The summary's check rows: worst |difference| against its tolerance,
    PASS or FAIL."""
    checks = []
    for metric, tol in CHECK_TOL.items():
        worst = max(abs(r[f'hist_{metric}'] - r[f'gmsp_{metric}'])
                    for r in table)
        checks.append((f'Group MSP {metric}, histograms vs sweep', worst, tol))
        worst = abs(flat[f'hist_{metric}'] - flat[f'msp_{metric}'])
        checks.append((f'flat MSP {metric}, histograms vs sweep', worst, tol))
    n_common, worst = agreement
    checks.append((f'{n_common} splits in both sweep logs', worst,
                   SINGLETON_TOL))
    return [OrderedDict([('set', label), ('check', check), ('worst', worst),
                         ('tolerance', tol),
                         ('status', 'PASS' if worst <= tol else 'FAIL')])
            for check, worst, tol in checks]


def robust_names(tables):
    """Splits with improvement > 0 on every set, in table order."""
    first = next(iter(tables.values()))
    by_set = [{r['name']: r['improvement'] for r in rows}
              for rows in tables.values()]
    return [r['name'] for r in first
            if all(imp[r['name']] > 0 for imp in by_set)]


def robust_table(tables, names):
    """One row per robust split: composition, improvement per set and the
    worst of them; the best worst case first."""
    by_set = OrderedDict((key, {r['name']: r for r in rows})
                         for key, rows in tables.items())
    rows = []
    for name in names:
        first = next(iter(by_set.values()))[name]
        row = OrderedDict([('name', name), ('size_A', first['size_A']),
                           ('group_A', first['group_A'])])
        for key in by_set:
            row[f'improvement_{key}'] = by_set[key][name]['improvement']
        row['worst'] = min(by_set[key][name]['improvement'] for key in by_set)
        rows.append(row)
    return sorted(rows, key=lambda r: -r['worst'])


# --------------------------------------------------------------------- I/O
def write_tsv(path, rows, header=None):
    """Rows (dicts sharing their keys) as a tab-separated table; ``header``
    names the columns when ``rows`` may be empty."""
    header = list(header or rows[0])
    with open(path, 'w', newline='') as fh:
        writer = csv.writer(fh, delimiter='\t', lineterminator='\n')
        writer.writerow(header)
        for row in rows:
            writer.writerow([f'{row[k]:.6g}' if isinstance(row[k], float)
                             else row[k] for k in header])


def read_tsv(path):
    """Rows of a :func:`write_tsv` table; numbers are parsed back (ints stay
    ints; 'inf' / 'nan' become floats)."""
    with open(path, newline='') as fh:
        return [OrderedDict((k, _parse(v)) for k, v in row.items())
                for row in csv.DictReader(fh, delimiter='\t')]


def _parse(text):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def md_table(header, rows):
    """A markdown table as a list of lines."""
    lines = ['| ' + ' | '.join(str(h) for h in header) + ' |',
             '| ' + ' | '.join('---' for _ in header) + ' |']
    return lines + ['| ' + ' | '.join(str(c) for c in row) + ' |'
                    for row in rows]


def fmt(value, digits=2):
    """A number for a markdown cell ('-' for NaN)."""
    if isinstance(value, str):
        return value
    if math.isnan(value):
        return '-'
    if math.isinf(value):
        return 'inf'
    return f'{value:.{digits}f}'


def pct(value):
    """A share that spans decades: three significant digits."""
    return '-' if math.isnan(value) else f'{value:.3g}'


def write_summary(path, sets, tables, flats, checks, robust_rows, rho,
                  headline=HEADLINE):
    key = delta_key(headline)
    lines = ['# Divided mass of two-group splits', '',
             'Divided at delta: m = min(P_A, P_B) >= delta. Flat uncertain: '
             f'u = 1 - max p >= delta. The tables use delta = {key}; the TSVs '
             'hold every delta.', '', '## Consistency checks', '']
    lines += md_table(['set', 'check', 'worst abs diff', 'tolerance',
                       'status'],
                      [(c['set'], c['check'], fmt(c['worst'], 3),
                        fmt(c['tolerance'], 3), c['status']) for c in checks])
    lines += ['', f'## Flat MSP reference (delta = {key})', '']
    lines += md_table(['set', 'OOD uncertain %', 'ID uncertain %',
                       'precision %', 'u95', 'ID uncertain % at u95',
                       'MSP FPR@95'],
                      [(f['set'], pct(f[f'ood_unc@{key}']),
                        pct(f[f'id_unc@{key}']), pct(f[f'precision@{key}']),
                        f'{f["u95"]:.3g}', fmt(f['id_unc@u95']),
                        fmt(f['msp_fpr95'])) for f in flats])
    for set_key, rows in tables.items():
        single = sorted((r for r in rows if r['size_A'] == 1),
                        key=lambda r: -r['improvement'])
        lines += ['', f'## Single-class splits, {sets[set_key]["label"]} '
                  f'(delta = {key})', '']
        lines += md_table(
            ['class', 'OOD div %', 'ID div %', 'precision %', 'selectivity',
             'delta95', 'ID div % at delta95', 'dAUROC', 'dAP', 'dFPR@95',
             'improvement'],
            [(r['group_A'], pct(r[f'ood_div@{key}']), pct(r[f'id_div@{key}']),
              pct(r[f'precision@{key}']), fmt(r[f'selectivity@{key}']),
              f'{r["delta95"]:.2g}', fmt(r['id_div@delta95']),
              fmt(r['d_auroc']), fmt(r['d_ap']), fmt(r['d_fpr95']),
              fmt(r['improvement'])) for r in single])
    last = list(tables)[-1]
    by_name = {r['name']: r for r in tables[last]}
    lines += ['', '## Robust splits: improvement > 0 on every set '
              f'({len(robust_rows)})', '']
    lines += md_table(
        ['name', 'classes']
        + [f'improvement {sets[k]["label"]}' for k in tables]
        + [f'OOD / ID div % ({sets[last]["label"]})'],
        [(r['name'], r['group_A'],
          *[fmt(r[f'improvement_{k}']) for k in tables],
          f'{pct(by_name[r["name"]][f"ood_div@{key}"])} / '
          f'{pct(by_name[r["name"]][f"id_div@{key}"])}')
         for r in robust_rows])
    lines += ['', f'## Spearman rho at delta = {key}']
    for set_key, rows in tables.items():
        for population, what in (('single', 'single-class splits'),
                                 ('all', 'all splits')):
            n = sum(1 for r in rows if population == 'all' or r['size_A'] == 1)
            sel = {(r['statistic'], r['target']): r['rho'] for r in rho
                   if r['set'] == set_key and r['population'] == population
                   and r['delta'] == key}
            lines += ['', f'{sets[set_key]["label"]}, {what} (n = {n}):', '']
            lines += md_table(['statistic'] + list(TARGETS),
                              [(s, *[fmt(sel[(s, t)]) for t in TARGETS])
                               for s in STATISTICS])
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------- main
def run(sets, subsets, out_dir, backend='numpy', device='cuda:0', chunk=64,
        deltas=DELTAS):
    """Every output for ``sets`` (see :func:`resolve_sets`) and the splits
    ``subsets``; returns dict(tables, checks, robust)."""
    names = [sb.partition_name(s) for s in subsets]
    a_mask = sb.subsets_to_mask(subsets)
    edges = bin_edges(deltas)
    per_dir = OrderedDict()
    for spec in sets.values():
        for d in spec['dumps']:
            if d in per_dir:
                continue
            files = sorted(glob.glob(osp.join(d, '*.npz')))
            if not files:
                raise FileNotFoundError(f'no .npz dumps in {d}')
            t0 = time.time()
            print(f'{d}: {len(files)} frames, {len(names)} splits', flush=True)
            per_dir[d] = directory_histograms(files, a_mask, edges, backend,
                                              device, chunk)
            print(f'  histograms in {time.time() - t0:.0f} s', flush=True)

    hists, tables, flats, checks = OrderedDict(), OrderedDict(), [], []
    for key, spec in sets.items():
        hist = sum_histograms([per_dir[d] for d in spec['dumps']])
        rows = sweep_rows(spec['sweep_log'], spec['singletons_log'])
        hists[key] = hist
        tables[key] = split_table(names, subsets, hist, edges,
                                  split_metrics(rows, names), deltas)
        flats.append(flat_row(spec['label'], hist, edges, rows, deltas))
        checks += consistency_checks(
            spec['label'], tables[key], flats[-1],
            log_agreement(spec['sweep_log'], spec['singletons_log']))
    robust = robust_names(tables)
    for rows in tables.values():
        for row in rows:
            row['robust'] = int(row['name'] in robust)
    rho = [OrderedDict([('set', key)] + list(r.items()))
           for key, rows in tables.items() for r in correlations(rows, deltas)]
    robust_rows = robust_table(tables, robust)

    os.makedirs(out_dir, exist_ok=True)
    arrays = dict(edges=edges, names=np.array(names), a_mask=a_mask)
    for key, hist in hists.items():
        for part, value in hist.items():
            arrays[f'{key}_{part}'] = value
    np.savez_compressed(osp.join(out_dir, 'histograms.npz'), **arrays)
    for key, rows in tables.items():
        write_tsv(osp.join(out_dir, f'{key}.tsv'), rows)
    write_tsv(osp.join(out_dir, 'flat.tsv'), flats)
    write_tsv(osp.join(out_dir, 'robust.tsv'), robust_rows,
              header=['name', 'size_A', 'group_A']
              + [f'improvement_{k}' for k in tables] + ['worst'])
    write_tsv(osp.join(out_dir, 'rho.tsv'), rho)
    write_summary(osp.join(out_dir, 'summary.md'), sets, tables, flats,
                  checks, robust_rows, rho)
    for c in checks:
        print(f'{c["status"]}  {c["set"]}: {c["check"]}: worst '
              f'{c["worst"]:.3f} (tolerance {c["tolerance"]})')
    print(f'{len(robust)} robust splits; wrote {out_dir}')
    return dict(tables=tables, checks=checks, robust=robust)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT,
                    help='directory holding the dumps and the sweeps (SETS)')
    ap.add_argument('--out-dir', default=None,
                    help='default: <root>/divided_mass')
    ap.add_argument('--backend', choices=['numpy', 'torch'], default='numpy',
                    help='numpy: one CPU process (hours on the full dumps); '
                    'torch: --device')
    ap.add_argument('--device', default='cuda:0',
                    help='torch device of --backend torch')
    ap.add_argument('--chunk', type=int, default=64,
                    help='splits per matmul block')
    args = ap.parse_args()
    sets = resolve_sets(args.root)
    run(sets, default_subsets(sets),
        args.out_dir or osp.join(args.root, 'divided_mass'),
        backend=args.backend, device=args.device, chunk=args.chunk)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$ENVPY -m pytest -q tests/test_divided_mass.py && $ENVPY tests/test_divided_mass.py`
Expected: every test passes, ending in `ALL TESTS PASSED`.

- [ ] **Step 5: Full suite, `--help`, commit**

Run: `$ENVPY -m pytest -q tests && $ENVPY tools/divided_mass.py --help`
Expected: all tests pass, and the help prints the first docstring paragraph and the options.

```bash
git add tools/divided_mass.py tests/test_divided_mass.py
git commit -m "$(cat <<'EOF'
Measure the divided mass of two-group splits from the logit dumps

tools/divided_mass.py histograms m = min(P_A, P_B) of the 24 single-class
splits and the 500 sweep splits, and u = 1 - max p for flat MSP, then
reports per split and threshold the OOD / ID divided shares, divided
precision, selectivity against flat MSP and delta95, joined with the
sweep's metric deltas, robust flags and Spearman correlations.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `tools/plot_divided_mass.py`

**Files:**
- Create: `tools/plot_divided_mass.py`
- Test: `tests/test_plot_divided_mass.py`

**Interfaces:**
- Consumes (Task 2): `dm.ROOT`, `dm.SETS`, `dm.DELTAS`, `dm.HEADLINE`, `dm.delta_key`, `dm.read_tsv`, `dm.write_tsv` (tests), `dm.STATISTICS`, `dm.TARGETS` (tests), plus the TSV columns listed in Task 2.
- Produces (used by Tasks 6, 7):
  - `STYLE`, `GAIN`, `DROP`, `GAIN_TXT`, `DROP_TXT`;
  - `tint(colour, k=0.35)`, `signed(value)`, `area(value)`, `save(fig, stem)`;
  - `bubble_grid(panels, xlabel, ylabel, stem, stars=None, backgrounds=None)`. `panels` is an OrderedDict mapping a set label to a list of dicts `x, y, value, text` (`text` None means no label); `stars` maps a label to `(x, y)`; `backgrounds` maps a label to a list of `(x, y)`;
  - `bubble_panel`, `place_labels`, `_box`, `_overlap`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_plot_divided_mass.py`:

```python
"""Tests for tools/plot_divided_mass.py (smoke tests on synthetic tables).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_plot_divided_mass.py
"""
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import divided_mass as dm  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402

SCRIPT = os.path.join(_REPO_ROOT, 'tools', 'plot_divided_mass.py')


def _fake_divided_dir(tmp):
    """The tables of tools/divided_mass.py that the figures read, with one
    split dividing no ID point at delta = 0.05."""
    rng = np.random.RandomState(0)
    subsets = sb.singleton_subsets() + [(0, 1), (2, 5, 16)]
    for key in dm.SETS:
        rows = []
        for s in subsets:
            row = OrderedDict([
                ('name', sb.partition_name(s)), ('size_A', len(s)),
                ('group_A', ', '.join(sb.CLASSES[c] for c in s))])
            for d in dm.DELTAS:
                k = dm.delta_key(d)
                row[f'ood_div@{k}'] = float(10**rng.uniform(-1, 1.5))
                row[f'id_div@{k}'] = float(10**rng.uniform(-2, 1))
            robust = s in [(16, ), (2, 5, 16)]
            row['improvement'] = float(rng.uniform(1, 35) if robust
                                       else rng.uniform(-40, 35))
            row['robust'] = int(robust)
            rows.append(row)
        rows[3]['id_div@0.05'] = 0.0  # {truck}: drawn on the axis floor
        dm.write_tsv(os.path.join(tmp, f'{key}.tsv'), rows)
    dm.write_tsv(os.path.join(tmp, 'flat.tsv'), [
        OrderedDict([('set', spec['label']), ('id_unc@0.05', 3.0),
                     ('ood_unc@0.05', 40.0)]) for spec in dm.SETS.values()])
    dm.write_tsv(os.path.join(tmp, 'rho.tsv'), [
        OrderedDict([('set', key), ('population', pop),
                     ('delta', dm.delta_key(d)), ('statistic', s),
                     ('target', t), ('rho', float(rng.uniform(-1, 1))),
                     ('n', 24)])
        for key in dm.SETS for pop in ('single', 'all') for d in dm.DELTAS
        for t in dm.TARGETS for s in dm.STATISTICS])


def test_figures_are_written():
    with tempfile.TemporaryDirectory() as tmp:
        _fake_divided_dir(tmp)
        proc = subprocess.run([sys.executable, SCRIPT, tmp, '--top', '2'],
                              capture_output=True, text=True, cwd=_REPO_ROOT)
        assert proc.returncode == 0, proc.stderr
        for name in ('bubble_singletons_0.05', 'bubble_robust_0.05',
                     'rho_vs_threshold'):
            for ext in ('.pdf', '.png'):
                path = os.path.join(tmp, name + ext)
                assert os.path.getsize(path) > 1000, path
        proc = subprocess.run([sys.executable, SCRIPT, tmp, '--threshold',
                               '0.07'], capture_output=True, text=True,
                              cwd=_REPO_ROOT)
        assert proc.returncode == 2 and '--threshold' in proc.stderr
    print('test_figures_are_written passed')


def test_labels_do_not_overlap_when_there_is_room():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    import plot_divided_mass as pdm
    fig, ax = plt.subplots(figsize=(6, 5))
    xy = [(1, 1), (3, 30), (10, 3), (30, 10), (1, 30), (30, 1)]
    points = [dict(x=x, y=y, value=5 + 3 * i, text=f'class {i}')
              for i, (x, y) in enumerate(xy)]
    items = pdm.bubble_panel(ax, points, (0.3, 100), (0.3, 100))
    pdm.place_labels(ax, items)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [pdm._box(Text.get_window_extent(t, renderer)) for t in ax.texts]
    assert len(boxes) == len(points)
    clashes = sum(pdm._overlap(a, b) > 0
                  for i, a in enumerate(boxes) for b in boxes[i + 1:])
    assert clashes == 0, clashes
    plt.close(fig)
    print('test_labels_do_not_overlap_when_there_is_room passed')


if __name__ == '__main__':
    test_figures_are_written()
    test_labels_do_not_overlap_when_there_is_room()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$ENVPY -m pytest -q tests/test_plot_divided_mass.py`
Expected: FAIL (the script does not exist yet, so the subprocess returns 2 or the import fails).

- [ ] **Step 3: Implement** — create `tools/plot_divided_mass.py`:

```python
#!/usr/bin/env python
"""Figures of tools/divided_mass.py: bubble charts and correlation curves.

Reads the tables of tools/divided_mass.py (default
work_dirs/p3former_2xb1_3x_dso_ood_dump/divided_mass) and writes, next to
them, each figure as a PDF (vector) and a PNG (300 dpi) in the style of the
paper figures: serif text, Okabe-Ito blue for a gain and orange for a drop,
bubble area proportional to |improvement|, log-log axes, one panel per set
in a 2 x 2 grid whose fourth cell holds the legends.

    bubble_singletons_<delta>   ID divided % (x) against OOD divided % (y) at
                                --threshold for the 24 single-class splits,
                                with flat MSP at the same threshold (star) and
                                the iso-ratio diagonal through it: a split
                                above it has a higher OOD:ID ratio than flat
    bubble_robust_<delta>       the same axes for the robust splits
                                (improvement > 0 on every set) over every
                                other split in grey; the --top robust splits
                                by worst-set improvement carry their names
                                (their classes are in robust.tsv)
    rho_vs_threshold            Spearman rho of each divided statistic
                                against each metric delta, over the
                                thresholds

Run from the repo root:
    python tools/plot_divided_mass.py [DIVIDED_DIR] [--threshold 0.05] [--top 8]
"""
import argparse
import math
import os.path as osp
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import divided_mass as dm  # noqa: E402

STYLE = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8,
    'axes.titlesize': 8.5,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.linewidth': 0.6,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.minor.width': 0.4,
    'ytick.minor.width': 0.4,
    'pdf.fonttype': 42,  # embedded TrueType fonts (camera-ready requirement)
    'ps.fonttype': 42,
}
GAIN, DROP, NA = '#0072B2', '#D55E00', '#8C8C8C'  # Okabe-Ito
GAIN_TXT, DROP_TXT = '#00507D', '#A34700'  # darker shades for label text
AREA_PER_UNIT = 12  # marker area (pt^2) per unit of |improvement|
SIZE_KEY = (5, 20, 40)
STAT_STYLE = OrderedDict([  # rho_vs_threshold: one colour per statistic
    ('ood_div', ('#0072B2', 'OOD divided %')),
    ('id_div', ('#D55E00', 'ID divided %')),
    ('log_ratio', ('#009E73', 'log OOD/ID divided ratio')),
    ('precision', ('#CC79A7', 'divided precision')),
])
TARGET_LABELS = OrderedDict([
    ('d_auroc', r'$\Delta$AUROC'), ('d_ap', r'$\Delta$AP'),
    ('d_fpr95', r'$\Delta$FPR@95'), ('improvement', 'improvement')])
# Label candidates: 8 directions, then the same further out with a leader.
LABEL_ANGLES = (0, 180, 90, 270, 45, 135, 315, 225)
LABEL_GAPS = (0.0, 7.0, 14.0)  # extra points beyond the bubble edge


def tint(colour, k=0.35):
    """Opaque light version of ``colour`` (k = share of the colour)."""
    return tuple(k * np.array(to_rgb(colour)) + (1 - k))


def area(value):
    return AREA_PER_UNIT * abs(value)


def signed(value):
    return f'{value:+.1f}'.replace('-', '−')  # typographic minus


def save(fig, stem):
    fig.savefig(stem + '.pdf')
    fig.savefig(stem + '.png', dpi=300)
    plt.close(fig)
    print(f'saved {stem}.pdf / .png')


def log_limits(values, pad=1.6):
    """(low, high) of a log axis around the positive finite ``values``."""
    pos = [v for v in values if v > 0 and math.isfinite(v)]
    if not pos:
        return 0.01, 100.0
    return min(pos) / pad, max(pos) * pad


def format_log_axes(ax, xlim, ylim):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_formatter(NullFormatter())
    ax.grid(which='major', color='0.88', lw=0.5)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------- labels
def _box(extent):
    return (extent.x0, extent.y0, extent.x1, extent.y1)


def _overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _outside(box, frame):
    return (box[2] - box[0]) * (box[3] - box[1]) - _overlap(box, frame)


def _annotate(ax, item, angle, gap, fontsize, leader):
    th = math.radians(angle)
    dist = item['radius'] + 3.0 + gap
    dx, dy = dist * math.cos(th), dist * math.sin(th)
    ha = 'left' if dx > 0.3 * dist else 'right' if dx < -0.3 * dist else 'center'
    va = 'bottom' if dy > 0.3 * dist else 'top' if dy < -0.3 * dist else 'center'
    arrow = (dict(arrowstyle='-', lw=0.4, color=item['colour'], shrinkA=0,
                  shrinkB=item['radius']) if leader else None)
    return ax.annotate(item['text'], (item['x'], item['y']), xytext=(dx, dy),
                       textcoords='offset points', ha=ha, va=va,
                       multialignment=ha, color=item['colour'],
                       fontsize=fontsize, linespacing=1.05, zorder=10,
                       arrowprops=arrow, annotation_clip=False)


def place_labels(ax, items, fontsize=6.5):
    """Label every item (dict: x, y in data units, text, colour, radius in
    points), largest bubble first, at the candidate position -- 8 directions
    at 3 distances -- whose text box overlaps the placed labels, the other
    bubbles and the outside of the axes least. A label pushed away from its
    bubble gets a thin leader line."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    bubbles = []
    for item in items:
        cx, cy = ax.transData.transform((item['x'], item['y']))
        r = item['radius'] * px
        bubbles.append((cx - r, cy - r, cx + r, cy + r))
    frame = _box(ax.get_window_extent(renderer))
    placed = []
    for i in sorted(range(len(items)), key=lambda j: -items[j]['radius']):
        best = None
        for gap in LABEL_GAPS:
            for angle in LABEL_ANGLES:
                ann = _annotate(ax, items[i], angle, gap, fontsize, False)
                box = _box(Text.get_window_extent(ann, renderer))
                ann.remove()
                cost = (sum(_overlap(box, b) for b in placed)
                        + sum(_overlap(box, b) for j, b in enumerate(bubbles)
                              if j != i)
                        + 10.0 * _outside(box, frame))
                if best is None or cost < best[0]:
                    best = (cost, angle, gap)
            if best[0] == 0.0:
                break
        _, angle, gap = best
        ann = _annotate(ax, items[i], angle, gap, fontsize, leader=gap > 0)
        placed.append(_box(Text.get_window_extent(ann, renderer)))


# ---------------------------------------------------------------- bubbles
def bubble_panel(ax, points, xlim, ylim, star=None, background=None):
    """Draw one panel. points: dicts with x, y (percent; 0 is drawn on the
    axis floor, hollow), value (improvement) and text (None: no label);
    star: flat MSP (x, y); background: (x, y) pairs drawn as grey dots.
    Returns the label items for :func:`place_labels`."""
    format_log_axes(ax, xlim, ylim)
    if background:
        ax.scatter([max(x, xlim[0]) for x, _ in background],
                   [max(y, ylim[0]) for _, y in background],
                   s=4, color='0.78', linewidths=0, zorder=1)
    if star is not None:
        x0, y0 = star
        xs = np.array(xlim)
        ax.plot(xs, xs * (y0 / x0), ls=(0, (4, 3)), lw=0.7, color='0.35',
                zorder=2)
        ax.scatter([x0], [y0], marker='*', s=70, color='black', zorder=20)
    items = []
    for z, pt in enumerate(sorted(points, key=lambda p: -abs(p['value']))):
        colour, text_colour = ((GAIN, GAIN_TXT) if pt['value'] > 0
                               else (DROP, DROP_TXT))
        zero = pt['x'] <= 0 or pt['y'] <= 0
        x, y = max(pt['x'], xlim[0]), max(pt['y'], ylim[0])
        ax.scatter([x], [y], s=area(pt['value']),
                   facecolor='none' if zero else tint(colour),
                   edgecolor=colour, linewidths=1.0, zorder=3 + z,
                   clip_on=False)
        if pt.get('text'):
            items.append(dict(x=x, y=y, colour=text_colour,
                              text=f'{pt["text"]}\n{signed(pt["value"])}',
                              radius=math.sqrt(area(pt['value'])) / 2))
    return items


def _bubble_handle(colour, value, face=True):
    return Line2D([], [], ls='none', marker='o',
                  markersize=math.sqrt(area(value)),
                  markerfacecolor=tint(colour) if face else 'none',
                  markeredgecolor=colour, markeredgewidth=1.0)


def legend_cell(ax, star=False, background=False, zeros=False):
    """The fourth cell of the grid: colour, marker and size keys."""
    ax.axis('off')
    handles = [_bubble_handle(GAIN, 10), _bubble_handle(DROP, 10)]
    labels = ['improvement > 0', 'improvement < 0']
    if zeros:
        handles.append(_bubble_handle(NA, 10, face=False))
        labels.append('0 %, drawn on the axis')
    if star:
        handles += [Line2D([], [], ls='none', marker='*', markersize=9,
                           color='black'),
                    Line2D([], [], ls=(0, (4, 3)), lw=0.7, color='0.35')]
        labels += ['flat MSP at the same threshold',
                   'same OOD:ID ratio as flat MSP']
    if background:
        handles.append(Line2D([], [], ls='none', marker='o', markersize=2.5,
                              color='0.78'))
        labels.append('other splits')
    first = ax.legend(handles, labels, loc='upper left', frameon=False,
                      handletextpad=0.5, borderaxespad=0.2)
    ax.add_artist(first)
    ax.legend([_bubble_handle('0.4', v, face=False) for v in SIZE_KEY],
              [str(v) for v in SIZE_KEY], title='|improvement| (bubble area)',
              loc='lower left', ncol=len(SIZE_KEY), frameon=False,
              handlelength=3.2, handleheight=3.2, columnspacing=1.0,
              borderaxespad=0.2)


def bubble_grid(panels, xlabel, ylabel, stem, stars=None, backgrounds=None):
    """2 x 2 figure: one bubble panel per set (panels: label -> points) and
    a legend cell; the log axes are the same in every panel."""
    stars, backgrounds = stars or {}, backgrounds or {}
    xs = [p['x'] for pts in panels.values() for p in pts]
    ys = [p['y'] for pts in panels.values() for p in pts]
    for x, y in stars.values():
        xs.append(x)
        ys.append(y)
    for points in backgrounds.values():
        xs += [x for x, _ in points]
        ys += [y for _, y in points]
    xlim, ylim = log_limits(xs), log_limits(ys)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.08,
                        wspace=0.22, hspace=0.3)
    cells = (axes[0, 0], axes[0, 1], axes[1, 0])
    labelled, zeros = [], False
    for i, (ax, (label, points)) in enumerate(zip(cells, panels.items())):
        labelled.append((ax, bubble_panel(ax, points, xlim, ylim,
                                          stars.get(label),
                                          backgrounds.get(label))))
        zeros = zeros or any(p['x'] <= 0 or p['y'] <= 0 for p in points)
        ax.set_title(f'({"abc"[i]}) {label}', loc='left')
        ax.set_xlabel(xlabel)
        if i != 1:
            ax.set_ylabel(ylabel)
    for ax, items in labelled:
        place_labels(ax, items)
    legend_cell(axes[1, 1], star=bool(stars), background=bool(backgrounds),
                zeros=zeros)
    save(fig, stem)


# ---------------------------------------------------------------- figures
def load(divided_dir):
    tables = OrderedDict((key, dm.read_tsv(osp.join(divided_dir, f'{key}.tsv')))
                         for key in dm.SETS)
    labels = OrderedDict((key, spec['label']) for key, spec in dm.SETS.items())
    flat = dm.read_tsv(osp.join(divided_dir, 'flat.tsv'))
    rho = dm.read_tsv(osp.join(divided_dir, 'rho.tsv'))
    return tables, labels, flat, rho


def flat_stars(flat, threshold):
    key = dm.delta_key(threshold)
    return {row['set']: (row[f'id_unc@{key}'], row[f'ood_unc@{key}'])
            for row in flat}


def singleton_panels(tables, labels, threshold):
    key = dm.delta_key(threshold)
    return OrderedDict(
        (labels[k], [dict(x=r[f'id_div@{key}'], y=r[f'ood_div@{key}'],
                          value=r['improvement'], text=r['group_A'])
                     for r in rows if r['size_A'] == 1])
        for k, rows in tables.items())


def robust_panels(tables, labels, threshold, top):
    """Robust splits as bubbles (the ``top`` by worst-set improvement named)
    and every other split as a grey background dot."""
    key = dm.delta_key(threshold)
    worst = {}
    for rows in tables.values():
        for r in rows:
            if r['robust']:
                worst[r['name']] = min(worst.get(r['name'], float('inf')),
                                       r['improvement'])
    named = set(sorted(worst, key=lambda n: -worst[n])[:top])
    panels, backgrounds = OrderedDict(), OrderedDict()
    for k, rows in tables.items():
        panels[labels[k]] = [
            dict(x=r[f'id_div@{key}'], y=r[f'ood_div@{key}'],
                 value=r['improvement'],
                 text=r['name'] if r['name'] in named else None)
            for r in rows if r['robust']]
        backgrounds[labels[k]] = [(r[f'id_div@{key}'], r[f'ood_div@{key}'])
                                  for r in rows if not r['robust']]
    return panels, backgrounds


def rho_figure(rho, labels, stem):
    """Spearman rho against the threshold: one row per target, one column
    per set; solid = single-class splits, dashed = all splits."""
    fig, axes = plt.subplots(len(TARGET_LABELS), len(labels),
                             figsize=(7.0, 8.0), sharex=True, sharey=True,
                             squeeze=False)
    for i, (target, target_label) in enumerate(TARGET_LABELS.items()):
        for j, (key, set_label) in enumerate(labels.items()):
            ax = axes[i, j]
            for stat, (colour, _) in STAT_STYLE.items():
                for population, ls in (('single', '-'), ('all', '--')):
                    pts = sorted(
                        (float(r['delta']), r['rho']) for r in rho
                        if r['set'] == key and r['target'] == target
                        and r['statistic'] == stat
                        and r['population'] == population)
                    ax.plot([d for d, _ in pts], [v for _, v in pts], ls=ls,
                            color=colour, lw=1.0, marker='o', markersize=2.2)
            ax.axhline(0.0, color='0.5', lw=0.5)
            ax.set_xscale('log')
            ax.set_ylim(-1.05, 1.05)
            ax.grid(which='major', color='0.9', lw=0.4)
            if i == 0:
                ax.set_title(set_label)
            if j == 0:
                ax.set_ylabel(r'$\rho$ vs ' + target_label)
            if i == len(TARGET_LABELS) - 1:
                ax.set_xlabel(r'threshold $\delta$')
    handles = [Line2D([], [], color=c, lw=1.2) for c, _ in STAT_STYLE.values()]
    handles += [Line2D([], [], color='0.3', ls='-', lw=1.0),
                Line2D([], [], color='0.3', ls='--', lw=1.0)]
    names = [n for _, n in STAT_STYLE.values()] + ['single-class splits',
                                                   'all splits']
    fig.legend(handles, names, loc='lower center', ncol=3, frameon=False)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.12,
                        hspace=0.25, wspace=0.12)
    save(fig, stem)


def main():
    choices = ', '.join(dm.delta_key(d) for d in dm.DELTAS)
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('divided_dir', nargs='?',
                    default=osp.join(dm.ROOT, 'divided_mass'),
                    help='output directory of tools/divided_mass.py; the '
                    'figures are written there')
    ap.add_argument('--threshold', type=float, default=dm.HEADLINE,
                    help=f'divided threshold of the bubble charts: {choices}')
    ap.add_argument('--top', type=int, default=8,
                    help='robust splits named in bubble_robust')
    args = ap.parse_args()
    if args.threshold not in dm.DELTAS:
        ap.error(f'--threshold must be one of {choices}')
    plt.rcParams.update(STYLE)
    tables, labels, flat, rho = load(args.divided_dir)
    key = dm.delta_key(args.threshold)
    xlabel = f'ID divided at $\\delta$ = {key} (%)'
    ylabel = f'OOD divided at $\\delta$ = {key} (%)'
    stars = flat_stars(flat, args.threshold)
    bubble_grid(singleton_panels(tables, labels, args.threshold), xlabel,
                ylabel, osp.join(args.divided_dir, f'bubble_singletons_{key}'),
                stars=stars)
    panels, backgrounds = robust_panels(tables, labels, args.threshold,
                                        args.top)
    bubble_grid(panels, xlabel, ylabel,
                osp.join(args.divided_dir, f'bubble_robust_{key}'),
                stars=stars, backgrounds=backgrounds)
    rho_figure(rho, labels, osp.join(args.divided_dir, 'rho_vs_threshold'))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$ENVPY -m pytest -q tests/test_plot_divided_mass.py && $ENVPY tests/test_plot_divided_mass.py`
Expected: every test passes. Then open one PNG the test wrote to confirm the bubbles, labels, star and legend render. Copy the file out first, e.g. temporarily point the test at a scratch directory, or run the script on `_fake_divided_dir` output yourself.

- [ ] **Step 5: Full suite, then commit**

```bash
$ENVPY -m pytest -q tests
git add tools/plot_divided_mass.py tests/test_plot_divided_mass.py
git commit -m "$(cat <<'EOF'
Plot the divided mass as bubble charts and correlation curves

Bubble charts of the single-class and the robust splits (ID vs OOD
divided share, area = |improvement|, flat MSP star and iso-ratio line,
automatic label placement) and Spearman rho against the threshold.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `tools/extract_point_features.py`

**Files:**
- Create: `tools/extract_point_features.py`
- Test: `tests/test_extract_point_features.py`

**Interfaces:**
- Consumes: the mmengine / mmdet3d model build (at run time only); none of the earlier tasks.
- Produces (used by Tasks 5, 7):
  - one `f<frame:06d>.npz` per frame in `--out-dir` with keys `feat` (f16 [n, 256]), `pos` (f16 [n, 256]), `logits` (f16 [n, 24]), `label` (int16), `raw` (int16), `ood` (bool), `weight` (float32), `index` (int32), `lidar_path` (str) and `frame` (int);
  - `meta.json`;
  - functions `FeatureCapture(head)` (context manager), `sample_frame(label, ood, per_class, ood_per_frame, rng)`, `frame_record(capture, data_sample, weight, per_class, ood_per_frame, rng, frame)`, `check_logits(feat, logits, weight)`, `index_dump(dump_dir)`, `compare_with_dump(record, dump_file)`, `extract(...)`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_extract_point_features.py`:

```python
"""Tests for tools/extract_point_features.py (stand-in head, no model).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_extract_point_features.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import extract_point_features as epf  # noqa: E402


def test_sample_frame():
    label = np.full(2000, epf.IGNORE_INDEX)
    label[:100] = 0
    label[100:110] = 5
    ood = np.zeros(2000, bool)
    ood[500:1500] = True  # OOD points carry the ignore label
    index, weight = epf.sample_frame(label, ood, 16, 64,
                                     np.random.default_rng([0, 3]))
    assert len(index) == len(set(index.tolist())) == 16 + 10 + 64
    assert (label[index] == 0).sum() == 16 and (label[index] == 5).sum() == 10
    assert ood[index].sum() == 64
    # the weights restore the stratum sizes
    assert np.isclose(weight[label[index] == 0].sum(), 100)
    assert np.isclose(weight[label[index] == 5].sum(), 10)
    assert np.isclose(weight[ood[index]].sum(), 1000)
    # ignored points that are not OOD are never drawn
    assert not np.any((label[index] == epf.IGNORE_INDEX) & ~ood[index])
    again, _ = epf.sample_frame(label, ood, 16, 64,
                                np.random.default_rng([0, 3]))
    assert np.array_equal(index, again)
    # a frame without OOD points, and one without any valid point
    index, _ = epf.sample_frame(label, np.zeros(2000, bool), 16, 64,
                                np.random.default_rng(1))
    assert len(index) == 26
    index, weight = epf.sample_frame(np.full(5, epf.IGNORE_INDEX),
                                     np.zeros(5, bool), 16, 64,
                                     np.random.default_rng(1))
    assert index.shape == (0, ) and weight.shape == (0, )
    print('test_sample_frame passed')


class _Head(torch.nn.Module):
    """Stand-in for _P3FormerHead: init_inputs returns (queries,
    pe_features, mpe, sem_preds) with sem_preds = pe_features @ W^T."""

    def __init__(self, feat, pos, weight):
        super().__init__()
        self.sem_queries = torch.nn.Conv3d(feat.shape[1], weight.shape[0], 1,
                                           bias=False)
        with torch.no_grad():
            self.sem_queries.weight.copy_(
                weight.reshape(self.sem_queries.weight.shape))
        self._feat, self._pos = feat, pos

    def init_inputs(self, features, voxel_coors, batch_size):
        w = self.sem_queries.weight.reshape(self.sem_queries.weight.shape[0], -1)
        return [None], [self._feat], [self._pos], [self._feat @ w.t()]

    def forward(self):
        return self.init_inputs(None, None, 1)


def _data_sample(p2v, label, raw):
    return SimpleNamespace(
        gt_pts_seg=SimpleNamespace(point2voxel_map=torch.from_numpy(p2v)),
        eval_ann_info=dict(
            pts_instance_mask=(np.arange(len(raw)) << 16) | raw,
            pts_semantic_mask=label),
        metainfo=dict(lidar_path='seq/000001.ply'))


def test_capture_and_frame_record():
    torch.manual_seed(0)
    voxels, dim = 50, 16
    feat, pos = torch.randn(voxels, dim), torch.randn(voxels, dim)
    weight = torch.randn(25, dim)  # 24 classes + the ignore channel
    head = _Head(feat, pos, weight)
    rng = np.random.RandomState(0)
    n = 400
    p2v = rng.randint(0, voxels, n)
    raw = rng.choice([1, 3, 17, 28, 40], n)
    label = np.where(np.isin(raw, [17, 28, 40]), 24, rng.randint(0, 24, n))
    sample = _data_sample(p2v, label, raw)
    with epf.FeatureCapture(head) as capture:
        head.forward()
        assert capture.feat[0] is feat and capture.pos[0] is pos
        record, diff = epf.frame_record(capture, sample,
                                        epf.classifier_weight(head), 8, 32,
                                        np.random.default_rng(0), frame=7)
    assert 'init_inputs' not in head.__dict__  # the wrapper is gone
    idx = record['index']
    assert np.allclose(record['feat'], feat[p2v[idx]].numpy(), atol=1e-2)
    assert np.allclose(record['pos'], pos[p2v[idx]].numpy(), atol=1e-2)
    want = (feat @ weight[:24].t())[p2v[idx]].numpy()
    assert np.allclose(record['logits'].astype(np.float32), want, atol=2e-2)
    assert np.array_equal(record['raw'], raw[idx])
    assert np.array_equal(record['ood'], np.isin(raw[idx], [17, 28]))
    assert np.array_equal(record['label'], label[idx])
    assert record['frame'] == 7 and record['lidar_path'] == 'seq/000001.ply'
    assert diff < 1e-4
    # the wrong tensor captured: the logits check raises
    capture.logits = [torch.randn(voxels, 25)]
    try:
        epf.frame_record(capture, sample, epf.classifier_weight(head), 8, 32,
                         np.random.default_rng(0), frame=7)
    except RuntimeError as err:
        assert 'reproduce the logits' in str(err)
    else:
        raise AssertionError('mismatched logits accepted')
    print('test_capture_and_frame_record passed')


def test_dump_comparison():
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.RandomState(1)
        n = 300
        logits = rng.randn(n, 24).astype(np.float16)
        mapped = rng.randint(0, 25, n).astype(np.int16)
        ood = mapped == 24
        path = os.path.join(tmp, 'r0_000000.npz')
        np.savez(path, logits=logits, ood=ood, valid=np.ones(n, bool),
                 mapped=mapped, lidar_path='seq/000001.ply')
        index = np.array([3, 10, 200], np.int32)
        record = dict(index=index, logits=logits[index], label=mapped[index],
                      ood=ood[index], frame=0, lidar_path='seq/000001.ply')
        assert epf.index_dump(tmp) == {'seq/000001.ply': path}
        assert epf.compare_with_dump(record, path) == 0.0
        for key, bad in (('logits', logits[index] + np.float16(0.5)),
                         ('label', (mapped[index] + 1) % 24),
                         ('ood', ~ood[index])):
            try:
                epf.compare_with_dump(dict(record, **{key: bad}), path)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f'mismatched {key} accepted')
    print('test_dump_comparison passed')


def test_out_dir_must_be_empty():
    with tempfile.TemporaryDirectory() as tmp:
        open(os.path.join(tmp, 'f000000.npz'), 'w').close()
        try:
            epf.extract('config.py', 'checkpoint.pth', tmp)
        except FileExistsError as err:
            assert tmp in str(err)
        else:
            raise AssertionError('non-empty out dir accepted')
    print('test_out_dir_must_be_empty passed')


if __name__ == '__main__':
    test_sample_frame()
    test_capture_and_frame_record()
    test_dump_comparison()
    test_out_dir_must_be_empty()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$ENVPY -m pytest -q tests/test_extract_point_features.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'extract_point_features'`.

- [ ] **Step 3: Implement** — create `tools/extract_point_features.py`:

```python
#!/usr/bin/env python
"""Sample per-point penultimate features of P3Former for offline analysis.

The auxiliary semantic classifier of _P3FormerHead is a bias-free linear
layer on ``pe_features``: the per-voxel 256-d backbone features after
``pe_conv`` plus the positional embedding ``mpe`` (cartesian + polar). Its
logits, sem_preds = pe_features @ sem_queries^T, are the ones every OOD
score and the logits dumps use. This tool runs the model on one split,
wraps ``decode_head.init_inputs`` on the model instance (the model code is
not changed) to keep ``pe_features``, ``mpe`` and ``sem_preds`` of each
frame, maps them to points with ``point2voxel_map`` and writes a
stratified sample of every frame: up to --per-class ID points of each
mapped class and up to --ood-per-frame OOD points (raw 17 Stop / 28
Others, as _OODPointMetric), each weighted by stratum size / points drawn,
so that weighted means over the samples estimate population means.

One ``f<frame>.npz`` per frame in --out-dir -- feat (float16 [n, 256]), pos
(float16 [n, 256], the mpe part of feat), logits (float16 [n, 24]), label
(int16 mapped train id, 24 for OOD), raw (int16 raw semantic id), ood
(bool), weight (float32), index (int32 point index in the frame),
lidar_path, frame -- plus meta.json. Checks: on every frame feat @ W^T
(W = sem_queries.weight) must reproduce the captured logits (float64
reference; tolerance 1e-2 * max(1, max |z|), room for the TF32 matmuls of
test.py's numerics, which also wrote the logits dumps); with --check-dump
the sampled logits, labels and OOD flags must equal those of a logits dump
at the same points (frames matched by lidar_path).

Run from the repo root (one GPU, ~25 GB; Cetran ~10 min, test ~25 min):
    CUDA_VISIBLE_DEVICES=0 python tools/extract_point_features.py \\
        configs/p3former/p3former_2xb1_3x_dso_ood.py \\
        work_dirs/p3former_2xb1_3x_dso/epoch_36.pth \\
        --ann dso_infos_cetran.pkl \\
        --out-dir work_dirs/p3former_2xb1_3x_dso_ood_dump/features_cetran \\
        --check-dump work_dirs/p3former_2xb1_3x_dso_ood_dump/logits
"""
import argparse
import glob
import json
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

NUM_CLASSES = 24  # DSO train classes; logit channel 24 is the ignore slot
IGNORE_INDEX = 24
OOD_RAW_IDS = (17, 28)  # Stop, Others: never trained
SEG_OFFSET = 2**16
LOGIT_TOL = 1e-2  # |feat @ W^T - logits| <= LOGIT_TOL * max(1, max |z|)
DUMP_TOL = 0.05  # sampled logits vs a logits dump, both float16


class FeatureCapture:
    """Keep the per-voxel tensors of the last ``init_inputs`` call of a
    decode head: ``feat`` (pe_features), ``pos`` (mpe) and ``logits``
    (sem_preds), each a list over the batch. Use as a context manager: the
    wrapper lives on the instance and is removed on exit."""

    def __init__(self, head):
        self.head = head
        self.feat = self.pos = self.logits = None

    def __enter__(self):
        original = self.head.init_inputs

        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            _, self.feat, self.pos, self.logits = out
            return out

        self.head.init_inputs = wrapped
        return self

    def __exit__(self, *exc):
        del self.head.init_inputs  # the class method is visible again
        return False


def sample_frame(label, ood, per_class, ood_per_frame, rng):
    """Stratified sample of one frame's valid points: up to ``per_class``
    ID points of every mapped class, then up to ``ood_per_frame`` OOD
    points, uniformly without replacement within each stratum. Returns
    (index int64 [n], weight float32 [n]), weight = stratum size / drawn."""
    strata = [np.flatnonzero((label == c) & ~ood) for c in range(NUM_CLASSES)]
    strata.append(np.flatnonzero(ood))
    caps = [per_class] * NUM_CLASSES + [ood_per_frame]
    index, weight = [], []
    for members, cap in zip(strata, caps):
        n = min(cap, len(members))
        if n == 0:
            continue
        index.append(np.sort(rng.choice(members, n, replace=False)))
        weight.append(np.full(n, len(members) / n, np.float32))
    if not index:
        return np.zeros(0, np.int64), np.zeros(0, np.float32)
    return np.concatenate(index).astype(np.int64), np.concatenate(weight)


def frame_ground_truth(data_sample):
    """(mapped label, raw semantic id, ood) per point, as _OODPointMetric."""
    ann = data_sample.eval_ann_info
    raw = np.asarray(ann['pts_instance_mask']) % SEG_OFFSET
    label = np.asarray(ann['pts_semantic_mask'])
    return label, raw, np.isin(raw, OOD_RAW_IDS)


def classifier_weight(head):
    """The semantic classifier's rows of the used classes, [NUM_CLASSES, C]."""
    w = head.sem_queries.weight.detach()
    return w.reshape(w.shape[0], -1)[:NUM_CLASSES]


def check_logits(feat, logits, weight):
    """Raise unless feat @ weight^T (float64) reproduces the captured
    logits; returns the largest |difference|."""
    if not len(feat):
        return 0.0
    ref = feat.double() @ weight.double().t()
    diff = float((ref - logits.double()).abs().max())
    scale = max(1.0, float(logits.abs().max()))
    if diff > LOGIT_TOL * scale:
        raise RuntimeError(
            'the captured features do not reproduce the logits (max |diff| '
            f'{diff:.4g} at scale {scale:.4g}): does init_inputs still return '
            '(queries, pe_features, mpe, sem_preds)?')
    return diff


def frame_record(capture, data_sample, weight, per_class, ood_per_frame, rng,
                 frame):
    """The sampled points of one frame (batch size 1) as a dict of arrays,
    and the logit-reconstruction error."""
    label, raw, ood = frame_ground_truth(data_sample)
    p2v = data_sample.gt_pts_seg.point2voxel_map.long()
    if p2v.shape[0] != label.shape[0]:
        raise ValueError(f'point2voxel_map has {p2v.shape[0]} points, the '
                         f'ground truth {label.shape[0]}')
    index, w = sample_frame(label, ood, per_class, ood_per_frame, rng)
    voxels = p2v[torch.from_numpy(index).to(p2v.device)]
    feat = capture.feat[0][voxels].float()
    pos = capture.pos[0][voxels].float()
    logits = capture.logits[0][voxels, :NUM_CLASSES].float()
    diff = check_logits(feat.cpu(), logits.cpu(), weight.cpu())
    record = dict(
        feat=feat.half().cpu().numpy(), pos=pos.half().cpu().numpy(),
        logits=logits.half().cpu().numpy(),
        label=label[index].astype(np.int16), raw=raw[index].astype(np.int16),
        ood=ood[index], weight=w, index=index.astype(np.int32),
        lidar_path=str(data_sample.metainfo.get('lidar_path', '')),
        frame=frame)
    return record, diff


def index_dump(dump_dir):
    """lidar_path -> npz file of a logits dump (_OODLogitsDumpMetric)."""
    files = sorted(glob.glob(osp.join(dump_dir, '*.npz')))
    if not files:
        raise FileNotFoundError(f'no .npz dumps in {dump_dir}')
    out = {}
    for path in files:
        with np.load(path) as data:
            out[str(data['lidar_path'])] = path
    return out


def compare_with_dump(record, dump_file):
    """Raise unless the record's logits, labels and OOD flags equal the
    dump's at the same points; returns the largest logit |difference|."""
    idx = record['index']
    with np.load(dump_file) as data:
        logits = data['logits'][idx].astype(np.float32)
        mapped, ood = data['mapped'][idx], data['ood'][idx]
    diff = (float(np.abs(logits - record['logits'].astype(np.float32)).max())
            if len(idx) else 0.0)
    same_label = np.array_equal(mapped, record['label'])
    same_ood = np.array_equal(ood, record['ood'])
    if diff > DUMP_TOL or not same_label or not same_ood:
        raise RuntimeError(
            f'frame {record["frame"]} ({record["lidar_path"]}) differs from '
            f'{dump_file}: max logit |diff| {diff:.4g}, labels equal '
            f'{same_label}, OOD flags equal {same_ood}')
    return diff


def build(config, checkpoint, ann=None, workers=4, device='cuda:0'):
    """(model, dataloader) as test.py builds them, with batch size 1."""
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner
    from mmengine.runner.checkpoint import load_checkpoint

    from mmdet3d.registry import MODELS

    cfg = Config.fromfile(config)  # also imports cfg.custom_imports
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    if ann:
        cfg.test_dataloader.dataset.dataset.ann_file = ann
    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = workers
    model = MODELS.build(cfg.model)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device).eval()
    return model, Runner.build_dataloader(cfg.test_dataloader)


def extract(config, checkpoint, out_dir, ann=None, per_class=64,
            ood_per_frame=512, seed=0, limit=0, check_dump=None, workers=4,
            device='cuda:0'):
    """Write one sample file per frame and meta.json; returns the meta."""
    os.makedirs(out_dir, exist_ok=True)
    if os.listdir(out_dir):
        raise FileExistsError(f'{out_dir} is not empty: delete its files or '
                              'choose another --out-dir')
    dump = index_dump(check_dump) if check_dump else None
    model, loader = build(config, checkpoint, ann, workers, device)
    weight = classifier_weight(model.decode_head)
    stats = dict(frames=0, points=0, ood_samples=0,
                 samples_per_class=[0] * NUM_CLASSES, max_logit_diff=0.0,
                 max_dump_diff=None if dump is None else 0.0)
    t0 = time.time()
    with FeatureCapture(model.decode_head) as capture, torch.no_grad():
        for frame, data in enumerate(loader):
            if limit and frame >= limit:
                break
            (sample, ) = model.test_step(data)
            record, diff = frame_record(capture, sample, weight, per_class,
                                        ood_per_frame,
                                        np.random.default_rng([seed, frame]),
                                        frame)
            stats['max_logit_diff'] = max(stats['max_logit_diff'], diff)
            if dump is not None:
                if record['lidar_path'] not in dump:
                    raise KeyError(f'frame {frame} ({record["lidar_path"]}) is '
                                   f'not in the dump {check_dump}')
                stats['max_dump_diff'] = max(
                    stats['max_dump_diff'],
                    compare_with_dump(record, dump[record['lidar_path']]))
            np.savez(osp.join(out_dir, f'f{frame:06d}.npz'), **record)
            per_class_count = np.bincount(record['label'][~record['ood']],
                                          minlength=NUM_CLASSES)[:NUM_CLASSES]
            stats['samples_per_class'] = [
                a + int(b) for a, b in zip(stats['samples_per_class'],
                                           per_class_count)]
            stats['ood_samples'] += int(record['ood'].sum())
            stats['points'] += int(len(record['index']))
            stats['frames'] += 1
            if stats['frames'] % 100 == 0:
                print(f'{stats["frames"]} frames ({time.time() - t0:.0f} s)',
                      flush=True)
    meta = dict(config=config, checkpoint=checkpoint, ann=ann,
                per_class=per_class, ood_per_frame=ood_per_frame, seed=seed,
                check_dump=check_dump, seconds=round(time.time() - t0),
                **stats)
    with open(osp.join(out_dir, 'meta.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='test config (as test.py)')
    ap.add_argument('checkpoint', help='checkpoint file')
    ap.add_argument('--ann', default=None,
                    help="info pkl of the split (default: the config's)")
    ap.add_argument('--out-dir', required=True,
                    help='empty directory for the per-frame samples')
    ap.add_argument('--per-class', type=int, default=64,
                    help='ID points sampled per mapped class and frame')
    ap.add_argument('--ood-per-frame', type=int, default=512,
                    help='OOD points sampled per frame')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0,
                    help='stop after this many frames (0 = all)')
    ap.add_argument('--check-dump', default=None, metavar='DIR',
                    help='logits dump of the same split to compare with')
    ap.add_argument('--workers', type=int, default=4,
                    help='dataloader worker processes')
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    meta = extract(args.config, args.checkpoint, args.out_dir, ann=args.ann,
                   per_class=args.per_class, ood_per_frame=args.ood_per_frame,
                   seed=args.seed, limit=args.limit,
                   check_dump=args.check_dump, workers=args.workers,
                   device=args.device)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$ENVPY -m pytest -q tests/test_extract_point_features.py && $ENVPY tests/test_extract_point_features.py`
Expected: every test passes.

- [ ] **Step 5: Smoke run on the 5-frame mini split, then commit**

GPU check first: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv`. Pick a GPU with at least 26 GB free.

```bash
SCRATCH=$(mktemp -d)
CUDA_VISIBLE_DEVICES=0 $ENVPY tools/extract_point_features.py \
    configs/p3former/p3former_2xb1_3x_dso_ood.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth \
    --ann dso_infos_mini.pkl --out-dir $SCRATCH/features_mini \
    --check-dump work_dirs/p3former_2xb1_3x_dso_ood_dump/logits_test
```

Expected: `meta.json` shows `"frames": 5`, `max_logit_diff` well below 1e-2 × the logit scale, and `max_dump_diff` ≤ 0.05; `ood_samples` > 0, because the mini frames contain Stop / Others. If `max_dump_diff` fails, stop and debug with superpowers:systematic-debugging before continuing. Record the three numbers for the DOCs entry, then `rm -r $SCRATCH`.

```bash
$ENVPY -m pytest -q tests
git add tools/extract_point_features.py tests/test_extract_point_features.py
git commit -m "$(cat <<'EOF'
Sample P3Former's penultimate features per point for offline analysis

tools/extract_point_features.py wraps decode_head.init_inputs at run time
(no model change), keeps pe_features, mpe and sem_preds, and writes a
stratified, weighted sample of every frame. Every frame checks that the
features reproduce the logits; --check-dump compares with a logits dump.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `tools/ood_class_resemblance.py`

**Files:**
- Create: `tools/ood_class_resemblance.py`
- Test: `tests/test_ood_class_resemblance.py`

**Interfaces:**
- Consumes:
  - Task 2: `dm.ROOT`, `dm.HEADLINE`, `dm.DELTAS`, `dm.delta_key`, `dm.read_tsv`, `dm.write_tsv`, `dm.md_table`, `dm.fmt`, `dm.pct`, `dm.ratio`, `dm.spearman`, and the `<set>.tsv` columns `name, group_A, ood_div@0.05, id_div@0.05, improvement, robust`.
  - Task 4: the feature-sample file format.
  - `sb.CLASSES`, `sb.NUM_CLASSES`, `sb.partition_name`, `sb.subsets_to_mask`, `sb.require_torch`.
- Produces (used by Tasks 6, 7):
  - constants `ROOT`, `SETS`, `SPACES`, `CLASS_PAIRS`, `SPLIT_PAIRS`;
  - functions `resolve_sets(root)`, `load_samples(feature_dirs)` (dict with `feat, pos, label, raw, ood, weight`), `space_features(samples, space, index)`, `select_reference`, `knn_class_counts`, `class_profile`, `query_weights`, `split_measures`, `subset_of`, `run(...) -> (profiles, split_tables, rho)`;
  - `profile_<set>_<space>.tsv` columns `class, bank, queries, r_ood, r_id, contrast, feat_div_ood, feat_div_id, ood_div, id_div, improvement`;
  - `splits_<set>_<space>.tsv` columns `name, size_A, group_A, R_A, E_A, feat_div_ood, feat_div_id, feat_log_ratio, ood_div, id_div, improvement, robust`;
  - `rho.tsv` columns `set, space, k, population, x, y, rho, n`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_ood_class_resemblance.py`:

```python
"""Tests for tools/ood_class_resemblance.py (synthetic features, CPU).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_ood_class_resemblance.py
"""
import os
import sys
import tempfile
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import divided_mass as dm  # noqa: E402
import ood_class_resemblance as res  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402

C = sb.NUM_CLASSES


def test_knn_shares_find_the_nearest_class():
    rng = np.random.RandomState(0)
    dim = 16
    centres = 3.0 * rng.randn(C, dim)
    bank = np.concatenate([centres[c] + 0.3 * rng.randn(50, dim)
                           for c in range(C)]).astype(np.float32)
    bank_label = np.repeat(np.arange(C), 50)
    ood = (centres[16] + 0.5 + 0.3 * rng.randn(200, dim)).astype(np.float32)
    counts = res.knn_class_counts(ood, bank, bank_label, k=10, device='cpu',
                                  chunk=64)
    assert counts.shape == (200, C) and np.all(counts.sum(axis=1) == 10)
    assert counts[:, 16].sum() > 0.8 * counts.sum()  # next to overhead-bridge
    q = ood[0] / np.linalg.norm(ood[0])
    b = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    top = np.argsort(-(b @ q))[:10]
    assert np.array_equal(np.bincount(bank_label[top], minlength=C), counts[0])
    try:
        res.knn_class_counts(ood, bank[:5], bank_label[:5], k=10,
                             device='cpu')
    except ValueError:
        pass
    else:
        raise AssertionError('k larger than the bank accepted')
    print('test_knn_shares_find_the_nearest_class passed')


def test_profile_weights_and_exclusion():
    k = 10
    ood_counts = np.zeros((2, C), np.int32)
    ood_counts[0, 3] = 10  # weight 3: all neighbours class 3
    ood_counts[1, 5] = 10  # weight 1: all neighbours class 5
    id_label = np.array([0, 0, 1, 1])
    id_counts = np.zeros((4, C), np.int32)
    id_counts[0, 3] = 10  # one class-0 query sits among class-3 points
    id_counts[1, 0] = 10
    id_counts[2:, 1] = 10
    population = np.zeros(C)
    population[0], population[1] = 90.0, 10.0
    id_weight = res.query_weights(id_label, population)
    assert np.allclose(id_weight, [45, 45, 5, 5])
    excluded = np.zeros(C, bool)
    excluded[23] = True
    r_ood, r_id, contrast = res.class_profile(
        ood_counts, np.array([3.0, 1.0]), id_counts, id_label, id_weight, k,
        excluded)
    assert np.isclose(r_ood[3], 75.0) and np.isclose(r_ood[5], 25.0)
    assert np.isclose(r_id[3], 45.0) and np.isclose(contrast[3], 75.0 / 45.0)
    assert r_id[5] == 0 and contrast[5] == float('inf')
    assert np.isnan(r_ood[23]) and np.isnan(r_id[23]) and np.isnan(contrast[23])
    print('test_profile_weights_and_exclusion passed')


def test_split_measures_match_brute_force():
    rng = np.random.RandomState(2)
    k = 10
    counts = rng.multinomial(k, np.ones(C) / C, size=300).astype(np.int32)
    counts[:50] = 0
    counts[:50, 7] = k  # pure neighbourhoods: never divided
    weight = rng.uniform(0.5, 2.0, 300)
    subsets = [(7, ), (0, 1, 2, 3, 4), tuple(range(12))]
    r_ood = 100.0 * rng.dirichlet(np.ones(C))
    R_A, E_A, fd_ood, fd_id = res.split_measures(
        sb.subsets_to_mask(subsets), r_ood, counts, weight, counts, weight, k,
        chunk=64)
    for s, subset in enumerate(subsets):
        n_a = counts[:, list(subset)].sum(axis=1)
        want = 100.0 * weight[(n_a > 0) & (n_a < k)].sum() / weight.sum()
        assert np.isclose(fd_ood[s], want) and np.isclose(fd_id[s], want)
        assert np.isclose(R_A[s], r_ood[list(subset)].sum())
        assert np.isclose(E_A[s], R_A[s] / (100.0 * len(subset) / C))
    print('test_split_measures_match_brute_force passed')


def test_reference_selection():
    label = np.concatenate([np.full(900, 0), np.full(20, 1), np.full(60, 2),
                            np.full(100, 24)])
    ood = label == 24
    bank, query, table = res.select_reference(
        label, ood, np.ones(len(label)), 500, 200, np.random.default_rng(0))
    assert not set(bank.tolist()) & set(query.tolist())
    assert table[0]['bank'] == 500 and table[0]['queries'] == 200
    assert table[1]['excluded'] == 1 and table[1]['bank'] == 0  # 20 < 30
    assert table[2]['queries'] == 20 and table[2]['bank'] == 40
    assert table[5]['samples'] == 0 and table[5]['excluded'] == 1  # absent
    assert np.all(label[bank] != 24) and np.all(label[query] != 24)
    print('test_reference_selection passed')


def _write_features(d, frames, rng, centres, dim=8):
    """Feature samples as tools/extract_point_features.py writes them; the
    OOD points sit next to class 16 (overhead-bridge)."""
    os.makedirs(d)
    for f in range(frames):
        label = np.concatenate([np.repeat(np.arange(C), 20), np.full(30, 24)])
        ood = label == 24
        noise = 0.3 * rng.randn(len(label), dim)
        feat = np.where(ood[:, None], centres[16] + 0.4 + noise,
                        centres[np.minimum(label, C - 1)] + noise)
        np.savez(os.path.join(d, f'f{f:06d}.npz'),
                 feat=feat.astype(np.float16),
                 pos=(0.1 * rng.randn(len(label), dim)).astype(np.float16),
                 logits=np.zeros((len(label), C), np.float16),
                 label=label.astype(np.int16),
                 raw=np.where(ood, 17, 1).astype(np.int16), ood=ood,
                 weight=np.ones(len(label), np.float32),
                 index=np.arange(len(label), dtype=np.int32),
                 lidar_path=f'x{f}', frame=f)


def test_run_end_to_end():
    rng = np.random.RandomState(3)
    centres = 3.0 * rng.randn(C, 8)
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'root')
        _write_features(os.path.join(root, 'features_cetran'), 3, rng, centres)
        _write_features(os.path.join(root, 'features_test'), 4, rng, centres)
        divided = os.path.join(tmp, 'divided')
        os.makedirs(divided)
        subsets = sb.singleton_subsets() + [(12, 16), (0, 1, 2, 3, 4)]
        for key in res.SETS:
            dm.write_tsv(os.path.join(divided, f'{key}.tsv'), [
                OrderedDict([
                    ('name', sb.partition_name(s)), ('size_A', len(s)),
                    ('group_A', ', '.join(sb.CLASSES[c] for c in s)),
                    ('ood_div@0.05', float(rng.uniform(0, 50))),
                    ('id_div@0.05', float(rng.uniform(0, 5))),
                    ('improvement', float(rng.uniform(-30, 30))),
                    ('robust', int(s == (16, )))]) for s in subsets])
        out = os.path.join(tmp, 'resemblance')
        profiles, splits, rho = res.run(
            res.resolve_sets(root), divided, out, k=5, bank_per_class=40,
            query_per_class=20, device='cpu')
        for key in res.SETS:
            for space in res.SPACES:
                for kind in ('profile', 'splits'):
                    assert os.path.exists(
                        os.path.join(out, f'{kind}_{key}_{space}.tsv'))
        profile = {r['class']: r for r in profiles[('test_cetran', 'full')]}
        assert max(profile, key=lambda c: profile[c]['r_ood']) == 'overhead-bridge'
        assert abs(sum(r['r_ood'] for r in profile.values()) - 100.0) < 1e-6
        assert len(splits[('cetran', 'full')]) == len(subsets)
        assert len(rho) == 3 * 2 * (len(res.CLASS_PAIRS) + len(res.SPLIT_PAIRS))
        with open(os.path.join(out, 'summary.md')) as fh:
            text = fh.read()
        for section in ('## Reference banks', '## Hypotheses',
                        '## Profile, Test + Cetran (full space)',
                        '## Robust splits, Cetran (full space)'):
            assert section in text, section
    print('test_run_end_to_end passed')


if __name__ == '__main__':
    test_knn_shares_find_the_nearest_class()
    test_profile_weights_and_exclusion()
    test_split_measures_match_brute_force()
    test_reference_selection()
    test_run_end_to_end()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$ENVPY -m pytest -q tests/test_ood_class_resemblance.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'ood_class_resemblance'`.

- [ ] **Step 3: Implement** — create `tools/ood_class_resemblance.py`:

```python
#!/usr/bin/env python
"""How much the OOD points resemble each ID class, in P3Former's features.

Reads the feature samples of tools/extract_point_features.py -- the 256-d
input of the auxiliary semantic classifier, ``pe_features`` -- and measures
with a k-nearest-neighbour search how the OOD points overlap the 24 ID
classes, against the split's own ID points:

- reference bank: per ID class up to --bank-per-class samples of the set,
  drawn in proportion to their sampling weight (class-balanced, so that no
  class collects neighbours by having more points) and L2-normalised; the
  other ID samples give up to --query-per-class ID queries per class;
- for a query x, n_c(x) is how many of its k nearest bank points (cosine)
  are class c, and n_c(x) / k the share of its neighbourhood in class c;
- r_OOD(c): weighted mean share of class c around the OOD points (100/24 %
  = no preference); r_ID(c): the same around the ID points of the other
  classes, population-weighted; contrast = r_OOD / r_ID;
- per split A | B (the splits of tools/divided_mass.py, A the smaller
  side): R_A = the sum of r_OOD over A, E_A = R_A / (|A| / 24), and the OOD
  / ID feature-divided shares -- a point is feature-divided when its k
  neighbours include classes of both sides, the feature-space twin of the
  divided mass;
- Spearman rho of these against the divided mass and the improvement of
  the same splits: for the hypothesis that the best split puts the classes
  the OOD points resemble together on one side (improvement ~ R_A, E_A)
  and for the alternative that it cuts through them (improvement ~ the OOD
  feature-divided share).

Two feature spaces: ``full`` (what the classifier sees) and ``appearance``
(feat - pos, without the added positional embedding).

Outputs in --out-dir (default <root>/resemblance):
    profile_<set>_<space>.tsv   one row per class
    splits_<set>_<space>.tsv    one row per split
    rho.tsv                     Spearman correlations, long format
    summary.md                  banks, hypotheses, profiles, robust splits

Run from the repo root after tools/divided_mass.py (minutes on a GPU):
    python tools/ood_class_resemblance.py --device cuda:0
    python tools/ood_class_resemblance.py --device cuda:0 --k 50 \\
        --out-dir work_dirs/p3former_2xb1_3x_dso_ood_dump/resemblance_k50
    python tools/plot_ood_class_resemblance.py
"""
import argparse
import glob
import math
import os
import os.path as osp
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import divided_mass as dm  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402

ROOT = dm.ROOT
# Feature-sample directories of each set, relative to --root.
SETS = OrderedDict([
    ('cetran', dict(label='Cetran', features=['features_cetran'])),
    ('test', dict(label='Test', features=['features_test'])),
    ('test_cetran', dict(label='Test + Cetran',
                         features=['features_test', 'features_cetran'])),
])
SPACES = ('full', 'appearance')
NUM_CLASSES = sb.NUM_CLASSES
MIN_CLASS_SAMPLES = 30  # fewer samples: the class is left out of the bank
SAMPLE_KEYS = ('feat', 'pos', 'label', 'raw', 'ood', 'weight')
CLASS_PAIRS = [(x, y) for x in ('r_ood', 'r_id', 'contrast', 'feat_div_ood',
                                'feat_div_id')
               for y in ('ood_div', 'id_div', 'improvement')]
SPLIT_PAIRS = [('feat_div_ood', 'ood_div'), ('feat_div_id', 'id_div'),
               ('R_A', 'improvement'), ('E_A', 'improvement'),
               ('feat_div_ood', 'improvement'),
               ('feat_log_ratio', 'improvement')]
HYPOTHESES = OrderedDict([('together', ('R_A', 'E_A')),
                          ('cut through', ('feat_div_ood', 'feat_log_ratio'))])


# ------------------------------------------------------------------ inputs
def resolve_sets(root, sets=SETS):
    return OrderedDict(
        (key, dict(label=spec['label'],
                   features=[osp.join(root, d) for d in spec['features']]))
        for key, spec in sets.items())


def load_samples(feature_dirs):
    """Concatenated feature samples of one or several directories:
    dict(feat, pos (float16), label (int64), raw, ood, weight (float64))."""
    parts = {key: [] for key in SAMPLE_KEYS}
    for d in feature_dirs:
        files = sorted(glob.glob(osp.join(d, 'f*.npz')))
        if not files:
            raise FileNotFoundError(
                f'no feature samples (f*.npz) in {d}: run '
                'tools/extract_point_features.py first')
        for path in files:
            with np.load(path) as data:
                for key in SAMPLE_KEYS:
                    parts[key].append(data[key])
    out = {key: np.concatenate(value) for key, value in parts.items()}
    out['label'] = out['label'].astype(np.int64)
    out['weight'] = out['weight'].astype(np.float64)
    return out


def space_features(samples, space, index):
    """float32 features of the samples ``index`` in ``space``."""
    feat = samples['feat'][index].astype(np.float32)
    if space == 'full':
        return feat
    if space == 'appearance':
        return feat - samples['pos'][index].astype(np.float32)
    raise ValueError(f'space must be one of {SPACES}, got {space}')


def subset_of(name):
    """'s1.3.17' -> (1, 3, 17)."""
    return tuple(int(c) for c in name[1:].split('.'))


# --------------------------------------------------------------- reference
def select_reference(label, ood, weight, bank_per_class, query_per_class,
                     rng, min_samples=MIN_CLASS_SAMPLES):
    """Class-balanced bank and disjoint ID queries, both drawn in proportion
    to the sampling weights: a class keeps n // 3 (at most query_per_class)
    samples as queries and puts up to bank_per_class of the rest in the
    bank. Returns (bank index, query index, one table row per class)."""
    bank, query, table = [], [], []
    for c in range(NUM_CLASSES):
        members = np.flatnonzero((label == c) & ~ood)
        n = len(members)
        excluded = n < min_samples
        n_query = 0 if excluded else min(query_per_class, n // 3)
        n_bank = 0 if excluded else min(bank_per_class, n - n_query)
        if not excluded:
            p = weight[members] / weight[members].sum()
            chosen = members[rng.choice(n, n_bank + n_query, replace=False,
                                        p=p)]
            bank.append(chosen[:n_bank])
            query.append(chosen[n_bank:])
        table.append(OrderedDict([
            ('class', sb.CLASSES[c]), ('samples', n), ('bank', n_bank),
            ('queries', n_query), ('excluded', int(excluded))]))
    if not bank:
        raise ValueError('no ID class has enough samples for a bank')
    return np.concatenate(bank), np.concatenate(query), table


def query_weights(query_label, class_population):
    """Population weight of each ID query: its class's population split
    evenly over that class's queries."""
    per_class = np.bincount(query_label, minlength=NUM_CLASSES)
    return class_population[query_label] / per_class[query_label]


def knn_class_counts(queries, bank, bank_label, k, device='cuda:0',
                     chunk=8192):
    """n_c(x) [Q, NUM_CLASSES]: how many of each query's k nearest bank
    points (cosine similarity) belong to each class."""
    torch = sb.require_torch()  # full float32 matmuls (TF32 off)
    F = torch.nn.functional
    if k > len(bank):
        raise ValueError(f'k = {k} exceeds the bank size {len(bank)}')
    dev = torch.device(device)
    ref = F.normalize(torch.from_numpy(np.asarray(bank, np.float32)).to(dev),
                      dim=1)
    ref_label = torch.from_numpy(np.asarray(bank_label, np.int64)).to(dev)
    counts = np.zeros((len(queries), NUM_CLASSES), np.int32)
    for i0 in range(0, len(queries), chunk):
        q = F.normalize(torch.from_numpy(
            np.asarray(queries[i0:i0 + chunk], np.float32)).to(dev), dim=1)
        nearest = (q @ ref.t()).topk(k, dim=1).indices
        c = torch.zeros((q.shape[0], NUM_CLASSES), dtype=torch.int64,
                        device=dev)
        c.scatter_add_(1, ref_label[nearest], torch.ones_like(nearest))
        counts[i0:i0 + q.shape[0]] = c.cpu().numpy()
    return counts


# ---------------------------------------------------------------- measures
def class_profile(ood_counts, ood_weight, id_counts, id_label, id_weight, k,
                  excluded):
    """r_OOD, r_ID (percent) and their contrast per class; NaN for the
    classes left out of the bank."""
    r_ood = 100.0 * (ood_weight @ (ood_counts / k)) / ood_weight.sum()
    r_id = np.empty(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        other = id_label != c
        r_id[c] = (100.0 * (id_weight[other] @ (id_counts[other, c] / k))
                   / id_weight[other].sum())
    contrast = np.array([dm.ratio(a, b) for a, b in zip(r_ood, r_id)])
    for values in (r_ood, r_id, contrast):
        values[excluded] = np.nan
    return r_ood, r_id, contrast


def split_measures(a_mask, r_ood, ood_counts, ood_weight, id_counts,
                   id_weight, k, chunk=65536):
    """Per split: R_A (%), E_A and the OOD / ID feature-divided shares (%)."""
    M = a_mask.astype(np.float64)
    R_A = M @ np.nan_to_num(r_ood)
    E_A = R_A / (100.0 * a_mask.sum(axis=1) / NUM_CLASSES)

    def divided(counts, weight):
        total = np.zeros(len(M))
        for i0 in range(0, len(counts), chunk):
            n_a = counts[i0:i0 + chunk].astype(np.float64) @ M.T
            total += weight[i0:i0 + chunk] @ ((n_a > 0) & (n_a < k))
        return 100.0 * total / weight.sum()

    return R_A, E_A, divided(ood_counts, ood_weight), divided(id_counts,
                                                               id_weight)


# ------------------------------------------------------------------ tables
def split_rows(names, subsets, divided, measures, threshold):
    key = dm.delta_key(threshold)
    R_A, E_A, fd_ood, fd_id = measures
    rows = []
    for s, (name, subset) in enumerate(zip(names, subsets)):
        d = divided[name]
        rows.append(OrderedDict([
            ('name', name), ('size_A', len(subset)), ('group_A', d['group_A']),
            ('R_A', R_A[s]), ('E_A', E_A[s]),
            ('feat_div_ood', fd_ood[s]), ('feat_div_id', fd_id[s]),
            ('feat_log_ratio', math.log10(max(fd_ood[s], 1e-3)
                                          / max(fd_id[s], 1e-3))),
            ('ood_div', d[f'ood_div@{key}']), ('id_div', d[f'id_div@{key}']),
            ('improvement', d['improvement']), ('robust', d['robust'])]))
    return rows


def profile_rows(profile, reference, splits):
    r_ood, r_id, contrast = profile
    by_name = {r['name']: r for r in splits}
    rows = []
    for c in range(NUM_CLASSES):
        single = by_name[sb.partition_name((c, ))]
        rows.append(OrderedDict([
            ('class', sb.CLASSES[c]), ('bank', reference[c]['bank']),
            ('queries', reference[c]['queries']), ('r_ood', r_ood[c]),
            ('r_id', r_id[c]), ('contrast', contrast[c]),
            ('feat_div_ood', single['feat_div_ood']),
            ('feat_div_id', single['feat_div_id']),
            ('ood_div', single['ood_div']), ('id_div', single['id_div']),
            ('improvement', single['improvement'])]))
    return rows


def alignment(set_key, space, k, profile, splits):
    """Spearman rho of CLASS_PAIRS over the classes and SPLIT_PAIRS over
    the splits."""
    out = []
    for population, rows, pairs in (('classes', profile, CLASS_PAIRS),
                                    ('splits', splits, SPLIT_PAIRS)):
        for x, y in pairs:
            rho, n = dm.spearman([r[x] for r in rows], [r[y] for r in rows])
            out.append(OrderedDict([
                ('set', set_key), ('space', space), ('k', k),
                ('population', population), ('x', x), ('y', y), ('rho', rho),
                ('n', n)]))
    return out


def write_summary(path, sets, references, profiles, split_tables, rho, k):
    lines = [
        '# OOD resemblance to the ID classes, in the feature space', '',
        f'kNN, k = {k}, cosine similarity, against a class-balanced bank of '
        "the set's own ID samples. r_OOD(c): share of class c among the "
        "OOD points' neighbours (100/24 = 4.17 % = no preference). r_ID(c): "
        'the same around the ID points of the other classes. contrast = '
        'r_OOD / r_ID. Feature-divided: the k neighbours hold classes of '
        'both sides of a split.', '', '## Reference banks', '']
    rows = []
    for c in range(NUM_CLASSES):
        cells = [sb.CLASSES[c]]
        for ref in references.values():
            r = ref[c]
            cells.append(f'{r["samples"]} / {r["bank"]} / {r["queries"]}'
                         + (' (left out)' if r['excluded'] else ''))
        rows.append(cells)
    lines += dm.md_table(
        ['class'] + [f'{sets[key]["label"]}: samples / bank / queries'
                     for key in references], rows)
    lines += ['', '## Hypotheses: Spearman rho with the improvement, over '
              'all splits', '']
    rows = []
    for key in sets:
        for space in SPACES:
            sel = {r['x']: r['rho'] for r in rho if r['set'] == key
                   and r['space'] == space and r['population'] == 'splits'
                   and r['y'] == 'improvement'}
            if sel:
                rows.append([sets[key]['label'], space]
                            + [dm.fmt(sel[x]) for xs in HYPOTHESES.values()
                               for x in xs])
    lines += dm.md_table(
        ['set', 'space'] + [f'{name}: {x}' for name, xs in HYPOTHESES.items()
                            for x in xs], rows)
    for (key, space), profile in profiles.items():
        lines += ['', f'## Profile, {sets[key]["label"]} ({space} space)', '']
        order = sorted(profile,
                       key=lambda r: -np.nan_to_num(r['r_ood'], nan=-1.0))
        lines += dm.md_table(
            ['class', 'r_OOD %', 'r_ID %', 'contrast', 'OOD feat-div %',
             'ID feat-div %', 'OOD div %', 'ID div %', 'improvement'],
            [[r['class'], dm.fmt(r['r_ood']), dm.fmt(r['r_id']),
              dm.fmt(r['contrast']), dm.fmt(r['feat_div_ood']),
              dm.fmt(r['feat_div_id']), dm.pct(r['ood_div']),
              dm.pct(r['id_div']), dm.fmt(r['improvement'])] for r in order])
        table = {(r['x'], r['y']): r['rho'] for r in rho if r['set'] == key
                 and r['space'] == space and r['population'] == 'classes'}
        ys = ('ood_div', 'id_div', 'improvement')
        xs = ('r_ood', 'r_id', 'contrast', 'feat_div_ood', 'feat_div_id')
        lines += ['', 'Spearman rho over the 24 classes:', '']
        lines += dm.md_table(['x'] + list(ys),
                             [[x] + [dm.fmt(table[(x, y)]) for y in ys]
                              for x in xs])
    for (key, space), splits in split_tables.items():
        if space != 'full':
            continue
        robust = sorted((r for r in splits if r['robust']),
                        key=lambda r: -r['improvement'])
        lines += ['', f'## Robust splits, {sets[key]["label"]} (full space)',
                  '']
        lines += dm.md_table(
            ['name', 'classes', 'R_A %', 'E_A', 'OOD feat-div %',
             'ID feat-div %', 'OOD div %', 'ID div %', 'improvement'],
            [[r['name'], r['group_A'], dm.fmt(r['R_A']), dm.fmt(r['E_A']),
              dm.fmt(r['feat_div_ood']), dm.fmt(r['feat_div_id']),
              dm.pct(r['ood_div']), dm.pct(r['id_div']),
              dm.fmt(r['improvement'])] for r in robust])
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# --------------------------------------------------------------------- main
def run(sets, divided_dir, out_dir, k=10, bank_per_class=4000,
        query_per_class=2000, threshold=dm.HEADLINE, device='cuda:0', seed=0,
        spaces=SPACES):
    """Every output for ``sets`` (see :func:`resolve_sets`); returns
    (profiles, split_tables, rho), the first two keyed by (set, space)."""
    os.makedirs(out_dir, exist_ok=True)
    profiles, split_tables, references, rho = (OrderedDict(), OrderedDict(),
                                               OrderedDict(), [])
    for key, spec in sets.items():
        t0 = time.time()
        divided = OrderedDict(
            (r['name'], r)
            for r in dm.read_tsv(osp.join(divided_dir, f'{key}.tsv')))
        names = list(divided)
        subsets = [subset_of(name) for name in names]
        a_mask = sb.subsets_to_mask(subsets)
        samples = load_samples(spec['features'])
        label, ood, weight = samples['label'], samples['ood'], samples['weight']
        bank, query, reference = select_reference(
            label, ood, weight, bank_per_class, query_per_class,
            np.random.default_rng(seed))
        references[key] = reference
        excluded = np.array([r['excluded'] == 1 for r in reference])
        population = np.bincount(label[~ood], weights=weight[~ood],
                                 minlength=NUM_CLASSES)[:NUM_CLASSES]
        ood_index = np.flatnonzero(ood)
        q_weight = query_weights(label[query], population)
        for space in spaces:
            ref = space_features(samples, space, bank)
            ood_counts = knn_class_counts(
                space_features(samples, space, ood_index), ref, label[bank],
                k, device)
            id_counts = knn_class_counts(space_features(samples, space, query),
                                         ref, label[bank], k, device)
            profile = class_profile(ood_counts, weight[ood_index], id_counts,
                                    label[query], q_weight, k, excluded)
            measures = split_measures(a_mask, profile[0], ood_counts,
                                      weight[ood_index], id_counts, q_weight,
                                      k)
            splits = split_rows(names, subsets, divided, measures, threshold)
            prof = profile_rows(profile, reference, splits)
            dm.write_tsv(osp.join(out_dir, f'profile_{key}_{space}.tsv'), prof)
            dm.write_tsv(osp.join(out_dir, f'splits_{key}_{space}.tsv'), splits)
            profiles[(key, space)] = prof
            split_tables[(key, space)] = splits
            rho += alignment(key, space, k, prof, splits)
        print(f'{spec["label"]}: {len(ood_index)} OOD / {int((~ood).sum())} '
              f'ID samples, bank {len(bank)}, {len(query)} ID queries '
              f'({time.time() - t0:.0f} s)', flush=True)
    dm.write_tsv(osp.join(out_dir, 'rho.tsv'), rho)
    write_summary(osp.join(out_dir, 'summary.md'), sets, references, profiles,
                  split_tables, rho, k)
    print(f'wrote {out_dir}')
    return profiles, split_tables, rho


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT,
                    help='directory holding the feature samples (SETS)')
    ap.add_argument('--divided-dir', default=None,
                    help='tools/divided_mass.py output; default '
                    '<root>/divided_mass')
    ap.add_argument('--out-dir', default=None,
                    help='default: <root>/resemblance')
    ap.add_argument('--k', type=int, default=10, help='neighbours per query')
    ap.add_argument('--bank-per-class', type=int, default=4000)
    ap.add_argument('--query-per-class', type=int, default=2000)
    ap.add_argument('--threshold', type=float, default=dm.HEADLINE,
                    help='divided-mass threshold the splits are compared at')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.threshold not in dm.DELTAS:
        ap.error('--threshold must be one of '
                 + ', '.join(dm.delta_key(d) for d in dm.DELTAS))
    run(resolve_sets(args.root),
        args.divided_dir or osp.join(args.root, 'divided_mass'),
        args.out_dir or osp.join(args.root, 'resemblance'), k=args.k,
        bank_per_class=args.bank_per_class,
        query_per_class=args.query_per_class, threshold=args.threshold,
        device=args.device, seed=args.seed)


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$ENVPY -m pytest -q tests/test_ood_class_resemblance.py && $ENVPY tests/test_ood_class_resemblance.py`
Expected: every test passes.

- [ ] **Step 5: Full suite, then commit**

```bash
$ENVPY -m pytest -q tests
git add tools/ood_class_resemblance.py tests/test_ood_class_resemblance.py
git commit -m "$(cat <<'EOF'
Measure how OOD points resemble each ID class in the feature space

tools/ood_class_resemblance.py: kNN shares against a class-balanced bank
of the split's own ID samples give r_OOD, r_ID and their contrast per
class, and R_A, E_A and the feature-divided shares per split, correlated
with the divided mass and the improvement (full and appearance spaces).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `tools/plot_ood_class_resemblance.py`

**Files:**
- Create: `tools/plot_ood_class_resemblance.py`
- Test: `tests/test_plot_ood_class_resemblance.py`

**Interfaces:**
- Consumes: Task 3's `pdm.STYLE`, `GAIN`, `DROP`, `GAIN_TXT`, `DROP_TXT`, `tint`, `signed`, `save` and `bubble_grid`; Task 5's `res.SETS`, `res.SPACES` and TSV columns; Task 2's `dm.ROOT`, `dm.read_tsv` and `dm.fmt`.
- Produces: `profile_<space>`, `bubble_feature_singletons_<space>` and `hypothesis_<space>` (each as `.pdf` and `.png`) in the resemblance directory.

- [ ] **Step 1: Write the failing test** — create `tests/test_plot_ood_class_resemblance.py`:

```python
"""Smoke test for tools/plot_ood_class_resemblance.py (synthetic tables).

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_plot_ood_class_resemblance.py
"""
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import divided_mass as dm  # noqa: E402
import ood_class_resemblance as res  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402


def test_figures_are_written():
    rng = np.random.RandomState(0)
    subsets = sb.singleton_subsets() + [(12, 16), (0, 1, 2, 3, 4)]
    with tempfile.TemporaryDirectory() as tmp:
        rho = []
        for key in res.SETS:
            for space in res.SPACES:
                shares = 100.0 * rng.dirichlet(np.ones(sb.NUM_CLASSES))
                dm.write_tsv(os.path.join(tmp, f'profile_{key}_{space}.tsv'), [
                    OrderedDict([
                        ('class', name), ('bank', 40), ('queries', 20),
                        ('r_ood', float(shares[c])),
                        ('r_id', float(rng.uniform(0, 8))),
                        ('contrast', float(rng.uniform(0, 5))),
                        ('feat_div_ood', float(rng.uniform(0.1, 60))),
                        ('feat_div_id', 0.0 if c == 3 else float(rng.uniform(0.1, 10))),
                        ('ood_div', float(rng.uniform(0, 50))),
                        ('id_div', float(rng.uniform(0, 5))),
                        ('improvement', float(rng.uniform(-40, 35)))])
                    for c, name in enumerate(sb.CLASSES)])
                dm.write_tsv(os.path.join(tmp, f'splits_{key}_{space}.tsv'), [
                    OrderedDict([
                        ('name', sb.partition_name(s)), ('size_A', len(s)),
                        ('group_A', 'x'), ('R_A', float(rng.uniform(0, 60))),
                        ('E_A', float(rng.uniform(0, 5))),
                        ('feat_div_ood', float(rng.uniform(0, 60))),
                        ('feat_div_id', float(rng.uniform(0, 10))),
                        ('feat_log_ratio', float(rng.uniform(-1, 2))),
                        ('ood_div', 1.0), ('id_div', 1.0),
                        ('improvement', float(rng.uniform(-40, 35))),
                        ('robust', int(s == (16, )))]) for s in subsets])
                for population, pairs in (('classes', res.CLASS_PAIRS),
                                          ('splits', res.SPLIT_PAIRS)):
                    rho += [OrderedDict([
                        ('set', key), ('space', space), ('k', 10),
                        ('population', population), ('x', x), ('y', y),
                        ('rho', float(rng.uniform(-1, 1))), ('n', 24)])
                        for x, y in pairs]
        dm.write_tsv(os.path.join(tmp, 'rho.tsv'), rho)
        script = os.path.join(_REPO_ROOT, 'tools',
                              'plot_ood_class_resemblance.py')
        for space in res.SPACES:
            proc = subprocess.run([sys.executable, script, tmp, '--space',
                                   space], capture_output=True, text=True,
                                  cwd=_REPO_ROOT)
            assert proc.returncode == 0, proc.stderr
            for name in ('profile', 'bubble_feature_singletons', 'hypothesis'):
                for ext in ('.pdf', '.png'):
                    path = os.path.join(tmp, f'{name}_{space}{ext}')
                    assert os.path.getsize(path) > 1000, path
    print('test_figures_are_written passed')


if __name__ == '__main__':
    test_figures_are_written()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$ENVPY -m pytest -q tests/test_plot_ood_class_resemblance.py`
Expected: FAIL (the script does not exist yet).

- [ ] **Step 3: Implement** — create `tools/plot_ood_class_resemblance.py`:

```python
#!/usr/bin/env python
"""Figures of tools/ood_class_resemblance.py.

Reads the tables of tools/ood_class_resemblance.py (default
work_dirs/p3former_2xb1_3x_dso_ood_dump/resemblance) and writes, next to
them, as PDF and PNG in the style of tools/plot_divided_mass.py:

    profile_<space>                    per set, the 24 classes (one order,
                                       by r_OOD on Test + Cetran): bars of
                                       r_OOD, coloured by the sign of
                                       {c} | rest's improvement, and of
                                       r_ID (grey); the appearance-space
                                       r_OOD as hollow circles (full space
                                       only); no preference (100/24 %) dashed
    bubble_feature_singletons_<space>  ID (x) against OOD (y)
                                       feature-divided % of the single-class
                                       splits, area = |improvement|: the
                                       feature-space twin of
                                       bubble_singletons
    hypothesis_<space>                 improvement against R_A ("together")
                                       and against the OOD feature-divided
                                       % ("cut through") over every split,
                                       robust splits highlighted, Spearman
                                       rho in the titles

Run from the repo root:
    python tools/plot_ood_class_resemblance.py [RESEMBLANCE_DIR] [--space full]
"""
import argparse
import os.path as osp
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import divided_mass as dm  # noqa: E402
import ood_class_resemblance as res  # noqa: E402
import plot_divided_mass as pdm  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402


def load(res_dir, space):
    read = lambda name: dm.read_tsv(osp.join(res_dir, name))  # noqa: E731
    labels = OrderedDict((k, s['label']) for k, s in res.SETS.items())
    profiles = OrderedDict((k, read(f'profile_{k}_{space}.tsv'))
                           for k in res.SETS)
    appearance = OrderedDict()
    if space == 'full':
        appearance = OrderedDict(
            (k, read(f'profile_{k}_appearance.tsv')) for k in res.SETS
            if osp.exists(osp.join(res_dir, f'profile_{k}_appearance.tsv')))
    splits = OrderedDict((k, read(f'splits_{k}_{space}.tsv'))
                         for k in res.SETS)
    return labels, profiles, appearance, splits, read('rho.tsv')


def profile_figure(labels, profiles, appearance, stem):
    """One panel per set, classes in one order (by r_OOD on the last set)."""
    last = list(profiles)[-1]
    order = [r['class'] for r in sorted(
        profiles[last], key=lambda r: np.nan_to_num(r['r_ood'], nan=-1.0))]
    y = np.arange(len(order))
    xmax = max(max(np.nan_to_num(r['r_ood']), np.nan_to_num(r['r_id']))
               for rows in profiles.values() for r in rows)
    fig, axes = plt.subplots(1, len(profiles), figsize=(7.0, 4.8),
                             sharey=True, squeeze=False)
    for ax, (key, rows) in zip(axes[0], profiles.items()):
        by = {r['class']: r for r in rows}
        r_ood = [float(np.nan_to_num(by[c]['r_ood'])) for c in order]
        r_id = [float(np.nan_to_num(by[c]['r_id'])) for c in order]
        imp = [by[c]['improvement'] for c in order]
        edge = [pdm.GAIN if v > 0 else pdm.DROP for v in imp]
        ax.barh(y + 0.2, r_ood, height=0.4, color=[pdm.tint(c) for c in edge],
                edgecolor=edge, linewidth=0.6)
        ax.barh(y - 0.2, r_id, height=0.4, color='0.85', edgecolor='0.55',
                linewidth=0.6)
        if key in appearance:
            app = {r['class']: float(np.nan_to_num(r['r_ood']))
                   for r in appearance[key]}
            ax.scatter([app[c] for c in order], y + 0.2, s=10,
                       facecolor='none', edgecolor='black', linewidths=0.6,
                       zorder=4)
        for yi, x, v in zip(y, r_ood, imp):
            ax.text(x + 0.015 * xmax, yi + 0.2, pdm.signed(v), va='center',
                    fontsize=5.5,
                    color=pdm.GAIN_TXT if v > 0 else pdm.DROP_TXT)
        ax.axvline(100.0 / sb.NUM_CLASSES, color='0.4', ls=(0, (3, 2)),
                   lw=0.6)
        ax.set_xlim(0, xmax * 1.3)
        ax.set_title(labels[key], loc='left')
        ax.set_xlabel('share of the neighbours (%)')
        ax.grid(axis='x', color='0.9', lw=0.4)
        ax.set_axisbelow(True)
    axes[0, 0].set_yticks(y)
    axes[0, 0].set_yticklabels(order, fontsize=6.5)
    handles = [
        Line2D([], [], ls='none', marker='s', markersize=6,
               markerfacecolor=pdm.tint(pdm.GAIN), markeredgecolor=pdm.GAIN),
        Line2D([], [], ls='none', marker='s', markersize=6,
               markerfacecolor=pdm.tint(pdm.DROP), markeredgecolor=pdm.DROP),
        Line2D([], [], ls='none', marker='s', markersize=6,
               markerfacecolor='0.85', markeredgecolor='0.55'),
        Line2D([], [], ls=(0, (3, 2)), lw=0.6, color='0.4')]
    names = [r'$r_\mathrm{OOD}$, {c} | rest improves (value: improvement)',
             r'$r_\mathrm{OOD}$, {c} | rest drops',
             r'$r_\mathrm{ID}$: ID points of the other classes',
             'no preference (100/24 %)']
    if appearance:
        handles.append(Line2D([], [], ls='none', marker='o', markersize=3.5,
                              markerfacecolor='none', markeredgecolor='black'))
        names.append(r'$r_\mathrm{OOD}$, appearance space')
    fig.legend(handles, names, loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=0.15, right=0.99, top=0.95, bottom=0.2,
                        wspace=0.08)
    pdm.save(fig, stem)


def feature_bubbles(labels, profiles, stem):
    panels = OrderedDict(
        (labels[key], [dict(x=r['feat_div_id'], y=r['feat_div_ood'],
                            value=r['improvement'], text=r['class'])
                       for r in rows])
        for key, rows in profiles.items())
    pdm.bubble_grid(panels, 'ID feature-divided (%)',
                    'OOD feature-divided (%)', stem)


def hypothesis_figure(labels, splits, rho, space, stem):
    fig, axes = plt.subplots(len(splits), 2, figsize=(7.0, 7.6),
                             squeeze=False)
    for i, (key, rows) in enumerate(splits.items()):
        sel = {r['x']: r['rho'] for r in rho if r['set'] == key
               and r['space'] == space and r['population'] == 'splits'
               and r['y'] == 'improvement'}
        other = [r for r in rows if not r['robust']]
        robust = [r for r in rows if r['robust']]
        for j, (x, name, xlabel) in enumerate((
                ('R_A', 'together', r'$R_A$: OOD resemblance on the smaller '
                 'side (%)'),
                ('feat_div_ood', 'cut through', 'OOD feature-divided (%)'))):
            ax = axes[i, j]
            ax.scatter([r[x] for r in other], [r['improvement'] for r in other],
                       s=5, color='0.72', linewidths=0)
            ax.scatter([r[x] for r in robust],
                       [r['improvement'] for r in robust], s=12,
                       color=pdm.GAIN, linewidths=0)
            ax.axhline(0.0, color='0.4', lw=0.6)
            ax.set_title(f'{labels[key]}, {name}: ' + r'$\rho$ = '
                         + dm.fmt(sel.get(x, float('nan'))), loc='left')
            ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel('improvement')
            ax.grid(color='0.92', lw=0.4)
            ax.set_axisbelow(True)
    handles = [Line2D([], [], ls='none', marker='o', markersize=3,
                      color='0.72'),
               Line2D([], [], ls='none', marker='o', markersize=4,
                      color=pdm.GAIN)]
    fig.legend(handles, ['splits', 'robust splits (improvement > 0 on every '
                         'set)'], loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.09,
                        hspace=0.55, wspace=0.18)
    pdm.save(fig, stem)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('res_dir', nargs='?',
                    default=osp.join(dm.ROOT, 'resemblance'),
                    help='output directory of tools/ood_class_resemblance.py')
    ap.add_argument('--space', choices=res.SPACES, default='full')
    args = ap.parse_args()
    plt.rcParams.update(pdm.STYLE)
    labels, profiles, appearance, splits, rho = load(args.res_dir, args.space)
    stem = lambda name: osp.join(args.res_dir, f'{name}_{args.space}')  # noqa: E731
    profile_figure(labels, profiles, appearance, stem('profile'))
    feature_bubbles(labels, profiles, stem('bubble_feature_singletons'))
    hypothesis_figure(labels, splits, rho, args.space, stem('hypothesis'))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `$ENVPY -m pytest -q tests/test_plot_ood_class_resemblance.py && $ENVPY tests/test_plot_ood_class_resemblance.py`
Expected: PASS.

- [ ] **Step 5: Full suite, then commit**

```bash
$ENVPY -m pytest -q tests
git add tools/plot_ood_class_resemblance.py tests/test_plot_ood_class_resemblance.py
git commit -m "$(cat <<'EOF'
Plot the OOD resemblance profiles and the hypothesis scatter

Per-class r_OOD / r_ID bars, the feature-space twin of the single-class
bubble chart, and improvement against R_A ('together') and against the
OOD feature-divided share ('cut through') over every split.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `tools/plot_feature_tsne.py`

**Files:**
- Create: `tools/plot_feature_tsne.py`
- Test: `tests/test_plot_feature_tsne.py`

**Interfaces:**
- Consumes: Task 5's `res.ROOT`, `res.SPACES`, `resolve_sets`, `load_samples` and `space_features`; Task 3's `pdm.STYLE` and `save`; `sb.CLASSES` and `sb.NUM_CLASSES`; `make_dso_hierarchy_variants.GROUPS`.
- Produces: the constant `CLASS_COLOURS`; the functions `tsne_sample(samples, per_class, n_ood, rng)`, `embed(features, seed=0, perplexity=30.0)` and `tsne_figure(panels, stem)`; and the figure `resemblance/tsne_<space>.{pdf,png}`.

The colours below were searched once in OKLCH: family anchors from the dataviz reference palette, lightness in 0.44–0.76, hue offsets up to ±12°, chosen to maximise the smallest pairwise distance.
- The dataviz validator: lightness band PASS, chroma floor PASS; worst normal-vision pair ΔE 9.6, worst deuteranopia pair ΔE 5.1.
- The all-pairs checks fail, as they must for 24 series. The class names written on the plot are the required second channel.

- [ ] **Step 1: Write the failing tests** — create `tests/test_plot_feature_tsne.py`:

```python
"""Tests for tools/plot_feature_tsne.py.

Run from the repo root:
    /home/khoadv/miniconda3/envs/p3former/bin/python tests/test_plot_feature_tsne.py
"""
import os
import re
import sys
import tempfile
from collections import OrderedDict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, 'tools'))

import plot_feature_tsne as tsne  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402
from make_dso_hierarchy_variants import GROUPS  # noqa: E402


def test_class_colours():
    assert list(tsne.CLASS_COLOURS) == list(sb.CLASSES)
    colours = [c.lower() for c in tsne.CLASS_COLOURS.values()]
    assert len(set(colours)) == sb.NUM_CLASSES
    assert all(re.fullmatch(r'#[0-9a-f]{6}', c) for c in colours)
    assert tsne.OOD_COLOUR.lower() not in colours
    covered = sorted(c for _, ids in GROUPS.values() for c in ids)
    assert covered == list(range(sb.NUM_CLASSES))  # every class has a family
    print('test_class_colours passed')


def test_tsne_figure_smoke():
    rng = np.random.RandomState(0)
    n = 240
    label = np.repeat([0, 12, 16, 24], n // 4)
    ood = label == 24
    raw = np.where(ood, np.where(np.arange(n) % 2 == 0, 17, 28), 1)
    feat = rng.randn(n, 8) + np.where(ood[:, None], 2.0, label[:, None] / 8.0)
    samples = dict(label=label, ood=ood, raw=raw, weight=np.ones(n))
    idx = tsne.tsne_sample(samples, 40, 50, np.random.default_rng(0))
    assert len(idx) == 3 * 40 + 50 and len(set(idx.tolist())) == len(idx)
    xy = tsne.embed(feat[idx].astype(np.float32), seed=0)
    assert xy.shape == (len(idx), 2) and np.all(np.isfinite(xy))
    panels = OrderedDict((name, (xy, label[idx], raw[idx], ood[idx]))
                         for name in ('Cetran', 'Test', 'Test + Cetran'))
    with tempfile.TemporaryDirectory() as tmp:
        stem = os.path.join(tmp, 'tsne_full')
        tsne.tsne_figure(panels, stem)
        for ext in ('.pdf', '.png'):
            assert os.path.getsize(stem + ext) > 1000
    print('test_tsne_figure_smoke passed')


if __name__ == '__main__':
    test_class_colours()
    test_tsne_figure_smoke()
    print('ALL TESTS PASSED')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$ENVPY -m pytest -q tests/test_plot_feature_tsne.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'plot_feature_tsne'`.

- [ ] **Step 3: Implement** — create `tools/plot_feature_tsne.py`:

```python
#!/usr/bin/env python
"""t-SNE view of the feature samples: where the OOD points sit among the
ID classes.

Per set (Cetran, Test, Test + Cetran) it draws up to --per-class ID points
of every class and --ood OOD points from the samples of
tools/extract_point_features.py, each in proportion to its weight,
L2-normalises the features (as the kNN of tools/ood_class_resemblance.py),
reduces them to 50 dimensions with PCA and embeds them with t-SNE
(perplexity 30, PCA initialisation, --seed). One panel per set: ID points
in their class's fixed colour (CLASS_COLOURS), OOD points black on top
(triangle = Stop, cross = Others), each class's name at the median of its
points, and a legend below. t-SNE keeps neighbourhoods, not distances
between clusters or cluster sizes: the figure illustrates the kNN
measures, it is not evidence on its own.

Run from the repo root (a few minutes per set):
    python tools/plot_feature_tsne.py [--space full|appearance]
"""
import argparse
import os
import os.path as osp
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

import ood_class_resemblance as res  # noqa: E402
import plot_divided_mass as pdm  # noqa: E402
import sweep_bipartitions as sb  # noqa: E402
from make_dso_hierarchy_variants import GROUPS  # noqa: E402

# One fixed colour per class, the same in every figure. Each semantic group
# of GROUPS is a hue family -- vehicle blue, human magenta, ground ochre,
# construction red, nature green, object violet (anchors from the dataviz
# reference palette) -- with lightness and small hue steps within a family
# searched to maximise the smallest pairwise distance. Twenty-four colours
# cannot all be told apart (worst pair dE 9.6 in OKLab x100, 5.1 under
# deuteranopia), so the class names are written on the plot as well.
CLASS_COLOURS = OrderedDict([
    ('car', '#4e75f8'), ('bicycle', '#6297ce'), ('motorcycle', '#065788'),
    ('truck', '#84b1fe'), ('bus', '#1f2dd9'),
    ('person', '#7d3460'), ('rider', '#f857a7'),
    ('traffic-sign', '#4d3ca9'), ('traffic-cone', '#6945ea'),
    ('paved-road', '#856004'), ('unpaved-road', '#b66b0a'),
    ('sidewalk', '#fa9602'),
    ('building', '#8d2e10'), ('window', '#b00b40'),
    ('perimeter-barrier', '#a95a64'), ('other-barrier', '#eb4d4a'),
    ('overhead-bridge', '#fd8a90'), ('gate', '#cb7b7d'),
    ('pole-like-object', '#a68cff'), ('drain', '#c38f00'),
    ('terrain', '#17643a'), ('trunks', '#639357'), ('vegetation', '#7ac955'),
    ('obscurant', '#6564ad'),
])
OOD_COLOUR = '#000000'
OOD_MARKERS = OrderedDict([(17, ('^', 'OOD: Stop')),
                           (28, ('x', 'OOD: Others'))])


def _weighted(members, weight, n, rng):
    n = min(n, len(members))
    if n == 0:
        return np.zeros(0, np.int64)
    p = weight[members] / weight[members].sum()
    return members[rng.choice(len(members), n, replace=False, p=p)]


def tsne_sample(samples, per_class, n_ood, rng):
    """Up to ``per_class`` ID samples of every class and ``n_ood`` OOD
    samples, each drawn in proportion to its weight."""
    label, ood, weight = samples['label'], samples['ood'], samples['weight']
    chosen = [_weighted(np.flatnonzero((label == c) & ~ood), weight,
                        per_class, rng) for c in range(sb.NUM_CLASSES)]
    chosen.append(_weighted(np.flatnonzero(ood), weight, n_ood, rng))
    return np.concatenate(chosen)


def embed(features, seed=0, perplexity=30.0):
    """2-D t-SNE of the L2-normalised features after PCA to 50 dimensions."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    x = features / np.maximum(
        np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    x = PCA(n_components=min(50, x.shape[1], len(x)),
            random_state=seed).fit_transform(x)
    return TSNE(n_components=2, perplexity=min(perplexity, (len(x) - 1) / 3),
                init='pca', learning_rate='auto',
                random_state=seed).fit_transform(x)


def tsne_figure(panels, stem):
    """One panel per set (panels: title -> (xy, label, raw, ood)): ID points
    in their class colour, OOD points black on top, class names at the
    class medians, a legend below grouped by family."""
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 3.5),
                             squeeze=False)
    for ax, (title, (xy, label, raw, ood)) in zip(axes[0], panels.items()):
        for c, name in enumerate(sb.CLASSES):
            sel = (label == c) & ~ood
            if sel.any():
                ax.scatter(xy[sel, 0], xy[sel, 1], s=1.2, linewidths=0,
                           color=CLASS_COLOURS[name], alpha=0.75)
        for raw_id, (marker, _) in OOD_MARKERS.items():
            sel = ood & (raw == raw_id)
            ax.scatter(xy[sel, 0], xy[sel, 1], s=4, marker=marker,
                       color=OOD_COLOUR, linewidths=0.4, zorder=5)
        for c, name in enumerate(sb.CLASSES):
            sel = (label == c) & ~ood
            if sel.sum() >= 5:
                mx, my = np.median(xy[sel], axis=0)
                ax.text(mx, my, name, fontsize=4.8, ha='center', va='center',
                        color=CLASS_COLOURS[name], zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.4,
                                                    foreground='white')])
        ax.set_title(title, loc='left')
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ('left', 'bottom'):
            ax.spines[side].set_visible(False)
    handles, names = [], []
    for _, ids in GROUPS.values():
        for c in ids:
            handles.append(Line2D([], [], ls='none', marker='o',
                                  markersize=3.5,
                                  color=CLASS_COLOURS[sb.CLASSES[c]]))
            names.append(sb.CLASSES[c])
    for marker, name in OOD_MARKERS.values():
        handles.append(Line2D([], [], ls='none', marker=marker,
                              markersize=3.5, color=OOD_COLOUR,
                              markeredgewidth=0.6))
        names.append(name)
    fig.legend(handles, names, loc='lower center', ncol=7, frameon=False,
               fontsize=6, handletextpad=0.2, columnspacing=0.9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.26,
                        wspace=0.05)
    pdm.save(fig, stem)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=res.ROOT,
                    help='directory holding the feature samples')
    ap.add_argument('--out-dir', default=None,
                    help='default: <root>/resemblance')
    ap.add_argument('--space', choices=res.SPACES, default='full')
    ap.add_argument('--per-class', type=int, default=300,
                    help='ID points per class and set')
    ap.add_argument('--ood', type=int, default=3000,
                    help='OOD points per set')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    plt.rcParams.update(pdm.STYLE)
    panels = OrderedDict()
    for spec in res.resolve_sets(args.root).values():
        samples = res.load_samples(spec['features'])
        idx = tsne_sample(samples, args.per_class, args.ood,
                          np.random.default_rng(args.seed))
        xy = embed(res.space_features(samples, args.space, idx),
                   seed=args.seed)
        panels[spec['label']] = (xy, samples['label'][idx],
                                 samples['raw'][idx], samples['ood'][idx])
        print(f'{spec["label"]}: t-SNE of {len(idx)} points', flush=True)
    out_dir = args.out_dir or osp.join(args.root, 'resemblance')
    os.makedirs(out_dir, exist_ok=True)
    tsne_figure(panels, osp.join(out_dir, f'tsne_{args.space}'))


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$ENVPY -m pytest -q tests/test_plot_feature_tsne.py && $ENVPY tests/test_plot_feature_tsne.py`
Expected: both tests pass.

- [ ] **Step 5: Full suite, then commit**

```bash
$ENVPY -m pytest -q tests
git add tools/plot_feature_tsne.py tests/test_plot_feature_tsne.py
git commit -m "$(cat <<'EOF'
Add a t-SNE view of the feature samples with fixed class colours

One panel per set; each class keeps one colour everywhere (a hue family
per semantic group), OOD points are black on top, and the class names
sit at the class medians because 24 colours cannot all be told apart.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Formality and maintainability review of the new tools

The user asked for this in these words: "spawn an agent to test the new python script you create, to check if it is formal and easy to manage".

**Files:** the seven tools and six test files from Tasks 1–7; fixes land in those same files.

- [ ] **Step 1: Dispatch the review agent** (general-purpose, fresh context) with this brief:

> Review the new analysis tools of this repository for formality and maintainability. The files are `tools/divided_mass.py`, `plot_divided_mass.py`, `extract_point_features.py`, `ood_class_resemblance.py`, `plot_ood_class_resemblance.py`, `plot_feature_tsne.py`, the `--subsets` change in `tools/sweep_bipartitions.py`, and their tests in `tests/`.
>
> Run from the repo root with `/home/khoadv/miniconda3/envs/p3former/bin/python`:
> 1. Every tool's `--help`.
> 2. `-m pytest -q tests`.
> 3. Each test file as a plain script.
>
> Read the code against the repo's conventions: the style of `tools/sweep_bipartitions.py` and `tools/summarize_hierarchy_ablation.py`, `.claude/rules/offline-tools.md` and `.claude/rules/testing.md`. Check:
> - docstrings that state inputs, outputs and a run command;
> - consistent CLI options and defaults across the tools;
> - error messages that tell the user what to do;
> - duplicated logic that should live in one place;
> - unclear names, dead code and magic numbers without a comment;
> - Python 3.8 compatibility;
> - files that do too much.
>
> Do not edit anything. Report each finding with file:line, a severity (must-fix / should-fix / nit) and a concrete suggested change, most severe first. Also report what you ran and its output.

- [ ] **Step 2: Triage the findings with superpowers:receiving-code-review.** Fix every must-fix and should-fix finding that holds up, test-first for anything behavioural. Record each nit you skip, with the reason, for the final report.

- [ ] **Step 3: Re-run the full suite and every `--help`, then commit the fixes**

```bash
$ENVPY -m pytest -q tests
git add <the files actually changed>
git commit -m "$(cat <<'EOF'
Address the maintainability review of the split-analysis tools

<one line per fix>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Part 1 runs and the divided-mass DOCs entry

Run by the controller, not by a subagent: the long GPU jobs need monitoring, and the reading needs the numbers.

**Files:** outputs under `work_dirs/p3former_2xb1_3x_dso_ood_dump/` (untracked); modify `DOCs.md`.

- [ ] **Step 1: Single-class sweeps on the three sets.** First `nvidia-smi --query-gpu=index,memory.used --format=csv`. Then:

```bash
ENVPY=/home/khoadv/miniconda3/envs/p3former/bin/python
W=work_dirs/p3former_2xb1_3x_dso_ood_dump
nohup setsid bash -c "$ENVPY tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits --subsets singletons --out-dir $W/singletons > $W/singletons.out 2>&1; $ENVPY tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits_test $W/logits --subsets singletons --out-dir $W/singletons_test_cetran > $W/singletons_test_cetran.out 2>&1" < /dev/null > /dev/null 2>&1 &
nohup setsid bash -c "$ENVPY tools/sweep_bipartitions.py --backend torch --device cuda:1 $W/logits_test --subsets singletons --out-dir $W/singletons_test > $W/singletons_test.out 2>&1" < /dev/null > /dev/null 2>&1 &
```

Wait for `wrote .../bipartitions.log` in all three `.out` files (Monitor with an until-loop, no polling sleeps). Expected: each log holds the flat rows plus `s0..s23` rows.

- [ ] **Step 2: Divided mass**

```bash
nohup setsid bash -c "$ENVPY tools/divided_mass.py --backend torch --device cuda:0 > $W/divided_mass.out 2>&1" < /dev/null > /dev/null 2>&1 &
```

When it finishes, read `$W/divided_mass/summary.md`.
- Every check must be PASS. A FAIL on the Group MSP histogram checks means the bins are too coarse: raise `BINS_PER_DECADE` (500, then 1000) and rerun, with a regression test that pins the new value, then note it.
- A FAIL on the 21-split agreement means the two sweeps disagree, which is a bug. Stop and use superpowers:systematic-debugging.
- Note the robust count; 32 before the three missing single-class splits were scored.

- [ ] **Step 3: Figures**

```bash
$ENVPY tools/plot_divided_mass.py
$ENVPY tools/plot_divided_mass.py --threshold 0.001   # the deep threshold, for delta95 / FPR@95
```

Open `bubble_singletons_0.05.png`, `bubble_robust_0.05.png` and `rho_vs_threshold.png`, and look at them. Check for label collisions, bubbles cut by the axes and legend overlap. If there are problems, adjust `AREA_PER_UNIT`, `LABEL_GAPS` or the figure size in `tools/plot_divided_mass.py`, rerun the test file, and commit the tweak separately.

- [ ] **Step 4: Write the DOCs entry.** Append `## <date> — Why single-class splits beat flat: divided mass`, with `<date>` = `date +%F` on the day the runs finish. It contains:
  1. What was measured, in 3 sentences: m, δ, u, and the fact that divided % is Group MSP's TPR/FPR at δ.
  2. The commands of Steps 1–3.
  3. The check results (one line: all PASS, or the fix made).
  4. **Table A.** Single-class splits at δ = 0.05 on each set, from `summary.md`: class, OOD div %, ID div %, selectivity, δ95, ID div % at δ95, improvement. Keep every class that beats flat on any set, plus {vegetation}, {building} and {bicycle}.
  5. **Table B.** Spearman ρ at δ = 0.05 and at the δ with the largest |ρ| against improvement, for single-class and all splits, per set.
  6. **Table C.** The robust splits (top 10 by worst-set improvement) with their OOD / ID divided % on Test + Cetran.
  7. The figures' paths.
  8. **Reading** (bullets). Answer, with numbers from the tables:
     - (a) Which single-class splits beat flat, and where they sit relative to flat MSP's star? Is the selectivity above 1?
     - (b) Which statistic, and at which δ, tracks each metric delta best? Is it the share or the absolute-count precision?
     - (c) Does δ95 explain FPR@95? Use {bicycle} on Test, FPR@95 89.2, as the worked example.
     - (d) What do the robust splits share?
  9. The caveat: selection on the evaluation data, and the mechanical link to the ROC.

- [ ] **Step 5: Commit**

```bash
git add DOCs.md
git commit -m "$(cat <<'EOF'
Log the divided-mass analysis of the two-group splits

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: Part 2 runs, the resemblance DOCs entry, the progress log

Run by the controller.

**Files:** outputs under `work_dirs/.../features_*` (on `/mnt/sandisk`) and `resemblance*`; modify `DOCs.md`, `.claude/CLAUDE.md`, `.claude/rules/offline-tools.md` and `.claude/rules/testing.md`.

- [ ] **Step 1: Storage and symlinks**

```bash
S=/mnt/sandisk/khoadv/P3Former_OOD/p3former_2xb1_3x_dso_ood_dump
mkdir -p $S/features_cetran $S/features_test
ln -s $S/features_cetran $W/features_cetran
ln -s $S/features_test $W/features_test
```

- [ ] **Step 2: Extraction, one GPU per split, detached** (after `nvidia-smi`; each needs ~25 GB):

```bash
CFG=configs/p3former/p3former_2xb1_3x_dso_ood.py
CKPT=work_dirs/p3former_2xb1_3x_dso/epoch_36.pth
nohup setsid bash -c "CUDA_VISIBLE_DEVICES=0 $ENVPY tools/extract_point_features.py $CFG $CKPT --ann dso_infos_cetran.pkl --out-dir $W/features_cetran --check-dump $W/logits > $W/features_cetran.out 2>&1" < /dev/null > /dev/null 2>&1 &
nohup setsid bash -c "CUDA_VISIBLE_DEVICES=1 $ENVPY tools/extract_point_features.py $CFG $CKPT --ann dso_infos_test.pkl --out-dir $W/features_test --check-dump $W/logits_test > $W/features_test.out 2>&1" < /dev/null > /dev/null 2>&1 &
```

Expected: `meta.json` in both, with 980 and 2,625 frames, `max_logit_diff` within tolerance and `max_dump_diff` ≤ 0.05. Record both numbers and the sample counts.

- [ ] **Step 3: Resemblance, the sensitivity run, the figures**

```bash
$ENVPY tools/ood_class_resemblance.py --device cuda:0
$ENVPY tools/ood_class_resemblance.py --device cuda:0 --k 50 --out-dir $W/resemblance_k50
$ENVPY tools/plot_ood_class_resemblance.py
$ENVPY tools/plot_ood_class_resemblance.py --space appearance
$ENVPY tools/plot_feature_tsne.py
```

Open `profile_full.png`, `bubble_feature_singletons_full.png`, `hypothesis_full.png` and `tsne_full.png`, and look at them. Fix layout problems as in Task 9, Step 3.

- [ ] **Step 4: Write the DOCs entry.** Append `## <date> — OOD resemblance to the ID classes (feature space)`. It contains:
  1. The layer and the check numbers: `max_logit_diff`, `max_dump_diff`, frames and samples.
  2. The commands.
  3. **Table A.** The top 8 classes by r_OOD per set (full space): r_OOD, r_ID, contrast, and {c} | rest's improvement.
  4. **Table B.** Class-level ρ: r_OOD and contrast against improvement and OOD divided %, per set, full vs appearance, k = 10 vs 50.
  5. **Table C.** The hypotheses: ρ(improvement, R_A), ρ(improvement, E_A), ρ(improvement, OOD feature-divided %) and ρ(improvement, feature log ratio), per set and space.
  6. **Table D.** The robust splits' R_A and feature-divided % on Test + Cetran.
  7. The figures' paths.
  8. **Reading** (bullets):
     - (a) Which classes OOD points resemble on each set, and whether that matches the winning single-class splits (overhead-bridge / gate / truck);
     - (b) whether the resemblance survives without the positional embedding;
     - (c) the verdict on the hypothesis "together" against "cut through", with the {overhead-bridge} (+34.8 on test) vs {building, overhead-bridge} (+14.3) example;
     - (d) how this agrees with Part 1;
     - (e) the k = 50 stability;
     - (f) what the t-SNE shows (qualitative only).

- [ ] **Step 5: Update the project docs**
  - `.claude/rules/offline-tools.md`: one bullet per new tool (inputs, what it writes, the run command), and the `--subsets` note on `sweep_bipartitions.py`.
  - `.claude/rules/testing.md`: the new test count, from `$ENVPY -m pytest -q tests | tail -1`.
  - `.claude/CLAUDE.md`:
    - bump "Progress log (updated …)";
    - add the completed items (single-class sweeps, divided mass, feature samples, resemblance) with their dates;
    - add status rows for the divided mass and the resemblance, with the headline numbers;
    - replace the analysis plan of "Next step" with the findings, and what they leave open, in 3–6 bullets;
    - add a key-decisions bullet on the feature layer (`pe_features`, eval-split reference, kNN share).

- [ ] **Step 6: Full suite, then commit**

```bash
$ENVPY -m pytest -q tests
git add DOCs.md .claude/CLAUDE.md .claude/rules/offline-tools.md .claude/rules/testing.md
git commit -m "$(cat <<'EOF'
Log the OOD-resemblance analysis and update the progress log

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```
