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
