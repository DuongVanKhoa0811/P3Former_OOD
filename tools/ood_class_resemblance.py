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
