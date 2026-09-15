#!/usr/bin/env python
"""Plot ID vs OOD Group-MSP score distributions, per hierarchy group.

Reads a logits dump written by ``_OODLogitsDumpMetric`` (config
``p3former_2xb1_3x_dso_ood_dump.py``) and recomputes the Group-MSP score
``-max_g P_g`` (P_g = softmax mass of group g) for one or more class
hierarchies, entirely offline. Following trash/ID_OOD_scores.png, each
hierarchy is a column: one row per group with the score densities of the ID
and OOD points *claimed* by that group (the points whose largest group mass
is this group, i.e. whose final score this group sets), then a bottom row
with the combined distribution over all points. The x axis is the group
uncertainty ``1 - max_g P_g`` on a log scale (a monotone transform of the
Group-MSP score, so every ranking metric is unchanged; on a linear axis
everything collapses into a spike at full confidence). Each curve is
normalised to its own point count (ID outnumbers OOD ~30x). AUROC / AP /
FPR@95 of the recomputed score are printed and drawn on the bottom row;
they should match the ``*_group_msp`` rows logged by the dump run up to
the float16 rounding of the stored logits.

Hierarchy names follow tools/make_dso_hierarchy_variants.py: ``p_`` +
blocks of group initials (v vehicle, h human, g ground, c construction,
n nature, o object), e.g. ``p_v_hgcno`` = {vehicle} | {rest} and
``p_v_h_g_c_n_o`` = the current six-group hierarchy.

Run from the repo root:
    python tools/plot_ood_score_distributions.py \\
        work_dirs/p3former_2xb1_3x_dso_ood_dump/logits \\
        --hierarchies p_v_h_g_c_n_o p_v_hgcno
"""
import argparse
import glob
import os.path as osp
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from make_dso_hierarchy_variants import GROUPS  # noqa: E402
from evaluation.functional.ood_eval import binary_ood_metrics  # noqa: E402

CURRENT = 'p_' + '_'.join(GROUPS)  # the six-group hierarchy, p_v_h_g_c_n_o
# Uncertainties 1 - max_g P_g below this land in the leftmost bin.
U_MIN = 1e-7


def parse_hierarchy(name):
    """``'p_vh_gcno'`` -> ``[('vh', [ids...]), ('gcno', [ids...])]``."""
    if not name.startswith('p_') or len(name) <= 2:
        raise ValueError(
            f"hierarchy name must look like 'p_v_hgcno', got '{name}'")
    seen = set()
    groups = []
    for block in name[2:].split('_'):
        if not block:
            raise ValueError(f"empty group block in '{name}'")
        ids = []
        for initial in block:
            if initial not in GROUPS:
                raise ValueError(f"unknown group initial '{initial}' in "
                                 f"'{name}' (have {'/'.join(GROUPS)})")
            if initial in seen:
                raise ValueError(f"group '{initial}' appears twice in "
                                 f"'{name}'")
            seen.add(initial)
            ids.extend(GROUPS[initial][1])
        groups.append((block, ids))
    return groups


def softmax(logits):
    z = logits.astype(np.float32)
    z -= z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z


def group_msp(probs, groups):
    """Per-point Group-MSP score and claiming group.

    Returns ``(score, claimed)``: ``score = -max_g P_g`` (higher = more OOD)
    and ``claimed = argmax_g P_g``, both [N].
    """
    mass = np.stack([probs[:, ids].sum(axis=1) for _, ids in groups], axis=1)
    return -mass.max(axis=1), mass.argmax(axis=1)


class HierarchyDistributions:
    """Streaming per-claiming-group ID/OOD histograms of the score."""

    def __init__(self, name, bins):
        self.name = name
        self.groups = parse_hierarchy(name)
        self.bins = bins
        # equal-width bins in log10(1 - max_g P_g), shared by every panel
        self.edges = np.logspace(np.log10(U_MIN), 0.0, bins + 1)
        # [label (0 ID / 1 OOD), claiming group, bin]
        self.hist = np.zeros((2, len(self.groups), bins), dtype=np.int64)
        self.score_chunks = []
        self.label_chunks = []
        self.metrics = None

    def update(self, probs, ood):
        score, claimed = group_msp(probs, self.groups)
        # uncertainty u = 1 - max_g P_g = score + 1, binned in log space
        u = np.clip(score + 1.0, U_MIN, 1.0)
        decades = -np.log10(U_MIN)
        b = np.minimum(
            ((np.log10(u) + decades) / decades * self.bins).astype(np.int64),
            self.bins - 1)
        flat = claimed * self.bins + b
        size = len(self.groups) * self.bins
        for lab, mask in ((0, ~ood), (1, ood)):
            self.hist[lab] += np.bincount(
                flat[mask], minlength=size).reshape(len(self.groups),
                                                    self.bins)
        self.score_chunks.append(score.astype(np.float32))
        self.label_chunks.append(ood)

    def finalise(self):
        """Exact AUROC/AP/FPR@95 of the recomputed score; frees the chunks."""
        self.metrics = binary_ood_metrics(self.score_chunks,
                                          self.label_chunks)
        self.score_chunks = self.label_chunks = None
        return self.metrics


def accumulate(files, hiers):
    n_id = n_ood = 0
    for i, path in enumerate(files):
        with np.load(path) as data:
            valid = data['valid']
            probs = softmax(data['logits'][valid])
            ood = data['ood'][valid]
        for hier in hiers:
            hier.update(probs, ood)
        n_ood += int(ood.sum())
        n_id += int(ood.size) - int(ood.sum())
        if (i + 1) % 100 == 0 or i + 1 == len(files):
            print(f'  {i + 1}/{len(files)} frames', flush=True)
    return n_id, n_ood


def _draw(ax, edges, hist_id, hist_ood, label):
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric bin centers
    for hist, color, lab in ((hist_id, 'tab:blue', 'ID'),
                             (hist_ood, 'tab:red', 'OOD')):
        total = hist.sum()
        if total == 0:
            continue
        # fraction per (equal-log-width) bin: unit area on the log axis
        density = hist / total
        ax.fill_between(centers, density, color=color, alpha=0.3, lw=0)
        ax.plot(centers, density, color=color, lw=1.3, label=lab)
    ax.set_xscale('log')
    ax.set_xlim(U_MIN, 1.0)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=11)
    for spine in ('top', 'right', 'left'):
        ax.spines[spine].set_visible(False)


def plot(hiers, n_id, n_ood, out, title):
    nrows = max(len(h.groups) for h in hiers) + 1
    fig = plt.figure(figsize=(6.5 * len(hiers), 1.5 * nrows + 1.4))
    gs = GridSpec(nrows, len(hiers), figure=fig, hspace=0.6, wspace=0.3)
    note_kwargs = dict(ha='right', va='top', fontsize=8, color='0.35')
    for col, hier in enumerate(hiers):
        for row, (label, _) in enumerate(hier.groups):
            ax = fig.add_subplot(gs[row, col])
            _draw(ax, hier.edges, hier.hist[0, row], hier.hist[1, row],
                  label)
            # share = (f'claims {hier.hist[0, row].sum() / max(n_id, 1):.1%} '
            #          f'of ID, {hier.hist[1, row].sum() / max(n_ood, 1):.1%} '
            #          'of OOD')
            share = ""
            ax.text(0.99, 0.97, share, transform=ax.transAxes, **note_kwargs)
            if row == 0:
                suffix = ' (current hierarchy)' if hier.name == CURRENT else ''
                ax.set_title(hier.name + suffix, fontsize=12)
        # combined score, directly below the group rows (as in the sketch)
        ax = fig.add_subplot(gs[len(hier.groups), col])
        _draw(ax, hier.edges, hier.hist[0].sum(axis=0),
              hier.hist[1].sum(axis=0), hier.name)
        m = hier.metrics
        ax.text(0.99, 0.97, f'AUROC {100 * m["auroc"]:.2f}   '
                f'AP {100 * m["ap"]:.2f}   FPR@95 {100 * m["fpr95"]:.2f}',
                transform=ax.transAxes, **note_kwargs)
        ax.set_xlabel('group uncertainty  $1 - \\max_g P_g$  '
                      '(log scale, higher = more OOD)', fontsize=9)
    fig.axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.text(0.005, 0.005,
             'v vehicle · h human · g ground · c construction · n nature · '
             'o object — group rows hold the points whose largest group '
             f'mass is that group; {n_id:,} ID / {n_ood:,} OOD points',
             fontsize=8, color='0.4')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dump_dir', help='directory of _OODLogitsDumpMetric npzs')
    ap.add_argument('--hierarchies', nargs='+',
                    default=[CURRENT, 'p_v_hgcno'],
                    help='one column per hierarchy (default: %(default)s)')
    ap.add_argument('--bins', type=int, default=400,
                    help='histogram bins over the log-scaled uncertainty '
                    f'range [{U_MIN:g}, 1]')
    ap.add_argument('--out', default=None,
                    help='output PNG (default: id_ood_group_msp.png next to '
                    'the dump dir)')
    ap.add_argument('--title',
                    default='ID vs OOD Group-MSP score distributions')
    args = ap.parse_args()

    files = sorted(glob.glob(osp.join(args.dump_dir, '*.npz')))
    if not files:
        raise FileNotFoundError(f'no .npz dumps in {args.dump_dir}')
    print(f'{len(files)} frames in {args.dump_dir}')
    hiers = [HierarchyDistributions(name, args.bins)
             for name in args.hierarchies]
    n_id, n_ood = accumulate(files, hiers)
    print(f'{n_id} ID / {n_ood} OOD points')
    header = f'{"hierarchy":>16} | {"AUROC":>8} | {"AP":>8} | {"FPR@95":>8}'
    print(header)
    print('-' * len(header))
    for hier in hiers:
        m = hier.finalise()
        print(f'{hier.name:>16} | {100 * m["auroc"]:8.2f} | '
              f'{100 * m["ap"]:8.2f} | {100 * m["fpr95"]:8.2f}')
    out = args.out or osp.join(osp.dirname(osp.normpath(args.dump_dir)),
                               'id_ood_group_msp.png')
    plot(hiers, n_id, n_ood, out, args.title)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
