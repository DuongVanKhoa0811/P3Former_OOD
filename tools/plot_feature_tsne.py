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
