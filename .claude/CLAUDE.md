# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fork of [P3Former](https://github.com/InternRobotics/P3Former) (DETR-style transformer for LiDAR point cloud panoptic segmentation, on the OpenMMLab mmdetection3d stack), used as the baseline for out-of-distribution (OOD) panoptic segmentation research (`origin` = DuongVanKhoa0811/P3Former_OOD, `upstream` = InternRobotics/P3Former). On top of upstream it adds:

- **DSO dataset** support: a 24-class LiDAR dataset with a native PLY loader, an info-pkl converter, and extra Cetran test sets.
- **Post-hoc point-level OOD scoring** from the auxiliary semantic branch: MSP, MaxLogit, ODIN, Energy and Entropy, plus an OOD metric.

This is branch `ood-baselines/flat`, the base of the OOD research branches; `rules/branches.md` lists them and how changes move between them. `trash/`, `work_dirs/`, `data/`, `checkpoint/`, `papers/`, a root-level `CLAUDE.md` and `AGENTS.md` are gitignored.

## Rules

The detailed instructions are split by topic into `.claude/rules/`. Rules without `paths` frontmatter load at the start of every session; the scoped ones load when Claude reads a matching file.

| Rule | Topic |
| --- | --- |
| `branches.md` | the OOD branches and how changes move between them |
| `environment.md` | Python interpreter, pinned stack, run from the repo root |
| `commands.md` | train, test, multi-GPU and data-preparation commands; outputs and checkpoints |
| `testing.md` | unit-test suite |
| `configs.md` | config naming, variants, overrides |
| `experiment-log.md` | `DOCs.md` working log and reference documents |
| `machine-resources.md` | GPU, RAM and disk budgets; detached long jobs |
| `dso-dataset.md` | DSO class set, OOD classes, splits |
| `torch-cuda-pitfalls.md` | TF32 and CUDA `bincount` in torch 1.10 (scoped: Python code) |
| `architecture/registry.md` | registry pattern and `custom_imports` |
| `architecture/data-pipeline.md` | datasets, loading, class mapping, voxelization (scoped: data code) |
| `architecture/model.md` | segmentor and P3Former head internals (scoped: `p3former/`) |
| `architecture/ood-pipeline.md` | how OOD scores flow from the head to the metric |

## Progress log (updated 2026-09-29)

Numbers are AUROC / AP / FPR@95 in %. The dates in brackets are the `DOCs.md` entries with the commands and full tables.

### Completed

- **Flat OOD baselines**: MSP, MaxLogit, ODIN and Energy, then Entropy, computed from the auxiliary semantic branch. They were evaluated on SemanticKITTI val (official and our 2xb1 checkpoint) and on DSO test and test + Cetran [08-12, 08-19, 08-21].
- **Evaluator memory**: `_PanopticSegMetric` stores only its two masks instead of copying every `ood_*` array.
- **Branch split** (2026-09-29): the old `ood-baselines` became this branch plus `ood-baselines/grouping`, which holds the Group / GN scores, the hierarchy ablation, the logit dumps and the bipartition sweep. See `rules/branches.md`.

### Current status

| Component | Status |
| --- | --- |
| Trained models | `work_dirs/p3former_2xb1_3x_{dso,semantickitti}/epoch_36.pth`. DSO test PQ 46.50, mIoU 48.66. SemanticKITTI val PQ 60.32, against 62.63 for the official checkpoint. |
| Online OOD scoring and metric | Complete; 39 tests pass. Energy is the best flat score on DSO test + Cetran, 92.88 / 35.55 / 37.94. On SemanticKITTI val with the 2xb1 checkpoint, ODIN is best, 91.46 / 26.94 / 38.33. |

### Next step: OCCUQ on `ood-baselines/occuq`

Port the ideas of OCCUQ (Heidrich, Beemelmanns et al., ICRA 2025; `papers/RelatedPapers/OCCUQ_*.pdf`). Start with superpowers brainstorming → spec → plan. What matters for the port:

- OCCUQ is DDU for 3D occupancy. The prediction head becomes spectrally normalised residual MLP blocks, then one full-covariance Gaussian per class is fitted on the head's penultimate per-voxel features over the **training** set. Epistemic score = −log Σ_c π_c N(z; μ_c, Σ_c); aleatoric = softmax entropy.
- In P3Former the analogous feature is `pe_features` [V, 256] in `p3former/decode_heads/p3former_head.py`, the input to the bias-free `sem_queries` classifier. A post-hoc variant runs on the existing checkpoints; the faithful one fine-tunes a spectrally normalised side branch and leaves PQ unchanged.
- It needs no class hierarchy, only the flat plumbing: `ood_cfg`, `point_ood_scores`, `postprocess_result`, and `_OODPointMetric` with the new key listed in `score_keys`.
- `tools/ood_distance.py` on `origin/duy/ood-baselines` already captures `pe_features` and fits per-class Gaussians with a tied covariance, but on the split it scores, with that split's labels. Fit on the training split instead.

### Key decisions and how they were handled

- **Score source.** Scores are post-hoc, from the auxiliary semantic branch's first `num_ood_logits` channels with the ignore channel dropped. There is no retraining; this was the approved design on 08-12.
- **Protocol.** REL-style point-level AUROC / AP / FPR@95, where a higher score means more OOD.
  - OOD classes: SemanticKITTI raw 52/99 and DSO raw 17/28. Other ignored points are excluded.
  - Hyperparameters come from `OOD_Baseline.pdf`: ODIN T=1000 with ε=0, and Energy T=1.
- **Sign convention.** The MSP family uses `-max` rather than `1 - max`, so every score is higher for more OOD.
- **Memory.**
  - The metrics are histogram-based (2^20 bins).
  - The PQ metric stores only its two masks.
- **Long jobs.** They are started detached (`nohup setsid`), after a session interruption killed two background sweeps.
- **Selection bias.** A choice made on one evaluation split (a hyperparameter, a feature, a class split) must be confirmed on data it was not made on before it is reported.
- **Git.** Commits and pushes happen only when asked. Pushes to protected branches use the owner bypass of the pull-request rule (see `rules/branches.md`). `CLAUDE.md` lives in `.claude/` and is tracked.

### Open housekeeping

- The docstrings in `p3former/utils/ood_scores.py` still cite the deleted `trash/Done/eval_ood_from_logits.py`.
- Cleanup candidates are still on disk:
  - `work_dirs/less_use/p3former_{1xb2,4xb1}_3x_semantickitti/`, 8.3 GB.
  - The intermediate checkpoints (epochs 5–35) of the two 2xb1 runs, 12.4 GB.
