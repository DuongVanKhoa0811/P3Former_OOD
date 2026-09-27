# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Fork of [P3Former](https://github.com/InternRobotics/P3Former) (DETR-style transformer for LiDAR point cloud panoptic segmentation, on the OpenMMLab mmdetection3d stack), used as the baseline for out-of-distribution (OOD) panoptic segmentation research (`origin` = DuongVanKhoa0811/P3Former_OOD, `upstream` = InternRobotics/P3Former). On top of upstream it adds:

- **DSO dataset** support: a 24-class LiDAR dataset with a native PLY loader, an info-pkl converter, and extra Cetran test sets.
- **Post-hoc point-level OOD scoring** from the auxiliary semantic branch: MSP, MaxLogit, ODIN, Energy and Entropy, each also in hierarchy-aware Group and Group-Normalised (GN) variants, plus an OOD metric.
- **Offline analysis tools** that recompute any score or class hierarchy from dumped per-point logits, without running the model.

Work happens on branch `ood-baselines`; `dso-dataset` is merged into it, and `duy/ood-baselines` belongs to a collaborator. `trash/`, `work_dirs/`, `data/`, `checkpoint/`, `papers/`, a root-level `CLAUDE.md` and `AGENTS.md` are gitignored.

## Rules

The detailed instructions are split by topic into `.claude/rules/`. Rules without `paths` frontmatter load at the start of every session; the scoped ones load when Claude reads a matching file.

| Rule | Topic |
| --- | --- |
| `environment.md` | Python interpreter, pinned stack, run from the repo root |
| `commands.md` | train, test, multi-GPU and data-preparation commands; outputs and checkpoints |
| `testing.md` | unit-test suite |
| `configs.md` | config naming, variants, generated configs, overrides |
| `experiment-log.md` | `DOCs.md` working log and reference documents |
| `machine-resources.md` | GPU, RAM and disk budgets; detached long jobs |
| `dso-dataset.md` | DSO class set, OOD classes, splits |
| `offline-tools.md` | analysis scripts in `tools/` |
| `torch-cuda-pitfalls.md` | TF32 and CUDA `bincount` in torch 1.10 (scoped: Python code) |
| `architecture/registry.md` | registry pattern and `custom_imports` |
| `architecture/data-pipeline.md` | datasets, loading, class mapping, voxelization (scoped: data code) |
| `architecture/model.md` | segmentor and P3Former head internals (scoped: `p3former/`) |
| `architecture/ood-pipeline.md` | how OOD scores flow from the head to the metric |
