# Point-level OOD detection baselines for P3Former on SemanticKITTI

Date: 2026-08-12
Status: approved

## Goal

Add three post-hoc point-level OOD detection baselines — MSP, ODIN, and Energy Score
(plus MaxLogit, which comes for free and disambiguates the request "MSP (= Max Logit)") —
to the P3Former codebase, and evaluate them on SemanticKITTI val with `other-structure`
and `other-object` as the OOD classes. No model retraining; the official checkpoint
`checkpoint/semantickitti_val_62.6.pth` is used as-is.

## Protocol (follows "Relative Energy Learning for LiDAR OOD detection", ISPRS 2026, §4.1.2/§4.2)

- Dataset: SemanticKITTI val split (sequence 08, 4071 scans), via the existing
  `semantickitti_infos_val.pkl`.
- Per-point ground truth, derived from the raw panoptic label
  (`eval_ann_info['pts_instance_mask']`, raw semantic = label `% 2**16`):
  - **OOD (positive, y=1)**: raw semantic id ∈ {52 (`other-structure`), 99 (`other-object`)}.
  - **ID (negative, y=0)**: mapped train label ∈ [0, 18] (i.e. `eval_ann_info['pts_semantic_mask'] != 19`).
  - **Excluded**: everything else — points whose mapped label is 19 (ignore) and whose raw id
    is not 52/99 (in practice raw 0 `unlabeled` and 1 `outlier`). These points are dropped
    before computing metrics.
- Metrics (point-level, threshold-free): **AUROC** ↑, **AP** (area under precision-recall,
  OOD = positive) ↑, **FPR@95** ↓ (fraction of ID points scored above the threshold that
  captures 95% of OOD points).
- Score convention: **higher score = more OOD** (matches REL Eq. 2).
- Object-level metrics are out of scope: SemanticKITTI outlier classes have no instance masks.

## Score definitions

All scores are computed from per-voxel class logits `z ∈ R^19` and projected to points
(see "Model-side changes"). `softmax_T(z) = exp(z/T) / Σ_j exp(z_j/T)`.

| Method   | Score (higher = more OOD)          | Hyperparameters (from docs/others/baselines/OOD_Baseline.pdf) |
|----------|------------------------------------|----------------------------------------------------------------|
| MSP      | `− max_c softmax_1(z)_c`           | — (Hendrycks & Gimpel, ICLR 2017)                              |
| MaxLogit | `− max_c z_c`                      | — (Hendrycks et al., ICML 2022)                                |
| ODIN     | `− max_c softmax_T(z)_c`           | T = 1000, ε = 0 (per VOS/SAFE convention; ε = 0 means **no** input-gradient perturbation) |
| Energy   | `− T · log Σ_c exp(z_c / T)`       | T = 1 (Liu et al., NeurIPS 2020; free energy, higher for OOD)  |
| Entropy  | `− Σ_c softmax_1(z)_c · log softmax_1(z)_c` | — (Shannon entropy of the softmax, natural log; added 2026-08-19 following `trash/Done/eval_ood_from_logits.py`) |

Numerical safety: softmax/logsumexp computed with the max-subtraction trick
(`torch.softmax` / `torch.logsumexp` already do this).

### Group and Group-Normalised variants (added 2026-08-21)

Hierarchy-aware versions of the five scores, following "Understanding Confidence
Fragmentation in OOD Detection for 3D LiDAR Semantic Segmentation"
(`papers/RelatedPapers/GroupPaper.pdf`, Eq. 2–8) as implemented in
`trash/Done/eval_ood_from_logits.py`; only the score arithmetic is adopted, everything else
(sign convention `−max`, FPR@95 implementation, ground-truth derivation, storage) stays as
above. Given a partition of the ID classes into groups `g` with sizes `K_g`
(`ood_cfg.class_groups`, a list of train-id lists; `None` = flat scores only):

- `P_g = Σ_{c∈g} softmax(z)_c` (and `P_g^T` at the ODIN temperature),
  `L^ML_g = max_{c∈g} z_c`, `L^E_g = T·log Σ_{c∈g} exp(z_c/T)`.
- Group: `group_msp = −max_g P_g`, `group_odin = −max_g P_g^T`,
  `group_maxlogit = −max_g L^ML_g` (= MaxLogit for a partition),
  `group_energy = −max_g L^E_g`, `group_entropy = −Σ_g P_g log P_g` (P clipped at 1e-12).
- Group Normalisation: `Q_g = [P_g − K_g/C]_+` (C = number of ID logits), likewise `Q_g^T`;
  `gn_msp = −max_g Q_g`, `gn_odin = −max_g Q_g^T`, `gn_maxlogit = −max_g (L^ML_g − log K_g)`,
  `gn_energy = −max_g (L^E_g − T·log K_g)`, `gn_entropy = −Σ_g (Q_g+ε) log(Q_g+ε)`, ε = 1e-12
  (the script's epsilon placement; the paper's Eq. 7 writes `Q_g log(Q_g+ε)`, a ≤1e-10
  difference).
- Keys: `group_{msp,maxlogit,odin,energy,entropy}`, `gn_{…}`; `_OODPointMetric` evaluates
  every `ood_*` key the model emits (default `score_keys=None`), canonical order
  `ALL_SCORE_KEYS`.
- Hierarchies: SemanticKITTI = the paper's Table 2 (vehicle / human / ground / construction /
  nature / object); DSO = the script's `dso24` preset (same six groups over the 24 DSO ids).

### Logit source

The head's auxiliary semantic branch (`sem_preds` in `_P3FormerHead.forward`, produced by
`sem_queries` and trained with CrossEntropy + Lovasz on voxel labels). It is already computed
during `predict()`'s forward pass and currently discarded. Only its first **19 channels** are
used: channel 19 is the "unlabeled" slot that is never a supervision target (CE uses
`ignore_index=19`), so including it would add an implicitly-learned void detector to what are
supposed to be closed-set baselines. The channel count is taken from config, not hardcoded.

Decision (user-approved): use the aux semantic branch, not query-composed pseudo-logits.
P3Former's query classification head is sigmoid/focal-trained, so softmax-based scores on it
would be ill-founded. The `ood_cfg` structure should not preclude adding another logit source
later, but no such source is implemented now (YAGNI).

## Model-side changes

1. **New file `p3former/utils/ood_scores.py`** — pure functions, no registry, unit-testable
   without building a model:
   - `compute_ood_scores(logits: Tensor[V, C], odin_temperature: float = 1000.0,
     energy_temperature: float = 1.0) -> Dict[str, Tensor[V]]`
     returning keys `msp`, `maxlogit`, `odin`, `energy`.
2. **`_P3FormerHead`** (`p3former/decode_heads/p3former_head.py`):
   - New constructor arg `ood_cfg: dict | None = None` (default `None` → behavior identical
     to today for every existing config). Expected keys:
     `num_ood_logits` (19), `odin_temperature` (1000.0), `energy_temperature` (1.0).
   - `predict()`: bind the currently-discarded 4th forward output (`sem_preds`). When
     `ood_cfg` is set, for each sample: slice `sem_preds[b][:, :num_ood_logits]`, compute the
     score dict, index every score with the same `point2voxel_map` used for the panoptic
     projection, and return per-point numpy score dicts alongside the existing predictions.
   - `use_sem_loss=True` is required when `ood_cfg` is set (assert with a clear message).
3. **`_P3Former`** (`p3former/segmentors/p3former.py`):
   - `predict()` / `postprocess_result()` pass the optional score dicts through and store each
     score as `ood_msp`, `ood_maxlogit`, `ood_odin`, `ood_energy` in the `pred_pts_seg`
     `PointData` (same point count as `pts_semantic_mask`, so `PointData`'s length invariant
     holds). mmdet3d's `SegMetric.process` already forwards every `pred_pts_seg` key to the
     evaluators, and `_PanopticSegMetric` ignores unknown keys — no change needed there.

## Evaluation-side changes

1. **New file `evaluation/functional/ood_eval.py`** — numpy-only:
   - `ood_point_eval(scores_per_scan, labels_per_scan, method_names, logger) -> dict`
     computing AUROC, AP, FPR@95 per method.
   - Implementation: single pass to get per-method global min/max, then accumulate two
     histograms (ID and OOD) with **2^20 bins** per method, then tie-aware AUROC
     (trapezoidal over the ROC built from cumulative histogram counts), AP (step-wise
     precision-recall, sklearn-style), and FPR@95 (first threshold with TPR ≥ 0.95).
     Rationale: val has ~5·10^8 labeled points; histograms avoid multi-GB sorts, and with
     2^20 bins the quantization error on these metrics is negligible (< 1e-4). No sklearn
     dependency (pinned Python 3.8 env).
2. **New file `evaluation/metrics/ood_metric.py`** — `_OODPointMetric(BaseMetric)`:
   - Constructor: `ood_raw_ids=[52, 99]`, `seg_offset=2**16`, `ignore_index=19`,
     `score_keys=('msp', 'maxlogit', 'odin', 'energy')`, `collect_device='cpu'`. Each score
     key `k` is read from the prediction as `pred_pts_seg['ood_{k}']` (the model-side names).
   - `process()`: per scan, derive `raw_sem = eval_ann['pts_instance_mask'] % seg_offset`,
     `mapped = eval_ann['pts_semantic_mask']` (train ids 0–19),
     `ood = isin(raw_sem, ood_raw_ids)`, `valid = (mapped != ignore_index) | ood`; store
     `(scores[valid].astype(float32) per method, ood[valid])`. Assert score keys are present
     with a message pointing at `decode_head.ood_cfg` if missing.
   - `compute_metrics()`: call `ood_point_eval`, log a table (one row per method: AUROC, AP,
     FPR@95) plus total ID/OOD point counts, and return a flat dict
     (`{method}_AUROC`, `{method}_AP`, `{method}_FPR95`) for mmengine logging.

## Config

**New file `configs/p3former/p3former_8xb2_3x_semantickitti_ood.py`**:
- `_base_ = ['./p3former_8xb2_3x_semantickitti.py']`
- `model = dict(decode_head=dict(ood_cfg=dict(num_ood_logits=19, odin_temperature=1000.0,
  energy_temperature=1.0)))`
- `val_evaluator` / `test_evaluator`: a **list** of the existing `_PanopticSegMetric` config
  (unchanged, so the same run sanity-reports PQ) and the new `_OODPointMetric`.
- `custom_imports`: base list plus `evaluation.metrics.ood_metric` (the head/segmentor and
  `p3former.utils.ood_scores` are pulled in by the existing imports/module code).

Intended run (GPU 0; GPU 1 is training):

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_8xb2_3x_semantickitti_ood.py checkpoint/semantickitti_val_62.6.pth
```

## Verification

1. **Standalone checks (no GPU/data)**: a small script under `tests/` exercising
   - `compute_ood_scores` on hand-built logits (e.g. uniform logits → MSP = −1/19;
     one dominant logit → all four scores rank it as most-ID; ODIN@T=1000 ordering matches
     `(max z − mean z)` ranking on random logits; energy equals `−logsumexp`).
   - `ood_point_eval` against exact brute-force AUROC/AP/FPR95 on random small inputs
     (including ties and the all-one-class edge case).
2. **End-to-end**: run the command above; expect
   - PQ table identical to the 2026-08-05 run (62.63) — proves the panoptic path is untouched;
   - OOD table with plausible values (REL Table 3 reports MaxLogit at AUROC 84.9 /
     FPR@95 47.6 / AP 53.9 on a Mask4Former backbone — same order of magnitude expected,
     not the same numbers);
   - all four methods produce distinct, finite values; ID+OOD point counts logged and
     OOD fraction in the low percents.
3. Record commands and results in `DOCs.md` (repo convention).

## Non-goals

- No nuScenes support (user: "right now focus on the semantickitti dataset").
- No training-based OOD methods (REL, VOS, void classifier…) — this spec is baselines only.
- No ODIN input perturbation (ε=0 per the baseline PDF; would require input gradients
  through spconv voxelization).
- No object-level OOD metrics (no instance masks for outlier classes).
- No change to training behavior of any existing config.
