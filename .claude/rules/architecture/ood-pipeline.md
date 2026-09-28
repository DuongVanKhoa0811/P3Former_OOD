# OOD scoring and evaluation (spans five files)

1. **Head.** `_P3FormerHead(ood_cfg=...)` requires `use_sem_loss=True`. In `predict` it keeps the first `num_ood_logits` channels of `sem_preds`, dropping the ignore channel (19 for SemanticKITTI, 24 for DSO), and calls `point_ood_scores`. The other `ood_cfg` keys are `odin_temperature` (1000) and `energy_temperature` (1).
2. **Scores.** `p3former/utils/ood_scores.py` computes MSP, MaxLogit, ODIN, Energy and Entropy (`OOD_SCORE_KEYS`) with the convention **higher = more OOD**, using `-max` (not `1 - max`) for the MSP family.
3. **Postprocess.** `_P3Former.postprocess_result` stores each score as `pred_pts_seg.ood_<key>`.
4. **Metric.** `_OODPointMetric` (`evaluation/metrics/ood_metric.py`) derives per-point ground truth:
   - OOD: `pts_instance_mask % 2**16` is in `ood_raw_ids`.
   - ID: the mapped label is not `ignore_index`.
   - Every other point is excluded.

   The OOD classes are SemanticKITTI raw 52/99 and DSO raw 17/28 (Stop/Others, which training ignores). The metric evaluates the keys in its `score_keys`, which default to the five flat scores, so a new score must be listed there in its config. AUROC, AP and FPR@95 come from 2^20-bin histograms (`evaluation/functional/ood_eval.py`) and are logged as `method | AUROC | AP | FPR@95` rows.

`_PanopticSegMetric.process` (`evaluation/metrics/panoptic_seg_metric.py`, PQ/RQ/SQ with `min_num_points=50`) is overridden to store only its two masks. It used to copy every `ood_*` array, which doubled the evaluator's RAM.
