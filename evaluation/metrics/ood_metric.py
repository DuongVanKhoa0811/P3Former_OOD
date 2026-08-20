"""Point-level OOD detection metric for SemanticKITTI-style datasets.

Ground truth is derived per point from the raw panoptic label kept in
``eval_ann_info['pts_instance_mask']`` (raw semantic id = label %
``seg_offset``): raw ids in ``ood_raw_ids`` are OOD (positive), points
whose mapped train label is not ``ignore_index`` are ID (negative), and
everything else (SemanticKITTI raw 0 ``unlabeled`` / 1 ``outlier``) is
excluded. Scores are read from ``pred_pts_seg['ood_<key>']`` as emitted by
``_P3FormerHead`` when ``ood_cfg`` is set.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS

from ..functional.ood_eval import ood_point_eval


@METRICS.register_module()
class _OODPointMetric(BaseMetric):
    """Point-level OOD AUROC / AP / FPR@95 over the whole split."""

    default_prefix = 'ood'

    def __init__(self,
                 ood_raw_ids: Sequence[int] = (52, 99),
                 seg_offset: int = 2**16,
                 ignore_index: int = 19,
                 score_keys: Sequence[str] = ('msp', 'maxlogit', 'odin',
                                              'energy', 'entropy'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ood_raw_ids = np.asarray(list(ood_raw_ids), dtype=np.int64)
        self.seg_offset = seg_offset
        self.ignore_index = ignore_index
        self.score_keys = tuple(score_keys)

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred = data_sample['pred_pts_seg']
            eval_ann = data_sample['eval_ann_info']
            raw_panoptic = np.asarray(eval_ann['pts_instance_mask'])
            raw_sem = raw_panoptic % self.seg_offset
            mapped = np.asarray(eval_ann['pts_semantic_mask'])
            ood = np.isin(raw_sem, self.ood_raw_ids)
            valid = (mapped != self.ignore_index) | ood

            scores = dict()
            for key in self.score_keys:
                pred_key = f'ood_{key}'
                if pred_key not in pred:
                    raise KeyError(
                        f"'{pred_key}' missing from pred_pts_seg. Set "
                        "model.decode_head.ood_cfg in the config so "
                        "_P3FormerHead emits OOD scores.")
                value = pred[pred_key]
                if hasattr(value, 'detach'):
                    value = value.detach().cpu().numpy()
                value = np.asarray(value)
                if value.shape[0] != mapped.shape[0]:
                    raise ValueError(
                        f'{pred_key} has {value.shape[0]} points but the '
                        f'ground truth has {mapped.shape[0]}')
                scores[key] = value[valid].astype(np.float32)
            self.results.append((ood[valid], scores))

    def compute_metrics(self, results: List[tuple]) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        labels = [labels for labels, _ in results]
        scores = {
            key: [scan_scores[key] for _, scan_scores in results]
            for key in self.score_keys
        }
        return ood_point_eval(scores, labels, logger=logger)
