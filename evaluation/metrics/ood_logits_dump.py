"""Dump per-point semantic logits (plus OOD ground truth) to disk.

Companion to ``_OODPointMetric``: with ``ood_cfg.save_logits=True`` the head
attaches the float16 per-point logits it scored as
``pred_pts_seg['sem_logits']``; this metric writes one ``.npz`` per frame
(``logits`` float16 [N, C], ``ood`` / ``valid`` bool [N], ``mapped`` train
ids int16 [N], ``lidar_path`` str) so any OOD score or class hierarchy can
be recomputed offline — see tools/plot_ood_score_distributions.py — without
another GPU pass. ``ood`` / ``valid`` are derived exactly as in
``_OODPointMetric``.
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from mmengine.dist import get_rank
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS

_PRED_KEY = 'sem_logits'


@METRICS.register_module()
class _OODLogitsDumpMetric(BaseMetric):
    """Write one ``<out_dir>/r<rank>_<idx>.npz`` per evaluated frame.

    Args:
        out_dir: directory for the dump; created if missing, but must not
            already contain ``.npz`` files (stale frames from an earlier run
            would silently mix into offline analyses).
        ood_raw_ids / seg_offset / ignore_index: ground-truth derivation,
            same semantics as ``_OODPointMetric``.
    """

    default_prefix = 'ood_dump'

    def __init__(self,
                 out_dir: str,
                 ood_raw_ids: Sequence[int] = (52, 99),
                 seg_offset: int = 2**16,
                 ignore_index: int = 19,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.out_dir = out_dir
        self.ood_raw_ids = np.asarray(list(ood_raw_ids), dtype=np.int64)
        self.seg_offset = seg_offset
        self.ignore_index = ignore_index
        self._num_saved = 0
        os.makedirs(out_dir, exist_ok=True)
        stale = [f for f in os.listdir(out_dir) if f.endswith('.npz')]
        if stale:
            raise FileExistsError(
                f'{out_dir} already holds {len(stale)} .npz frames from an '
                'earlier dump; delete them or point out_dir elsewhere')

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        rank = get_rank()
        for data_sample in data_samples:
            pred = data_sample['pred_pts_seg']
            if _PRED_KEY not in pred:
                raise KeyError(
                    f"'{_PRED_KEY}' missing from pred_pts_seg. Set "
                    'model.decode_head.ood_cfg.save_logits=True so '
                    '_P3FormerHead emits the per-point logits.')
            logits = np.asarray(pred[_PRED_KEY])
            eval_ann = data_sample['eval_ann_info']
            raw_panoptic = np.asarray(eval_ann['pts_instance_mask'])
            raw_sem = raw_panoptic % self.seg_offset
            mapped = np.asarray(eval_ann['pts_semantic_mask'])
            if logits.shape[0] != mapped.shape[0]:
                raise ValueError(
                    f'{_PRED_KEY} has {logits.shape[0]} points but the '
                    f'ground truth has {mapped.shape[0]}')
            ood = np.isin(raw_sem, self.ood_raw_ids)
            valid = (mapped != self.ignore_index) | ood
            path = os.path.join(self.out_dir,
                                f'r{rank}_{self._num_saved:06d}.npz')
            np.savez(
                path,
                logits=logits.astype(np.float16),
                ood=ood,
                valid=valid,
                mapped=mapped.astype(np.int16),
                lidar_path=str(data_sample.get('lidar_path', '')))
            self._num_saved += 1
            self.results.append(int(mapped.shape[0]))

    def compute_metrics(self, results: List[int]) -> Dict[str, float]:
        logger = MMLogger.get_current_instance()
        logger.info(f'_OODLogitsDumpMetric: {len(results)} frames '
                    f'({int(sum(results))} points) dumped to {self.out_dir}')
        return dict(frames=len(results), points=int(sum(results)))
