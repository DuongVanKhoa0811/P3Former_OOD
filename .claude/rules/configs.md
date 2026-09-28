# Configs

Names follow `configs/p3former/p3former_<G>x<B>_3x_<dataset>[_variant].py`, meaning G GPUs × B samples per GPU with the same recipe. The `2xb1` configs are the ones actually trained: the same effective batch as `1xb2` at roughly half the per-GPU memory. Variants:

- `_ood`: adds `ood_cfg` to the head and an `_OODPointMetric` next to the PQ metric.
- `_trainval`: trains on train + val.
- `_submit`: test-set submission. It loads no annotations and packs `lidar_path` via `_Pack3DDetInputs`. Its evaluator writes SemanticKITTI `.label` files to `semantickitti_submission/` and returns None, so the crash at the end of a submission run is expected. Its test `ann_file` is `semantickitti_infos_mini.pkl`; change it to the real test split for an actual submission.

`configs/cylinder3d/` pretrains the backbone semantic-only. Upstream recommends it to stabilise P3Former training.

## Editing and overriding configs

- Dataloaders wrap datasets in `RepeatDataset(times=1)`, so dataset overrides need the nested form, `dataset=dict(dataset=dict(ann_file=...))`, or `test_dataloader.dataset.dataset.ann_file=...` on the command line.
- The dataset base configs set `batch_size=4`, but the P3Former configs override the train batch size.
- Two base dataset configs are panoptic and used by P3Former: `semantickitti_panoptic_lpmix.py` and `dso_panoptic_lpmix.py`. `semantickitti_lpmix.py` is semantic-only and used only by the Cylinder3D pretraining. Keep edits to the right one.
