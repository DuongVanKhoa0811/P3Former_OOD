---
paths:
  - "datasets/**/*.py"
  - "p3former/data_preprocessors/*.py"
  - "configs/_base_/datasets/*.py"
  - "tools/create_data.py"
  - "tools/dataset_converters/*.py"
---

# Data pipeline

1. `_SemanticKittiDataset` reads `semantickitti_infos_*.pkl`, and `_DSODataset` reads `dso_infos_*.pkl`.
2. Loading depends on the dataset:
   - SemanticKITTI: `LoadPointsFromFile` + `_LoadAnnotations3D` (`with_panoptic_3d=True`). The label file packs `(instance << 16) | semantic`.
   - nuScenes: packs the other way round, so semantic = `label // seg_offset`.
   - DSO: `_LoadDSOPointsAndAnnotations` (`datasets/transforms/dso_loading.py`) reads points and labels from one binary PLY per frame. It packs them the SemanticKITTI way but keeps instance ids only for `DSO_THING_RAW_IDS`, so every stuff class forms one segment.

   The full raw panoptic label is kept as `pts_instance_mask`, and the OOD metric derives its ground truth from it.
3. `PointSegClassMapping` maps raw ids to train ids. SemanticKITTI has 19 classes (things 0–7, stuff 8–18), ignore 19; the DSO class set is in `dso-dataset.md`. The head has one output per train class plus one for ignore (`num_classes=20` / `25`). Changing the class set changes the head, so checkpoints must be retrained.
4. `_LaserMix` / `_PolarMix` (`datasets/transforms/transforms_3d.py`) are panoptic-aware ports of the mmdet3d augmentations. They mix the instance masks too, and re-run the `pre_transform` loading pipeline on the mixed-in scan.
5. `_Det3DDataPreprocessor` (`p3former/data_preprocessors/data_preprocessor.py`) with `voxel_type='cylindrical'` voxelizes points onto a [480, 360, 32] polar grid. During training, `get_voxel_seg` builds the voxel-level GT by per-voxel majority vote. At test time it builds `point2voxel_map`, which projects voxel predictions (and OOD scores) back to points.
