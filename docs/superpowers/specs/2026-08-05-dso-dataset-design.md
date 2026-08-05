# P3Former on the DSO dataset — design

**Date:** 2026-08-05
**Branch:** `dso-dataset`
**Goal:** Train and evaluate P3Former (LiDAR panoptic segmentation) on the DSO dataset as the
in-distribution baseline for the OOD panoptic segmentation project.

## 1. Scope

In scope: DSO info-pkl generator, native PLY dataset/loading code, DSO configs, panoptic
evaluation on val/test, one full training run on GPU 1.
Out of scope (deliberately): OOD-specific methods, Cylinder3D pretraining (fallback only),
use of the raw `LiDAR_INS` point clouds, camera data, submission writers for DSO.

## 2. Data facts (measured, 2026-08-05)

- Annotations: `/mnt/ssd/khoadv/projects/OOD_PanSeg_3D/data/DSO_Dataset/Annotation_Final`,
  12 sequence dirs, 11,500 binary-little-endian PLY frames, ~129 GB.
- Per-point record (27 B): `float32 x, y, z, intensity, device_id; uint8 semantic;
  uint16 instance; uint8 isVisible; uint8 r, g, b`. The PLY is self-contained — points and
  panoptic labels in one file; the raw `LiDAR_INS` `.bin` files are not needed.
- Two merged LiDAR devices per frame (`device_id` 71 and 72, ~50/50); ~416k points/frame mean.
- Intensity is 1–255 (median 12). Height: ground ≈ −1.5..−0.8 m, structure up to ~75 m.
  Range: p50 ≈ 10 m, p95 ≈ 45 m, p99 ≈ 75 m. Pitch (atan2(z, rho)): p1 ≈ −8°, p99 ≈ +36°.
- Coverage of the 16 kept classes inside candidate cylindrical ranges (rho ≤ 50 m):
  z ∈ [−4, 2] → 54% (SemanticKITTI default, unusable); z ∈ [−3, 13] → ~95%.
- Instance ids: raw classes 1–7 carry real instances (car ≈ 9.8/frame, person ≈ 7.4/frame).
  Stuff classes carry one dummy instance id per frame — except traffic-sign (19) with
  ~17.6 real instance ids/frame, and traffic-cone (20) with ~4.2.

## 3. Splits (fixed by user; revised 2026-08-05)

Names below are the annotation-directory names. The user specified the split in raw
`LiDAR_INS` naming (hyphenated Data_Set1 names, "Pulau Ubin RouteN"); they map 1:1 onto
the annotation names used everywhere in this repo (hyphens → spaces,
"Pulau Ubin Route1/3" → "Ubin Route 1/3"). Do not "fix" the names back.

| Split | Sequences | Frames |
|---|---|---|
| train | (2024-04-29) One-North Route 1 Day, (2024-04-29) One-North Route 1 Night, (2024-05-13) One-North Route 2 Rain, (2024-05-13) Science Park Rain, (2024-05-17) Science Park Day, (2024-06-10) Chinatown Route 2 Day, (2024-06-10) Chinatown Route 2 Night, (2025-02-27) Tiong Bahru Rerun AM | 8,474 |
| val   | (2024-06-10) Chinatown Route 1 Day | 401 |
| test  | (2024-04-29) One-North Route 2 Day, (2025-03-02) Ubin Route 1, (2025-03-02) Ubin Route 3 | 2,625 |

Also generated: `dso_infos_mini.pkl` = first 32 frames of
`(2024-04-29) One-North Route 1 Day` (a train sequence), for smoke tests only.

## 4. Class mapping — 16 train classes, `ignore_index = 16`

Train-id order fixed by user: things first (0–6), then stuff (7–15).

| train id | class | raw id | role | palette (dataset RGB) |
|---|---|---|---|---|
| 0 | car | 3 | thing | 125, 46, 141 |
| 1 | bicycle | 7 | thing | 255, 127, 0 |
| 2 | motorcycle | 6 | thing | 255, 0, 0 |
| 3 | truck | 4 | thing | 118, 171, 47 |
| 4 | bus | 5 | thing | 161, 19, 46 |
| 5 | person | 1 | thing | 216, 82, 24 |
| 6 | rider | 2 | thing | 236, 176, 31 |
| 7 | road | 8 | stuff | 190, 190, 0 |
| 8 | sidewalk | 10 | stuff | 0, 0, 255 |
| 9 | building | 11 | stuff | 170, 0, 255 |
| 10 | fence | 14 | stuff | 84, 255, 0 |
| 11 | vegetation | 24 | stuff | 84, 0, 127 |
| 12 | trunk | 23 | stuff | 0, 255, 127 |
| 13 | terrain | 22 | stuff | 0, 170, 127 |
| 14 | pole | 18 | stuff | 255, 84, 0 |
| 15 | traffic-sign | 19 | stuff | 255, 170, 0 |

Mapped to ignore (16): 0 miss-label, 9 unpaved-road, 12 window, 13 net-fence,
15 overhead-bridge, 16 gate, 17 bus-stop, 20 traffic-cone, 21 drain, 27 airborne-raindrops,
28 barrier, 255 noise — **and every raw id not listed** (see §6 dataset class).

`labels_map` (config): `{0:16, 1:5, 2:6, 3:0, 4:3, 5:4, 6:2, 7:1, 8:7, 9:16, 10:8, 11:9,
12:16, 13:16, 14:10, 15:16, 16:16, 17:16, 18:14, 19:15, 20:16, 21:16, 22:13, 23:12, 24:11,
27:16, 28:16, 255:16}`, `max_label = 255`.

`learning_map_inv` (train → raw): `{0:3, 1:7, 2:6, 3:4, 4:5, 5:1, 6:2, 7:8, 8:10, 9:11,
10:14, 11:24, 12:23, 13:22, 14:18, 15:19, 16:0}`.

## 5. Repo data layout

- `data/dso/` — real directory (gitignored), holds the generated pkls.
- `data/dso/annotations` — symlink to `.../DSO_Dataset/Annotation_Final`.
- Info entries use relative paths `annotations/<seq>/<frame>.ply`; `data_root='data/dso/'`.
  No file is ever written into the annotation directory.

## 6. Components

### 6.1 Info generator — `tools/dataset_converters/dso_converter.py` + `create_data.py` wiring

`create_dso_info_file(pkl_prefix, save_path)` scans
`<save_path>/annotations/*/` for `*.ply` (sorted), assigns splits by the hardcoded
sequence-name lists from §3, and writes `dso_infos_{train,val,test,mini}.pkl` in the
SemanticKITTI info structure:
`{'metainfo': {'DATASET': 'DSO'}, 'data_list': [{'lidar_points': {'lidar_path':
'annotations/<seq>/<frame>.ply', 'num_pts_feats': 4}, 'pts_panoptic_mask_path':
'annotations/<seq>/<frame>.ply', 'sample_id': '<seq>/<frame-stem>'}, ...]}`.
It asserts the discovered sequence set is exactly the 12 known names and the split totals are
8,474 / 401 / 2,625. `tools/create_data.py` gets a `dso` branch:
`python tools/create_data.py dso --root-path data/dso --out-dir data/dso --extra-tag dso`.

### 6.2 Loading transform — `datasets/transforms/dso_loading.py`

One registered transform `_LoadDSOPointsAndAnnotations` (TRANSFORMS registry) replaces the
`LoadPointsFromFile` + `_LoadAnnotations3D` pair. It reads the PLY **once** (header parse +
`np.fromfile` structured dtype, file-size validated) and emits:

- `points`: `LiDARPoints`, N×4 float32 `(x, y, z, intensity/255.0)`. `device_id` is not
  included as a feature — keeps the model's `in_channels=6` untouched (future option).
- `pts_semantic_mask`: raw uint8 semantic ids as int64 (input to `PointSegClassMapping`).
- `pts_instance_mask`: int64 `(instance << 16) | raw_semantic` — SemanticKITTI packing, so
  voxel majority-vote GT, the head's `(instance << 16) + semantic` key derivation, and
  `id_offset=2**16` eval all work unchanged. **Instance bits are zeroed wherever
  raw_semantic ∉ {1..7}** (the thing raw ids): stuff GT must form one segment per class per
  frame — otherwise traffic-sign's ~17 instance ids/frame shatter its GT into segments the
  single stuff prediction can never match (IoU 1/17 < 0.5 → PQ collapses to 0 for that class).
- With `with_ann=False` (constructor arg) only `points` is produced.
- When `eval_ann_info` is present (test mode), both masks are copied into it — same mechanism
  as the proven SemanticKITTI path; `PointSegClassMapping` later overwrites the eval semantic
  mask with train ids exactly as it does today.

`_LaserMix`/`_PolarMix` reuse this transform in their `pre_transform` lists.

### 6.3 Dataset class — `datasets/dso_dataset.py`

`_DSODataset(Seg3DDataset)`, registered in DATASETS. METAINFO: the 16 class names/palette
from §4, `seg_valid_class_ids=tuple(range(16))`, `seg_all_class_ids=tuple(range(16))`.
`get_seg_label_mapping` builds a 256-entry lookup **default-filled with ignore_index 16**
(the `_SemanticKittiDataset` version zero-fills, which would silently map unlisted raw ids
to train id 0 = car), then applies `labels_map`. `parse_data_info` joins
`pts_panoptic_mask_path` with the data prefix, same as the SemanticKITTI class.

### 6.4 Evaluation — `evaluation/`

`_PanopticSegMetric` config: `thing_class_inds=[0..6]`, `stuff_class_inds=[7..15]`,
`min_num_points=50`, `id_offset=2**16`, `dataset_type='dso'`, `learning_map_inv` from §4.
`evaluation/functional/panoptic_seg_eval.py::add_panoptic_sample` gets a `'dso'` branch
identical to `'semantickitti'` (GT segment key = packed id with low bits replaced by train
semantic). DSO test/val pkls contain labels, so test-set evaluation is a normal metric run;
no submission writer for DSO (`format_results` with `dataset_type='dso'` stays
NotImplementedError — acceptable, never invoked without `submission_prefix`).

### 6.5 Configs

`configs/_base_/datasets/dso_panoptic_lpmix.py` — mirrors `semantickitti_panoptic_lpmix.py`:

- `dataset_type='_DSODataset'`, `data_root='data/dso/'`, 16 `class_names`, `labels_map`,
  `learning_map_inv`, `metainfo(max_label=255)`, `ignore_index=16`.
- Pipelines use `_LoadDSOPointsAndAnnotations` + `PointSegClassMapping`; train adds
  RandomChoice(`_LaserMix` p=0.2 / `_PolarMix` p=0.8, both prob=0.5 internally) →
  RandomFlip3D → GlobalRotScaleTrans → Pack3DDetInputs, all with SemanticKITTI values
  except: `_LaserMix pitch_angles=[-12, 45]` (measured DSO pitch span; the Velodyne default
  [−25, 3] would put ~half the scene into one band), `_PolarMix instance_classes=[0..6]`.
- Dataloaders: RepeatDataset(times=1) wrapping, train batch 4 (overridden to 2 below),
  num_workers 4; val/test batch 1, ann_files per §3.

`configs/p3former/p3former_1xb2_3x_dso.py` — mirrors the SemanticKITTI P3Former config:

- `point_cloud_range = [0, -3.14159265359, -3, 50, 3.14159265359, 13]` set in **both**
  `model.data_preprocessor.voxel_layer` and `model.decode_head` (the value is duplicated in
  the base model config). Grid `[480, 360, 32]` unchanged → 10.4 cm rho bins, 0.5 m z bins,
  ~95% of labeled points in-range; the preprocessor clamps the rest into boundary bins.
- Head: `num_classes=17`, `cls_channels=(256, 256, 17)`, `thing_class=[0,1,2,3,4,5,6]`,
  `stuff_class=[7,8,9,10,11,12,13,14,15]`, `ignore_index=16`. Other head/model settings
  (128 queries, 6 decoder layers, MPE, losses, assigners) unchanged.
- Schedule: AdamW lr 8e-4, weight-decay 0.01, 36 epochs, MultiStepLR milestones [24, 32]
  gamma 0.2, `val_interval=1`, CheckpointHook interval 5, `train_dataloader.batch_size=2`.
- `custom_imports`: the existing p3former/evaluation modules plus `datasets.dso_dataset`,
  `datasets.transforms.dso_loading`, `datasets.transforms.transforms_3d`.

## 7. Data flow (end to end)

PLY → `_LoadDSOPointsAndAnnotations` (points, raw semantic, packed+stuff-zeroed instance)
→ `PointSegClassMapping` (raw → 16 train ids / 16) → LaserMix/PolarMix (panoptic-aware)
→ flip/rot-scale → `_Det3DDataPreprocessor` cylindrical voxelization on [0,−π,−3, 50,π,13]
(train: voxel-majority semantic+instance GT; test: point2voxel map) → SegVFE →
`_Asymm3DSpconv` → `_P3FormerHead` (128 thing queries + 16 stuff/sem queries, Hungarian on
things only) → panoptic point predictions → `_PanopticSegMetric` (PQ/RQ/SQ/mIoU, things ids
0–6, stuff ids 7–15, min 50 pts, offset 2^16).

## 8. Error handling

- PLY reader validates magic, endianness, required fields, and header-vs-filesize
  consistency; a corrupt file raises with the file path in the message (no silent skip).
- Info generator hard-fails on unknown/missing sequence names or wrong frame totals.
- Unmapped raw semantic ids can never leak into training: the 256-entry lookup defaults to
  ignore (16).

## 9. Verification

1. `create_data.py dso` → pkl counts exactly 8,474 / 401 / 2,625 / 32.
2. Pipeline unit check (script, no GPU): run train pipeline on one frame; assert points
   shape N×4, intensity ∈ (0, 1]; assert mapped semantic histogram equals an independent
   numpy re-read of the same PLY; assert stuff points have `instance_mask >> 16 == 0` and
   thing instance counts match the raw file.
3. Smoke train: `CUDA_VISIBLE_DEVICES=1 python train.py configs/p3former/p3former_1xb2_3x_dso.py
  --cfg-options train_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl
  val_dataloader.dataset.dataset.ann_file=dso_infos_mini.pkl train_cfg.max_epochs=1` —
  must complete an epoch + val without OOM and print the 16-class metric table.
4. Full run on GPU 1; monitor first epochs for loss divergence (P3Former instability risk).
5. Final: `test.py` with best checkpoint on `dso_infos_test.pkl`
   (One-North Route 2 Day + Ubin Route 1 + Ubin Route 3, 2,625 frames).

## 10. Risks & fallbacks

- **OOM at batch 2** (416k pts/frame ≈ 3.5× SemanticKITTI): fall back to batch 1 +
  `accumulative_counts=2` in optim_wrapper; second resort `--amp`.
- **Training instability from scratch**: fall back to Cylinder3D backbone pretraining on DSO
  (README-recommended recipe; separate config, reusing this dataset base).
- **LaserMix band semantics with 2 merged LiDARs**: pitch bands are still geometrically
  well-defined w.r.t. the common origin; if augmentation hurts, drop LaserMix branch to
  PolarMix-only via the RandomChoice probabilities.
- **Val on a single 401-frame sequence** (Chinatown Route 1 Day): PQ will be noisy;
  acceptable — split fixed by user. The test split (2,625 frames) includes the two rural
  Pulau Ubin routes, a deliberate domain shift from the urban train/val scenes.
