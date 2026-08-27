# DOCs

Setup and usage notes for training P3Former on SemanticKITTI and DSO.

## Environment

Python 3.8 with a pinned OpenMMLab stack. (The base conda solver may fail on some
installs; `--solver=classic` avoids it.)

```bash
conda create -n p3former python=3.8 -y --solver=classic
conda activate p3former

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install openmim
mim install mmengine==0.7.4
mim install mmcv==2.0.0rc4
mim install mmdet==3.0.0
mim install mmdet3d==1.1.0
wget https://data.pyg.org/whl/torch-1.10.0%2Bcu113/torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install torch_scatter-2.0.9-cp38-cp38-linux_x86_64.whl
pip install yapf==0.40.1   # newer yapf breaks mmengine 0.7.4 (FormatCode 'verify' kwarg)
```

The repo is never installed as a package — **run every command from the repo root**.
Custom modules are imported by path via `custom_imports` in each config.

## Data

### SemanticKITTI

The code expects `data/semantickitti/sequences/...`; symlink your copy if it lives
elsewhere:

```bash
mkdir -p data
ln -s /path/to/SemanticKITTI data/semantickitti
python tools/create_data.py semantickitti --root-path data/semantickitti --out-dir data/semantickitti --extra-tag semantickitti
```

Produces `semantickitti_infos_{train,val,trainval,test}.pkl`.

### DSO

The converter expects `data/dso/annotations` to (sym)link the DSO `Annotation_Final`
directory, with one subdirectory per sequence:

```bash
mkdir -p data/dso
ln -s /path/to/Annotation_Final data/dso/annotations
python tools/create_data.py dso --out-dir data/dso --extra-tag dso
```

Produces `dso_infos_{train,val,test}.pkl`. Splits are fixed in
`tools/dataset_converters/dso_converter.py`: val is Chinatown Route 1 Day (401 frames),
the held-out test split is One-North Route 2 Day plus two rural Ubin routes (2,625
frames; a deliberate urban→rural domain shift). If the three optional Cetran
AV-test-centre sequences are present, two extra sets are written as well —
`dso_infos_cetran.pkl` (980 frames) and `dso_infos_test_cetran.pkl` (test + Cetran,
3,605 frames). They never enter train/val.

## DSO class set (24 classes)

- **Things (train 0–8):** car, bicycle, motorcycle, truck, bus, person, rider,
  traffic-sign (raw 19), traffic-cone (raw 20). Signs and cones carry real instance ids
  (~17.6 and ~4.2 per frame), so the loader keeps their instance bits
  (see `DSO_THING_RAW_IDS` in `datasets/transforms/dso_loading.py`).
- **Stuff (train 9–23):** paved-road, unpaved-road, sidewalk, building, window,
  perimeter-barrier, other-barrier, overhead-bridge, gate, pole-like-object, drain,
  terrain, trunks, vegetation, obscurant (ascending raw id).
- **Ignored (→ 24):** Noise (0), Stop (17), Others (28), Sky (29) and Water Body (30)
  (2D-only annotations), Unlabelled (255).

The head is 25-way (`num_classes=25`, `cls_channels=(256, 256, 25)`).

Note for evaluation: many traffic-cone instances fall below the `min_num_points=50` PQ
cutoff (~40 points per instance on average), so cone PQ reflects only the larger
instances. This is the standard SemanticKITTI convention and is left as is.

## Training

Single GPU, batch 2:

```bash
python train.py configs/p3former/p3former_1xb2_3x_dso.py
python train.py configs/p3former/p3former_1xb2_3x_semantickitti.py
```

Two GPUs, batch 1 each (same effective batch 2 and same recipe, roughly half the
per-GPU activation memory — the recommended setting for DSO, whose instance-dense
frames can spike the thing-mask loss well past the batch-2 steady state):

```bash
CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso.py 2
CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_semantickitti.py 2
```

Four GPUs, batch 1 each:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh configs/p3former/p3former_4xb1_3x_dso.py 4
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh configs/p3former/p3former_4xb1_3x_semantickitti.py 4
```

Notes:

- Use `bash`, not `sh`, for the dist scripts (they use bash-only `${@:4}`; Ubuntu's `sh`
  is dash and fails with "Bad substitution").
- Append `--resume` to continue an interrupted run from the latest checkpoint.
- Prefix with `PORT=29511 ...` if the default port is busy.
- Outputs land in `work_dirs/<config-name>/`; checkpoints every 5 epochs, val PQ table
  every epoch. Reference cost: DSO at batch 2 is ~0.9 s/iter and ~19.6 GB, roughly 40 h
  for 36 epochs on one modern card. DSO frames are large (~416k points), so 24 GB cards
  can run out of memory at batch 2 — use the 2xb1 config there.

## Testing

`epoch_36.pth` is the final checkpoint. Check the training log's per-epoch val PQ and
substitute the best epoch (saved at 5, 10, ..., 35, 36) if it is not the last one.

Without a `--cfg-options` override, `test.py` evaluates the **val** split:

```bash
python test.py configs/p3former/p3former_2xb1_3x_dso.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth
python test.py configs/p3former/p3former_2xb1_3x_semantickitti.py work_dirs/p3former_2xb1_3x_semantickitti/epoch_36.pth
```

Point it at another split by overriding the annotation file:

```bash
python test.py configs/p3former/p3former_2xb1_3x_dso.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl
python test.py configs/p3former/p3former_2xb1_3x_dso.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test_cetran.pkl
```

Multi-GPU evaluation works the same way, though evaluation needs only ~1–2 GB:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_test.sh configs/p3former/p3former_4xb1_3x_semantickitti.py work_dirs/p3former_4xb1_3x_semantickitti/epoch_36.pth 4
```

Note: SemanticKITTI "testing" means the val split — the official test split is unlabeled
and submission-only, via the `_submit` config.

## OOD testing

OOD scores are computed post hoc from the auxiliary semantic branch of a normally trained
model — no retraining. The `*_ood.py` configs add `ood_cfg` to the head and an
`_OODPointMetric` evaluator (point-level AUROC / AP / FPR@95, higher score = more OOD) next to
the usual PQ metric. Scores: MSP, MaxLogit, ODIN (T=1000), Energy (T=1), Entropy, each also in
a Group and a Group-Normalised (GN) variant over the six class groups of
`ood_cfg.class_groups`. The log ends with one `ood/<score>_AUROC|_AP|_FPR95` line per score
and the ID / OOD point counts.

- **SemanticKITTI:** OOD = other-structure (raw 52) + other-object (raw 99); val split.
- **DSO:** OOD = Stop (raw 17) + Others (raw 28); held-out test split or test + Cetran.

```bash
# SemanticKITTI, val split (use p3former_8xb2_3x_semantickitti_ood.py for the official checkpoint)
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_semantickitti_ood.py work_dirs/p3former_2xb1_3x_semantickitti/epoch_36.pth

# DSO, held-out test split
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_dso_ood.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir work_dirs/p3former_2xb1_3x_dso_ood/test --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl

# DSO, test + Cetran
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_2xb1_3x_dso_ood.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir work_dirs/p3former_2xb1_3x_dso_ood/test_cetran --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test_cetran.pkl
```

Notes:

- One eval needs ~17 GB (SemanticKITTI) to ~25 GB (DSO) of GPU memory and fails inside
  spconv (`cuda execution failed with error 2`) when less is free — run one eval per GPU.
- The evaluator keeps every score of every point in RAM; DSO test + Cetran (1.26B points)
  needs on the order of 100 GB.
- Smoke test: slice a few frames off an annotated split and point `ann_file` at them, e.g.
  `python -c "import mmengine as m; d=m.load('data/dso/dso_infos_test.pkl'); d['data_list']=d['data_list'][:20]; m.dump(d, 'data/dso/dso_infos_ood_mini.pkl')"`
  then add `--cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_ood_mini.pkl`.
  Check the OOD point count in the log is non-zero (the converter's `dso_infos_mini.pkl` is
  cut from a train sequence and is not meant for this).

### Class-hierarchy ablation (DSO)

Extra hierarchies for the Group / GN scores are scored in the same inference pass
(`class_groups_variants` in `ood_cfg`). `tools/make_dso_hierarchy_variants.py` writes one
config per batch of 12 hierarchies — every set partition of the six base groups plus a split
hierarchy, 203 in total, on the Cetran-only split — and `tools/summarize_hierarchy_ablation.py`
ranks them against the flat scores:

```bash
python tools/make_dso_hierarchy_variants.py   # -> configs/p3former/hier/*_b{1..17}.py
for i in $(seq 1 17); do
  CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/hier/p3former_2xb1_3x_dso_ood_hier_b$i.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir work_dirs/p3former_2xb1_3x_dso_ood_hier/b$i
done
python tools/summarize_hierarchy_ablation.py --family group [--exclude odin] [--plot top4_group.png] work_dirs/p3former_2xb1_3x_dso_ood_hier/b*/*/*.log
```

Each batch takes ~10–12 min and ~140 GB RAM. `--family` is `group` or `gn`; MaxLogit is
always excluded. The summary prints, per hierarchy, Δ = hierarchy − flat for each score and
`improvement` = mean ΔAUROC + mean ΔAP − mean ΔFPR@95 (the sort key). To confirm a winner on
test / test + Cetran, copy its groups into `class_groups` of `p3former_2xb1_3x_dso_ood.py`
and rerun the commands above.
