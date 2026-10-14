# DOCs

Working log for P3Former_OOD (OOD panoptic segmentation baseline).

## 2026-08-05 — Environment setup

Conda env `p3former` (base conda's libmamba solver is broken, so use `--solver=classic`):

```bash
conda create -n p3former python=3.8 -y --solver=classic
conda activate p3former
```

Install the pinned stack (the README wraps the PyTorch URL across two lines — it must be one line):

```bash
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



## 2026-08-05 — Data setup

SemanticKITTI lives on the SSD; the repo expects `data/semantickitti`, so symlink it:

```bash
mkdir -p data
ln -s /mnt/ssd/khoadv/projects/OOD_PanSeg_3D/data/SemanticKITTI data/semantickitti
```

Info pkls (`semantickitti_infos_{train,val,trainval,test}.pkl`) are generated with:

```bash
python tools/create_data.py semantickitti --root-path data/semantickitti --out-dir data/semantickitti --extra-tag semantickitti
```



## 2026-08-05 — Validate pretrained checkpoint on SemanticKITTI val

Checkpoint: `checkpoint/semantickitti_val_62.6.pth` (official release, reference PQ 62.6 on val).

```bash
CUDA_VISIBLE_DEVICES=1 bash dist_test.sh configs/p3former/p3former_8xb2_3x_semantickitti.py checkpoint/semantickitti_val_62.6.pth 1
```

Notes:

- Use `bash`, not `sh` (script uses bash-only `${@:4}`; Ubuntu's `sh` is dash → "Bad substitution").
- `CUDA_VISIBLE_DEVICES` picks the GPU; `PORT=...` overrides the default 29500 if the port is busy.
- Logs and results go to `work_dirs/p3former_8xb2_3x_semantickitti/`.

Results (4071 val scans, ~6.5 min on one RTX 6000 Ada, ~1.2 GB GPU memory):


| PQ    | PQ†   | RQ    | SQ    | mIoU  | PQ_things | PQ_stuff |
| ----- | ----- | ----- | ----- | ----- | --------- | -------- |
| 62.63 | 66.25 | 72.42 | 76.17 | 66.77 | 69.36     | 57.74    |


Matches the reference PQ 62.6 for this checkpoint.

## 2026-08-06 — Training commands

DSO (this machine, single GPU — ~0.9 s/iter, ~18.8 GB at batch 2, ≈40 h for 36 epochs):

```bash
CUDA_VISIBLE_DEVICES=1 python train.py configs/p3former/p3former_1xb2_3x_dso.py
```

DSO (4x A5000 24 GB server, batch 1 per GPU, effective batch 4):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh configs/p3former/p3former_4xb1_3x_dso.py 4
```

SemanticKITTI (single GPU, batch 2):

```bash
CUDA_VISIBLE_DEVICES=1 python train.py configs/p3former/p3former_1xb2_3x_semantickitti.py
```

SemanticKITTI (4x A5000, batch 1 per GPU):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dist_train.sh configs/p3former/p3former_4xb1_3x_semantickitti.py 4
```

Notes:

- Append `--resume` to continue an interrupted run from the latest checkpoint.
- `PORT=29511 ...` in front of dist_train.sh if the default port is busy.
- Outputs land in `work_dirs/<config-name>/`; checkpoints every 5 epochs, val PQ table every epoch.
- Run status: DSO 1xb2 ✅ done · SemanticKITTI 4xb1 ✅ done · SemanticKITTI 1xb2 🔄 running ·
DSO 4xb1 ❌ out-of-memory on 24 GB A5000s (DSO frames are ~416k pts; no checkpoint produced).



## 2026-08-11 — Testing commands

`epoch_36.pth` is the final checkpoint; check the training log's per-epoch val PQ and substitute
the best epoch's checkpoint (saved at 5, 10, ..., 35, 36) if it isn't the last one.

DSO 1xb2 — val split (Chinatown Route 1 Day, 401 frames; this is what test.py evaluates by default):

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_1xb2_3x_dso.py work_dirs/p3former_1xb2_3x_dso/epoch_36.pth
```

DSO 1xb2 — held-out test split (One-North Route 2 Day + Ubin Route 1 + Ubin Route 3, 2625 frames;
the two rural Ubin routes are a deliberate urban→rural domain shift):

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_1xb2_3x_dso.py work_dirs/p3former_1xb2_3x_dso/epoch_36.pth --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl
```

SemanticKITTI 1xb2 — val split (4071 scans; run after training finishes):

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_1xb2_3x_semantickitti.py work_dirs/p3former_1xb2_3x_semantickitti/epoch_36.pth
```

SemanticKITTI 4xb1 — val split:

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_4xb1_3x_semantickitti.py work_dirs/p3former_4xb1_3x_semantickitti/epoch_36.pth
```

Notes:

- SemanticKITTI "testing" means the val split — the official test split is unlabeled
(submission-only, via the `_submit` config).
- The two DSO evals contend with whatever is training on GPU 1; run them after the
SemanticKITTI 1xb2 run finishes (or on GPU 0 when it is free — eval is light).



## 2026-08-12 — Point-level OOD baselines (MSP / MaxLogit / ODIN / Energy)

Post-hoc OOD scoring from the aux semantic branch; OOD classes = raw 52
(other-structure) + 99 (other-object); raw 0/1 (unlabeled/outlier) excluded. Spec:
`docs/superpowers/specs/2026-08-12-ood-baselines-design.md`, branch `ood-baselines`.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_8xb2_3x_semantickitti_ood.py checkpoint/semantickitti_val_62.6.pth
```

Results (val, 4071 scans, official checkpoint; PQ table unchanged at 62.63;
476.8M ID / 9.4M OOD points = 1.94% OOD):


| method         | AUROC | AP    | FPR@95 |
| -------------- | ----- | ----- | ------ |
| MSP            | 87.48 | 16.29 | 43.45  |
| MaxLogit       | 91.28 | 40.28 | 39.98  |
| ODIN           | 90.87 | 32.42 | 40.70  |
| Energy         | 91.59 | 43.71 | 39.98  |
| Entropy        | 88.57 | 24.70 | 42.91  |
| Group MSP      | 90.11 | 17.17 | 36.89  |
| Group MaxLogit | 91.28 | 40.28 | 39.98  |
| Group ODIN     | 33.45 | 1.30  | 93.68  |
| Group Energy   | 91.51 | 41.97 | 39.96  |
| Group Entropy  | 90.55 | 24.32 | 36.50  |
| GN MSP         | 68.69 | 12.01 | 89.41  |
| GN MaxLogit    | 89.75 | 36.29 | 45.16  |
| GN ODIN        | 89.02 | 13.17 | 40.55  |
| GN Energy      | 89.93 | 37.70 | 45.17  |
| GN Entropy     | 68.68 | 10.88 | 89.41  |


ODIN = temperature-scaled MSP (T=1000, ε=0 per docs/others/baselines/OOD_Baseline.pdf);
Energy uses T=1. Higher score = more OOD everywhere. Smoke test: add
`--cfg-options test_dataloader.dataset.dataset.ann_file=semantickitti_infos_mini.pkl`
(20-scan mini pkl generated from the val infos). Unit tests: `tests/test_ood_scores.py`,
`tests/test_ood_eval.py`, `tests/test_ood_metric.py`.

## 2026-08-14 — DSO class-set revision (24 classes)

The DSO class definitions changed (new names; e.g. road→Paved Road, fence→Other Barrier,
net-fence→Perimeter Barrier). Training now covers **24 classes** instead of 16:

- **Things (train 0–8):** car, bicycle, motorcycle, truck, bus, person, rider,
**traffic-sign** (raw 19), **traffic-cone** (raw 20) — signs/cones carry real instance ids
(~17.6 and ~4.2 per frame) and are now things; the loader keeps their instance bits
(`DSO_THING_RAW_IDS` gained 19, 20).
- **Stuff (train 9–23):** paved-road, unpaved-road, sidewalk, building, window,
perimeter-barrier, other-barrier, overhead-bridge, gate, pole-like-object, drain, terrain,
trunks, vegetation, obscurant (ascending raw id).
- **Ignored (→ 24):** Noise (0), **Stop (17)** and **Others (28)** — reserved as future OOD
classes — Sky (29) and Water Body (30) — 2D-only — Unlabelled (255).

Head is now 25-way (`num_classes=25`, `cls_channels=(256,256,25)`); pkls/splits unchanged.
**Old 16-class checkpoints are incompatible — DSO must be retrained** (same commands as
2026-08-06; smoke-verified: 19.6 GB at batch 2, so the 4xb1 A5000 OOM situation is unchanged).
Committed on `dso-dataset`, merged into `ood-baselines`.

Caveat for eval: many traffic-cone instances are below the `min_num_points=50` PQ cutoff
(~40 pts/instance on average), so cone PQ reflects only the larger instances — standard
SemanticKITTI convention, left as is.

Update: the 24-class 1xb2 run OOM'd mid-epoch-3 in the thing-mask loss (~27 GB allocation
spike on instance-dense frames — signs/cones roughly double the thing masks; the run also
shared GPU 0 with a ~15 GB job). Use the 2-GPU batch-1 config instead (same effective
batch 2, ~half the per-GPU activations):

```bash
CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_dso.py 2
```



## 2026-08-15 — 2xb1 SemanticKITTI config; DSO 24-class test commands

SemanticKITTI on both of this machine's GPUs (batch 1 per GPU, effective batch 2 — same
recipe as 1xb2, ~half the per-GPU memory and wall-clock):

```bash
CUDA_VISIBLE_DEVICES=0,1 bash dist_train.sh configs/p3former/p3former_2xb1_3x_semantickitti.py 2
```

DSO 24-class run — held-out test split (2,625 frames; One-North Route 2 Day + the two rural
Ubin routes). Without the `--cfg-options` override, test.py evaluates the val split:

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_2xb1_3x_dso.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl
```



## 2026-08-18 — Cetran test sets

The three Cetran AV-test-centre sequences (401 + 202 + 377 = 980 frames, ~300k pts/frame,
fully labeled; both future-OOD classes Stop/Others present in all three) are now optional
test sets. Manual review of the per-frame PNGs confirmed the three runs cover
non-overlapping content (the 01-28 "AM-clean" re-export shares frame stems with "AM" but
different scenes). `create_data.py dso` writes them only when all three dirs exist and they
never enter train/val:

- `dso_infos_cetran.pkl` — Cetran only (980 frames)
- `dso_infos_test_cetran.pkl` — held-out test + Cetran (3,605 frames)

Evaluate the 24-class model on them with the usual override, e.g.:

```bash
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_2xb1_3x_dso.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test_cetran.pkl
```



## 2026-08-19 — Point-level OOD baselines on DSO (MSP / MaxLogit / ODIN / Energy)

Same post-hoc protocol as the 2026-08-12 SemanticKITTI entry, ported to the 24-class DSO
model via `configs/p3former/p3former_2xb1_3x_dso_ood.py`: scores from the first 24 channels
of the aux semantic branch; OOD = raw 17 (Stop) + 28 (Others); raw 0/29/30/255
(Noise/Sky/Water Body/Unlabelled) excluded. Checkpoint:
`work_dirs/p3former_2xb1_3x_dso/epoch_36.pth`.

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_dso_ood.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir work_dirs/p3former_2xb1_3x_dso_ood/test --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test.pkl
CUDA_VISIBLE_DEVICES=1 python test.py configs/p3former/p3former_2xb1_3x_dso_ood.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir work_dirs/p3former_2xb1_3x_dso_ood/test_cetran --cfg-options test_dataloader.dataset.dataset.ann_file=dso_infos_test_cetran.pkl
```

Held-out test split (2,625 frames; PQ 46.50, mIoU 48.66; 1.008B ID / 15.9M OOD points
= 1.55% OOD):


| method         | AUROC | AP    | FPR@95 |
| -------------- | ----- | ----- | ------ |
| MSP            | 86.26 | 13.76 | 51.15  |
| MaxLogit       | 92.08 | 34.24 | 42.70  |
| ODIN           | 86.53 | 19.51 | 69.63  |
| Energy         | 92.36 | 35.47 | 42.55  |
| Entropy        | 87.84 | 20.23 | 50.40  |
| Group MSP      | 89.33 | 14.50 | 45.99  |
| Group MaxLogit | 92.08 | 34.24 | 42.70  |
| Group ODIN     | 22.40 | 0.92  | 96.21  |
| Group Energy   | 92.35 | 35.01 | 42.56  |
| Group Entropy  | 89.67 | 18.91 | 46.41  |
| GN MSP         | 92.35 | 18.08 | 34.11  |
| GN MaxLogit    | 93.54 | 35.99 | 33.18  |
| GN ODIN        | 65.68 | 6.10  | 96.09  |
| GN Energy      | 93.79 | 36.53 | 32.81  |
| GN Entropy     | 92.11 | 13.70 | 34.11  |


Test + Cetran (3,605 frames; PQ 46.19, mIoU 47.94; 1.261B ID / 23.6M OOD points
= 1.84% OOD):


| method         | AUROC | AP    | FPR@95 |
| -------------- | ----- | ----- | ------ |
| MSP            | 87.76 | 18.06 | 45.76  |
| MaxLogit       | 92.67 | 35.90 | 38.12  |
| ODIN           | 89.36 | 27.19 | 55.64  |
| Energy         | 92.88 | 35.55 | 37.94  |
| Entropy        | 89.38 | 26.12 | 44.90  |
| Group MSP      | 89.97 | 16.30 | 40.66  |
| Group MaxLogit | 92.67 | 35.90 | 38.12  |
| Group ODIN     | 24.43 | 1.11  | 96.08  |
| Group Energy   | 92.88 | 36.03 | 37.95  |
| Group Entropy  | 90.45 | 21.85 | 40.78  |
| GN MSP         | 92.07 | 19.81 | 39.89  |
| GN MaxLogit    | 93.75 | 36.28 | 30.36  |
| GN ODIN        | 70.95 | 8.60  | 94.82  |
| GN Energy      | 93.94 | 36.31 | 29.96  |
| GN Entropy     | 91.75 | 14.49 | 39.89  |


Energy and MaxLogit are the strongest on both splits (as on SemanticKITTI); ODIN's
T=1000 flattening hurts noticeably more here than on SemanticKITTI. Smoke test: 5-frame
`dso_infos_mini.pkl` (frames of `dso_infos_test.pkl` verified to contain Stop/Others).

## 2026-08-21 — Entropy added as a fifth OOD score

`entropy` = softmax Shannon entropy `−Σ_c p_c log p_c` (natural log, higher = more OOD;
as in `trash/Done/eval_ood_from_logits.py`). Emitted as `ood_entropy` and included in
`_OODPointMetric`'s default keys — no config changes. The Entropy rows in the tables above
come from re-running the three evals on 2026-08-21 (the other rows reproduced exactly).
Note: one eval needs ~17 GB (SemanticKITTI) to ~25 GB (DSO) of GPU memory; it OOMs in
spconv (`cuda execution failed with error 2`) when less is free.

## 2026-08-21 — Group / Group-Normalised OOD scores (GroupPaper)

Hierarchy-aware variants of the five scores from `papers/RelatedPapers/GroupPaper.pdf`
(Eq. 2–8), ported from `trash/Done/eval_ood_from_logits.py` with our sign convention (`−max`):
Group sums softmax mass per semantic group (logits keep max / logsumexp within the group),
then max over groups; GN chance-corrects with `[P_g − K_g/C]_+` (probabilities) or `− log K_g`
(logits). Six groups (vehicle / human / ground / construction / nature / object) = the paper's
Table 2 for SemanticKITTI and the script's `dso24` preset for DSO, set via
`ood_cfg.class_groups` in both OOD configs; `_OODPointMetric` now evaluates every `ood_*` key
the model emits. The Group/GN rows in the three tables above come from the 2026-08-21 re-runs
(flat rows reproduced exactly). Reading: Group MSP/Entropy help on all splits (SemanticKITTI
MSP FPR@95 43.45 → 36.89); Group MaxLogit ≡ MaxLogit by construction; Group ODIN collapses
(at T=1000 the group sums are dominated by group size); GN hurts the probability scores on
SemanticKITTI but is the best family on DSO (GN Energy 93.8–93.9 AUROC, FPR@95 ≈ 30–33).
`_PanopticSegMetric` now stores only its two masks — it used to copy every `pred_pts_seg` key,
doubling the evaluator's RAM with 15 scores (the first Cetran re-run was OOM-killed at 251 GB).

## 2026-08-21 — OOD baselines for the 2xb1 SemanticKITTI model

`configs/p3former/p3former_2xb1_3x_semantickitti_ood.py` mirrors the 8xb2 OOD config on top
of `p3former_2xb1_3x_semantickitti.py` (our own 2-GPU batch-1 run, last epoch):

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_semantickitti_ood.py work_dirs/p3former_2xb1_3x_semantickitti/epoch_36.pth
```

Val (4071 scans; PQ 60.32, PQ† 62.99, mIoU 62.49 vs 62.63 / 66.25 / 66.77 for the official
checkpoint; same 476.8M ID / 9.4M OOD points):


| method         | AUROC | AP    | FPR@95 |
| -------------- | ----- | ----- | ------ |
| MSP            | 87.25 | 12.79 | 41.39  |
| MaxLogit       | 90.03 | 32.50 | 44.38  |
| ODIN           | 91.46 | 26.94 | 38.33  |
| Energy         | 90.20 | 33.26 | 44.47  |
| Entropy        | 88.33 | 17.40 | 40.99  |
| Group MSP      | 90.65 | 17.26 | 36.16  |
| Group MaxLogit | 90.03 | 32.50 | 44.38  |
| Group ODIN     | 27.45 | 1.20  | 94.44  |
| Group Energy   | 90.39 | 35.17 | 44.37  |
| Group Entropy  | 91.03 | 22.14 | 35.74  |
| GN MSP         | 71.81 | 12.70 | 89.49  |
| GN MaxLogit    | 87.88 | 28.51 | 52.50  |
| GN ODIN        | 88.47 | 12.27 | 44.20  |
| GN Energy      | 88.14 | 30.65 | 52.56  |
| GN Entropy     | 71.86 | 12.49 | 89.49  |


Same picture as the official checkpoint, ~1 point lower across the board; here ODIN is the
best flat AUROC and Group MSP/Entropy give the lowest FPR@95.

## 2026-08-25 — Testing different class hierarchies (DSO)

Extra hierarchies are scored in the same inference pass: `class_groups_variants=dict(name=[...])`
under `ood_cfg` emits `name_group_*` / `name_gn_*` keys. Ablation runs on the Cetran-only
split (980 frames, ~3% OOD).

1. **Define** — `tools/make_dso_hierarchy_variants.py` enumerates every set partition of the
  six base groups in `GROUPS` (202 partitions with 2–6 groups, checked against the Stirling
   numbers 31/90/65/15/1; named by block initials, e.g. `p_vh_gcno` = {vehicle+human} |
   {ground+construction+nature+object}, `p_v_h_g_c_n_o` = the current hierarchy) plus `sp`
   (every group halved) — 203 hierarchies. Edit `GROUPS` / `SPLIT` to change the base, then
   `python tools/make_dso_hierarchy_variants.py` → `configs/p3former/hier/*_b{1..17}.py`
   (12 hierarchies each, Cetran split, OOD metric only, flat + current scores included).
2. **Run** — one batch at a time (~140 GB RAM, ~10–12 min each):
  ```bash
   for i in $(seq 1 17); do
     CUDA_VISIBLE_DEVICES=1 python test.py \
       configs/p3former/hier/p3former_2xb1_3x_dso_ood_hier_b$i.py \
       work_dirs/p3former_2xb1_3x_dso/epoch_36.pth \
       --work-dir work_dirs/p3former_2xb1_3x_dso_ood_hier/b$i
   done
  ```
3. **Rank** — `python tools/summarize_hierarchy_ablation.py --family group|gn [--exclude odin,entropy] work_dirs/p3former_2xb1_3x_dso_ood_hier/b*/*/*.log`
  prints per hierarchy Δ = hierarchy − flat (dAUROC/dAP/dFPR@95) per baseline, their mean,
   and `improvement` = mean dAUROC + mean dAP − mean dFPR@95 (sort key). MaxLogit is always
   excluded (its group/GN formulation is wrong).
4. **Confirm** the winner on test / test+Cetran: copy its groups into `class_groups` of
  `p3former_2xb1_3x_dso_ood.py` and run the 2026-08-19 commands.



## 2026-09-10 — Why `p_v_hgcno` (vehicle vs rest) beats the current 6-group hierarchy

```bash
python tools/summarize_hierarchy_ablation.py --family group --exclude odin work_dirs/p3former_2xb1_3x_dso_ood_hier/b*/*/*.log
```

Full 203-hierarchy ablation on Cetran-only: best is `p_v_hgcno` = {vehicle} | {all other classes}, Group MSP **96.00 AUROC / 47.23 AP / 18.42 FPR@95** vs flat MSP 90.42/28.27/32.32; the current 6-group hierarchy lands *below* flat (improvement −6).

- Group-MSP calls a point ID when one group collects most of its softmax mass, so putting classes in the same group absorbs confusion between them.
- Principle: the best grouping is determined by **where the model is reliable on the test distribution**, not by the semantic ontology. A group boundary is useful only if the model separates ID points across OOD points. 

Caveat: selected on Cetran-only among 203 candidates — confirm on test / test+Cetran
(step 4 above) before reporting.

## 2026-09-10 — Recording per-point logits; ID/OOD score-distribution figure

`ood_cfg.save_logits=True` now attaches the per-point float16 semantic logits
(`pred_pts_seg['sem_logits']`), and `_OODLogitsDumpMetric` writes one npz per frame
(logits + ood/valid/mapped GT; refuses a non-empty out_dir) — any score or hierarchy is
then recomputable offline, no further GPU passes. `p3former_2xb1_3x_dso_ood_dump.py` =
Cetran split + dump (~20 GB) + `_OODPointMetric` incl. the `p_v_hgcno` variant as
cross-check (PQ dropped):

```bash
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_dso_ood_dump.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth --work-dir work_dirs/p3former_2xb1_3x_dso_ood_dump
python tools/plot_ood_score_distributions.py work_dirs/p3former_2xb1_3x_dso_ood_dump/logits
```

The figure (after trash/ID_OOD_scores.png): one column per hierarchy
(`--hierarchies`, default current + `p_v_hgcno`), one row per group with the ID vs OOD
densities of the points that group claims (largest group mass), then the combined score
row annotated with the recomputed AUROC/AP/FPR@95; x = `1 − max_g P_g` on a log scale
(monotone in Group-MSP, metrics unchanged; a linear axis collapses into a spike at full
confidence). Smoke-verified on the 5-frame mini: offline recomputation matches the logged
`group_msp` / `p_v_hgcno_group_msp` rows to ±0.02 (float16). Unit tests:
`tests/test_ood_logits_dump.py`.

## 2026-09-15 — Random two-group partitions of the 24 classes (offline sweep)

`tools/sweep_bipartitions.py` scores 500 random two-group partitions of the 24 classes
(seed 0) straight from the logit dumps, without running the model; vehicle-vs-rest
(`s0.1.2.3.4` = `p_v_hgcno`) is always included. A held-out test dump was added, so the
sweep runs on Cetran, test and test + Cetran. With log-spaced bins the flat and Group rows
reproduce the online metric to <= 0.05, with one exception; the GN family is not ranked
offline (see 2026-09-24).

Results (Group MSP as AUROC/AP/FPR@95, improvement = mean dAUROC + mean dAP - mean dFPR@95):

| split (flat MSP; flat Energy) | above flat | best split | its Group MSP | improvement |
| --- | --- | --- | --- | --- |
| Cetran (90.42/28.27/32.32; 93.14/40.33/26.35) | 61 | `s1.3.17` {bicycle, truck, gate} | 96.84/54.34/17.13 | +27.69 |
| test (86.26/13.75/51.15; 92.36/35.47/42.56) | 136 | `s16` {overhead-bridge} | 94.21/39.75/29.06 | +34.78 |
| test + Cetran (87.76/18.06/45.76; 92.88/35.54/37.95) | 99 | `s16` {overhead-bridge} | 94.04/40.45/28.52 | +27.42 |

The Cetran winners do not transfer: on test alone `s1.3.17` and vehicle-vs-rest fall below
flat (-0.27 and -10.88), and the Cetran~test rank correlation of the improvement is +0.61.
32 splits beat flat on both Cetran and test. The best split isolates the classes the OOD
points are confused with, and those depend on the scenes (truck / gate on Cetran,
overhead-bridge on the test routes); the gain is in the MSP / Entropy scores, while Group
Energy moves by < 1 point. These are selections on the evaluation data: confirm a split
online on data it was not picked on before reporting it.

Full record (commands, backends, bins, top 5, single classes, most robust splits): [docs/experiments/2026-09-15-bipartition-sweep.md](docs/experiments/2026-09-15-bipartition-sweep.md)

## 2026-09-24 — Offline GN rankings withdrawn (Codex review)

Codex finding on the sweep: its bins are refined only towards the range ends, but GN MSP /
GN Entropy pile up at an interior value, so their offline metrics are invalid (test
vehicle-vs-rest GN MSP FPR@95 98.38 offline vs 76.99 online) — yet the tool still ranked
them and the 2026-09-15 entry concluded "GN never beats flat". Action: that GN conclusion is
withdrawn; `sweep_bipartitions.py` no longer prints a GN ranking and
`summarize_hierarchy_ablation.py --family gn` refuses sweep logs (marker line
`# offline bipartition sweep`). The GN rows stay in the log (GN Energy is exact); rank the
GN family from test.py logs, or add interior-adaptive bins first.

## 2026-09-30 — Why some two-group splits beat flat: divided mass

For a split A | B, Group MSP is m − 1, where m = min(P_A, P_B) is the *divided mass*; a point
is divided at δ when m ≥ δ, so the OOD / ID divided shares are Group MSP's TPR / FPR at the
threshold δ − 1. `tools/divided_mass.py` histograms m (and flat MSP's u = 1 − max p) from the
logit dumps for the 24 single-class splits, now all scored with `sweep_bipartitions.py
--subsets singletons`, and the 500 sweep splits, then joins the sweep's metric deltas; all
21 checks pass. Spec and plan: `docs/superpowers/{specs,plans}/2026-09-*-divided-mass-resemblance*`.

- **Robust splits** (improvement > 0 on all three sets): there are now **33**, because
  {building} joined {overhead-bridge} and {gate}; 29 of them contain one of those three.
- **Mechanism.** A split beats flat when it keeps far more of flat MSP's uncertain OOD points
  than of its uncertain ID points. At δ = 0.05 every single-class winner has a selectivity of
  1.8–31 and a divided precision of 10–78 %, against flat's 6–11 %. A clean ratio is
  necessary, not sufficient: FPR@95 equals the ID divided share at δ95, the depth at which
  95 % of the OOD points are divided ({bicycle} on Test: δ95 = 9.3e-10, improvement −40.1).
- **Divided precision is the best single predictor**: ρ 0.87–0.95 with ΔAP and 0.72–0.74 with
  the improvement at δ = 0.05, 0.79–0.84 at δ = 0.3.
- **Cetran vs Test: the OOD population shifts.** Test's OOD points lean far less toward
  truck, gate and bicycle (truck 15.6 → 6.4 %, gate 11.4 → 0.53 %), so truck and bicycle do
  not transfer and gate weakens (+17.5 → +6.6); overhead-bridge's share grows (11.1 → 18.2 %)
  and it becomes Test's best split.
- **Float32 ties.** On 21 / 50 / 43 splits the implemented Group MSP FPR@95 depends on float32
  rounding, and online also on the bins; the `exact_*` columns hold float32-free values.

The splits are still selected on the evaluation data, so any split must be confirmed online
before it is reported.

Full record (commands, checks, float32 analysis, Tables A–C, figures, reading): [docs/experiments/2026-09-30-divided-mass.md](docs/experiments/2026-09-30-divided-mass.md)

## 2026-09-30 — Do the best splits group the classes the OOD points resemble? (feature space)

Which ID classes do the OOD points look like in `pe_features`, the 256-d input of the
semantic classifier, and in its "appearance" part without the positional embedding? And
does the best split put those classes on one side, as the 09-15 reading suggested?
`tools/extract_point_features.py` samples the features (Cetran 1.25 M, Test 3.0 M samples);
`tools/ood_class_resemblance.py` runs a cosine kNN (k = 10) against a class-balanced ID
bank, where r_OOD(c) is class c's share of the OOD points' neighbours (4.17 % = no
preference). Re-run on 2026-10-04 after a sampling fix (`84521b0`), with seeds 1 and 2 as a
noise check; no conclusion changed.

- **The OOD points resemble structures**: building, perimeter-barrier and gate on Test (26 /
  16 / 9 %); gate, perimeter-barrier and person on Cetran against the full bank.
- **Class level: resemblance matches the winning splits.** ρ(r_OOD, improvement) is 0.76 /
  0.58 / 0.61 on Test / Test + Cetran / Cetran (full bank), 0.92 / 0.79 / 0.69 in the
  appearance space. Overhead-bridge is the exception: the best split on Test, yet r_OOD only
  1.4 %, because the positional part of `pe_features` hides the resemblance from the kNN.
- **Split level: the literal hypothesis gets mixed support.** Grouping the most-resembled
  classes on the small side is about as good as separating them on Test; lumping them with
  everything else hurts (median improvement −18 to −25).
- **Refined:** the best split cuts *through* the OOD points' neighbourhoods. The "cut
  through" log ratio predicts the improvement on every set (ρ 0.48–0.70), less well than
  the divided precision of the entry above (0.72–0.84).

Caveats: correlational only, over splits selected on the evaluation data; seed noise in ρ
reaches 0.12 at the class level and 0.03 at the split level.

Full record (method, commands, Tables A–D, seed and k = 50 checks, figures, t-SNE): [docs/experiments/2026-09-30-ood-resemblance.md](docs/experiments/2026-09-30-ood-resemblance.md)
