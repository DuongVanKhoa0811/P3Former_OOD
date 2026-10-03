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

`tools/sweep_bipartitions.py` scores random 2-group partitions of the 24 classes straight
from a logits dump, without running the model: the smaller group's size is drawn uniformly
from 1..12, then its classes at random; the vehicle-vs-rest split (`s0.1.2.3.4` =
`p_v_hgcno`) is always included as a cross-check. Names are the train ids of the smaller
group (`partitions.tsv` lists the class names). Same score formulas as `ood_scores.py`,
same metric formulas as `_OODPointMetric`; `bipartitions.log` is read by
`summarize_hierarchy_ablation.py`. Several dump directories are evaluated as one split.

- **Splits.** `dso_infos_test_cetran.pkl` is exactly test + Cetran (same frames, same order),
  so only the held-out test split needed a new dump:
  `p3former_2xb1_3x_dso_ood_dump_test.py` (2,625 frames, 55 GB, written to
  `/mnt/sandisk/khoadv/...` and symlinked as `work_dirs/p3former_2xb1_3x_dso_ood_dump/logits_test`;
  ~25 min, ~103 GB RAM). Passing `logits_test` + `logits` gives test + Cetran, `logits_test`
  alone the test split.
- **Backend.** `--backend numpy` (default) uses CPU workers (Cetran: ~1 h with 8 workers);
  `--backend torch --device cuda:N` does the same two passes on a GPU: Cetran 13 min, test
  54 min, test + Cetran 67 min. TF32 matmuls are switched off (they shift the group masses by
  ~4e-4 and FPR@95 by up to 57 points) and CUDA `bincount` is replaced by a sort-based count
  (~340 ms vs ~1 ms per block in torch 1.10).
- **Bins.** 2^16 bins, log-spaced towards both ends of every score range (`--bin-eps`;
  `--bin-eps 0` = the equal-width bins of the online metric). Equal-width bins at 2^16 cannot
  separate very confident OOD points from the ID mass: on the test split more than 5% of the
  OOD points sit within 7.6e-6 of full confidence and FPR@95 read 100% for 43/500 splits.
  With log-spaced bins the flat and Group rows reproduce the online metric to <= 0.05 on all
  three splits (vehicle-vs-rest Group MSP on Cetran: 96.00/47.23/18.42 offline = online). One
  exception, Group MSP FPR@95 of vehicle-vs-rest on test (67.10 offline, 68.02 online), is
  the online value's own bin limit: for two groups Group Entropy is a monotone function of
  Group MSP, so both must give the same FPR@95 — 67.07 / 67.10 offline, 67.12 / 68.02 online. GN
  MSP / GN Entropy stay approximate — they pile up at an interior value, -(1 - prior) — e.g.
  test vehicle-vs-rest GN MSP FPR@95 98.38 offline vs 76.99 online; GN Energy is exact.

```bash
# held-out test dump (once), then the sweeps; long jobs: start them with nohup setsid ... &
CUDA_VISIBLE_DEVICES=0 python test.py configs/p3former/p3former_2xb1_3x_dso_ood_dump_test.py work_dirs/p3former_2xb1_3x_dso/epoch_36.pth
W=work_dirs/p3former_2xb1_3x_dso_ood_dump
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits                                                        # Cetran      -> $W/bipartitions
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits_test --out-dir $W/bipartitions_test                    # test
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits_test $W/logits --out-dir $W/bipartitions_test_cetran   # test + Cetran
python tools/summarize_hierarchy_ablation.py --family group --exclude odin --plot $W/bipartitions_test_cetran/top4_group.png $W/bipartitions_test_cetran/bipartitions.log
```

Results (500 partitions, seed 0; family group, ODIN excluded — the GN family is not ranked
offline, see 2026-09-24; Group MSP as AUROC/AP/FPR@95, improvement = mean dAUROC + mean dAP
- mean dFPR@95):

| split (flat MSP; flat Energy) | above flat | best split | its Group MSP | improvement |
| --- | --- | --- | --- | --- |
| Cetran (90.42/28.27/32.32; 93.14/40.33/26.35) | 61 | `s1.3.17` {bicycle, truck, gate} | 96.84/54.34/17.13 | +27.69 |
| test (86.26/13.75/51.15; 92.36/35.47/42.56) | 136 | `s16` {overhead-bridge} | 94.21/39.75/29.06 | +34.78 |
| test + Cetran (87.76/18.06/45.76; 92.88/35.54/37.95) | 99 | `s16` {overhead-bridge} | 94.04/40.45/28.52 | +27.42 |

- **The Cetran winners do not transfer.** On test + Cetran `s1.3.17` drops to #45 (+7.38,
  91.70/26.56/41.73) and vehicle-vs-rest (`p_v_hgcno`, #12 on Cetran with +21.37) to #142
  (-3.17, 90.07/25.30/54.59); on test alone they fall below flat (-0.27 and -10.88; online
  test `p_v_hgcno` Group MSP 87.21/17.23/68.02 vs flat MSP 86.26/13.76/51.15). Rank
  correlation of the improvement: Cetran~test +0.61, test~test+Cetran +0.97 (test has 4x the
  points). 32 splits beat flat on both Cetran and test.
- **test + Cetran top 5:** `s16` +27.42, `s2.16` {motorcycle, overhead-bridge} +26.24
  (94.16/37.98/27.90), `s4.16` {bus, overhead-bridge} +23.50, `s2.3.4.5.6.16` +21.01,
  `s4.5.6.16` +19.39. Overhead-bridge alone is the best split on test and test + Cetran but
  only #40 on Cetran (+7.03); single classes above flat on test + Cetran: overhead-bridge
  +27.4, gate +10.1, truck +4.0, perimeter-barrier +1.2 (Cetran: truck +25.2, gate +17.5,
  overhead-bridge +7.0, bicycle +5.8). Building, drain, unpaved-road, bus and overhead-bridge
  are over-represented in the top-20 smaller groups; paved-road, sidewalk, terrain, trunks
  and vegetation never appear there on any split.
- **Most robust splits** (largest worst-case improvement over Cetran and test):
  `s0.1.3.5.10.16` {car, bicycle, truck, person, unpaved-road, overhead-bridge} +17.87 / +19.04
  (test + Cetran +18.66, 93.52/33.19/33.63) and `s2.3.4.5.6.16` {motorcycle, truck, bus, person,
  rider, overhead-bridge} +16.94 / +22.23 (test + Cetran +21.01, 93.84/34.65/31.91).
- **Reading.** A two-group Group-MSP flags a point when its mass is split between the two
  groups, so the best split isolates the classes the OOD points are confused with and
  merges every within-ID confusion. Which classes those are depends on the scenes — truck /
  gate on Cetran, overhead-bridge (and building / drain / unpaved-road) on the test routes —
  so a split picked on one split is tuned to its OOD objects. The gain is in the MSP /
  Entropy scores; Group Energy moves by < 1 point everywhere.

Caveat: these are selections among 500 random splits on the evaluation data itself — report
a split only after confirming it online (`class_groups_variants`) on data it was not picked on.

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

For a split A | B, Group MSP is m − 1, where m = min(P_A, P_B) is the *divided mass*. A point is *divided* at δ when m ≥ δ; at δ = 0.05 that means 0.05 ≤ P_A ≤ 0.95. So the OOD / ID divided shares at δ are Group MSP's TPR / FPR at the threshold δ − 1. Flat MSP has the same reading through u = 1 − max p, and u ≥ m for every point. `tools/divided_mass.py` histograms m and u from the logit dumps for two sets of splits, then joins the sweep's metric deltas:
- the 24 single-class splits. The 500-split sweep had missed {unpaved-road}, {sidewalk} and {building}, so all 24 are now also scored with `sweep_bipartitions.py --subsets singletons`;
- the 500 sweep splits.

The spec and plan are `docs/superpowers/{specs,plans}/2026-09-*-divided-mass-resemblance*`.

```bash
W=work_dirs/p3former_2xb1_3x_dso_ood_dump
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits --subsets singletons --out-dir $W/singletons
python tools/sweep_bipartitions.py --backend torch --device cuda:1 $W/logits_test --subsets singletons --out-dir $W/singletons_test
python tools/sweep_bipartitions.py --backend torch --device cuda:0 $W/logits_test $W/logits --subsets singletons --out-dir $W/singletons_test_cetran
python tools/divided_mass.py --backend torch --device cuda:0        # -> $W/divided_mass/ (~7 min)
python tools/plot_divided_mass.py && python tools/plot_divided_mass.py --threshold 0.001
```

**Checks** (`summary.md`): all 21 PASS.
- Flat MSP from the u histograms matches the sweep within 0.004 AUROC, 0.003 AP and 0.15 FPR@95.
- Group MSP matches the sweep within 0.006 AUROC, 0.052 AP and 0.27 FPR@95 on the 482 / 453 / 460 well-conditioned splits (δ95 ≥ 1e-5).
- The 21 single-class splits scored by both sweeps agree exactly.

The first run caught two defects before these checks passed: the u histograms were clipped at 0.5, and the float32 effect below.

**Float32 ties in the implemented Group MSP.** `test.py` and the sweep rank points by the float32 score −max(P_A, P_B), which cannot rank points with m < 2⁻²⁴ ≈ 6e-8. On 21 / 50 / 43 splits, catching 95 % of the OOD points needs a threshold below m ≈ 1e-5. For those splits the implemented FPR@95 depends on float32 rounding. For example, {bicycle} on Test reads 89.20 in the sweep, 100 under a float32 emulation, and **80.43 exactly**. The `exact_*` columns hold the float32-free values.

**Table A — single-class splits at δ = 0.05.** The rows are every class that beats flat on some set, plus {vegetation}.
- *div*: the % of points divided.
- *prec*: the share of the divided points that are OOD. Flat MSP's share of *uncertain* points (u ≥ δ) that are OOD is 11.0 / 6.2 / 7.3 % on Cetran / Test / Test + Cetran.
- *sel*: selectivity, the OOD retention over the ID retention. Retention is the fraction of flat MSP's uncertain points that stay divided.
- *δ95*: the depth at which 95 % of the OOD points are divided. The ID div there equals the exact FPR@95 (flat MSP: 32.4 / 51.2 / 45.9).
- *imp*: the improvement.

| set | split | OOD div % | ID div % | prec % | sel | δ95 | ID div % at δ95 | imp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cetran | truck | 15.6 | 0.285 | 62.4 | 13.4 | 5.3e-5 | 18.4 | +25.2 |
| Cetran | gate | 11.4 | 0.099 | 77.8 | 28.2 | 8.7e-6 | 21.7 | +17.5 |
| Cetran | overhead-bridge | 11.1 | 0.098 | 77.4 | 27.6 | 8.0e-6 | 33.0 | +7.0 |
| Cetran | bicycle | 1.65 | 0.037 | 57.3 | 10.8 | 1.0e-6 | 28.5 | +5.8 |
| Cetran | building | 52.3 | 2.80 | 36.2 | 4.6 | 3.2e-4 | 32.1 | +4.3 |
| Cetran | bus | 1.37 | 0.015 | 73.6 | 22.5 | 4.0e-6 | 31.7 | +0.3 |
| Cetran | perimeter-barrier | 48.3 | 8.61 | 14.6 | 1.37 | 1.5e-4 | 26.1 | −11.7 |
| Cetran | vegetation | 33.0 | 7.41 | 11.9 | 1.09 | 1.2e-4 | 58.0 | −40.4 |
| Test | overhead-bridge | 18.2 | 0.142 | 66.9 | 30.8 | 3.6e-6 | 29.1 | +34.8 |
| Test | building | 51.4 | 2.56 | 24.0 | 4.8 | 1.5e-4 | 37.4 | +16.3 |
| Test | gate | 0.53 | 0.072 | 10.3 | 1.76 | 4.0e-7 | 41.5 | +6.6 |
| Test | perimeter-barrier | 24.6 | 1.52 | 20.4 | 3.9 | 3.3e-5 | 44.6 | +5.5 |
| Test | truck | 6.42 | 0.254 | 28.5 | 6.1 | 2.8e-6 | 57.9 | −2.9 |
| Test | bus | 0.81 | 0.096 | 11.8 | 2.04 | 2.9e-7 | 59.8 | −6.1 |
| Test | vegetation | 38.5 | 12.0 | 4.8 | 0.77 | 3.6e-4 | 70.5 | −28.6 |
| Test | bicycle | 0.48 | 0.067 | 10.1 | 1.72 | 9.3e-10 | 80.4 | −40.1 |
| Test + Cetran | overhead-bridge | 15.9 | 0.133 | 69.0 | 28.2 | 4.6e-6 | 28.5 | +27.4 |
| Test + Cetran | building | 51.7 | 2.61 | 27.0 | 4.7 | 2.0e-4 | 33.8 | +13.3 |
| Test + Cetran | gate | 4.07 | 0.078 | 49.5 | 12.4 | 7.1e-7 | 34.0 | +10.1 |
| Test + Cetran | truck | 9.42 | 0.260 | 40.4 | 8.6 | 4.6e-6 | 47.6 | +4.0 |
| Test + Cetran | perimeter-barrier | 32.3 | 2.94 | 17.1 | 2.6 | 4.8e-5 | 37.4 | +1.2 |
| Test + Cetran | bus | 1.00 | 0.079 | 19.0 | 2.96 | 4.8e-7 | 53.9 | −7.0 |
| Test + Cetran | vegetation | 36.7 | 11.0 | 5.8 | 0.79 | 2.6e-4 | 70.3 | −35.8 |
| Test + Cetran | bicycle | 0.86 | 0.061 | 20.8 | 3.3 | 2.7e-9 | 77.3 | −38.8 |

**Table B — Spearman ρ at δ = 0.05**, over the 24 single-class / all 503 splits. Precision and the log OOD/ID ratio rank the splits identically.

| set | prec ~ ΔAP | prec ~ ΔAUROC | prec ~ improvement | OOD div ~ ΔFPR@95 | ID div ~ ΔAP |
| --- | --- | --- | --- | --- | --- |
| Cetran | 0.63 / 0.87 | 0.43 / 0.77 | 0.45 / 0.72 | −0.43 / −0.28 | −0.37 / −0.41 |
| Test | 0.81 / 0.95 | 0.54 / 0.80 | 0.51 / 0.72 | −0.68 / −0.47 | −0.15 / −0.52 |
| Test + Cetran | 0.82 / 0.95 | 0.62 / 0.80 | 0.53 / 0.74 | −0.66 / −0.44 | −0.26 / −0.52 |

The largest |ρ| with improvement over δ:
- All splits: precision at δ = 0.3, with 0.79 / 0.80 / 0.84.
- Single-class splits: precision at δ = 0.3 on Cetran (0.71), and the OOD div % at δ = 1e-3 on Test (0.63) and on Test + Cetran (0.62).

**Table C — robust splits** (improvement > 0 on all three sets). There are now **33**, because {building} joined {overhead-bridge} and {gate}. The table shows the top 10 by worst-set improvement, with the divided shares on Test + Cetran at δ = 0.05; flat MSP's precision there is 7.3 %.

| split | classes (A) | imp. Cetran / Test / T+C | OOD div % | ID div % | prec % |
| --- | --- | --- | --- | --- | --- |
| `s0.1.3.5.10.16` | car, bicycle, truck, person, unpaved-road, overhead-bridge | +17.9 / +19.0 / +18.7 | 27.7 | 0.57 | 47.5 |
| `s2.3.4.5.6.16` | motorcycle, truck, bus, person, rider, overhead-bridge | +16.9 / +22.2 / +21.0 | 26.4 | 0.50 | 49.5 |
| `s0.2.5.16.17.18.19` | car, motorcycle, person, overhead-bridge, gate, pole-like-object, drain | +19.7 / +15.1 / +17.3 | 28.6 | 0.74 | 41.8 |
| `s2.3.10.12.17.18.19` | motorcycle, truck, unpaved-road, building, gate, pole-like-object, drain | +14.5 / +19.4 / +18.0 | 58.8 | 3.23 | 25.4 |
| `s10.12.17.18.19` | unpaved-road, building, gate, pole-like-object, drain | +12.4 / +20.1 / +17.9 | 56.5 | 2.99 | 26.1 |
| `s0.1.2.3.4.6.7.10.16.17.18` | car, bicycle, motorcycle, truck, bus, rider, traffic-sign, unpaved-road, overhead-bridge, gate, pole-like-object | +21.9 / +12.3 / +14.8 | 36.0 | 0.76 | 46.8 |
| `s3.16.18` | truck, overhead-bridge, pole-like-object | +14.9 / +12.2 / +12.7 | 28.5 | 0.54 | 49.7 |
| `s7.13.16.17.23` | traffic-sign, window, overhead-bridge, gate, obscurant | +11.8 / +12.0 / +12.2 | 25.8 | 0.75 | 39.1 |
| `s10.12.17.19` | unpaved-road, building, gate, drain | +10.8 / +21.9 / +18.3 | 55.2 | 2.83 | 26.7 |
| `s2.16` | motorcycle, overhead-bridge | +10.0 / +32.0 / +26.2 | 16.2 | 0.19 | 61.9 |

29 of the 33 contain overhead-bridge, gate or building. 29 contain at least one class that has **no ID point in Cetran**.

**Figures** (in `$W/divided_mass/`):
- `bubble_singletons_0.05` and `bubble_singletons_0.001`: ID vs OOD divided share of the 24 single-class splits, with flat MSP's star and its iso-ratio line;
- `bubble_robust_0.05`: the robust splits over the grey sweep, with the top 8 numbered and keyed;
- `rho_vs_threshold`: the ρ of each statistic against each metric delta, over δ.

**Reading.**
- **Mechanism.** Group MSP flags only divided points, so a split trades OOD uncertainty for ID uncertainty. It beats flat when it keeps much more of flat MSP's uncertain OOD points than of its uncertain ID points.
  - At δ = 0.05 every single-class winner does: selectivity 1.8–31 and precision 10–78 %, against flat's 6–11 %.
  - The strong winners (improvement > +10) reach a selectivity of 4.7–31 and a precision of 24–78 %.
  - {vegetation} is the clearest loser, with a selectivity of 0.77–1.09: it absorbs OOD uncertainty as fast as ID uncertainty.
- **A clean ratio is necessary, not sufficient.** Several losers also have one, such as {bicycle} and {bus} on Test (selectivity 1.7–2.0), but they divide under 1 % of the OOD points. FPR@95 is decided by the depth δ95, because the ID divided share there *is* the exact FPR@95.
  - {gate} and {bicycle} on Test look alike at δ = 0.05: OOD div 0.53 vs 0.48 %, precision 10.3 vs 10.1 %.
  - Their depths differ 400-fold: δ95 = 4.0e-7 vs 9.3e-10. The other 99.5 % of the OOD points are confidently non-bicycle.
  - So their ID div at δ95 is 41.5 vs 80.4 %, against flat's 51.2, and their improvements are +6.6 vs −40.1.
- **Which statistic predicts what.**
  - Divided precision (the absolute-count view) is the best single predictor over all splits. At δ = 0.05, ρ is 0.87–0.95 with ΔAP and 0.72–0.74 with improvement, rising to 0.79–0.84 at δ = 0.3 (`rho_vs_threshold`).
  - The ID divided % alone is a weak, negative predictor.
  - Over the single-class splits, the OOD divided share tracks ΔFPR@95, with ρ −0.43 / −0.68 / −0.66.
- **Robust splits come in two kinds** (Table C). Both reach a divided precision 3.5–8.5× flat's.
  - Splits led by overhead-bridge divide about 16–36 % of the OOD points at an ID cost of only 0.2–0.8 %, with a precision of 39–62 %.
  - Splits built on building, gate and drain divide 55–59 % of the OOD points at an ID cost of 2.8–3.2 %, with a precision of about 26 %.
- **Cetran vs Test.**
  - Cetran's winners (truck, gate, overhead-bridge, bicycle) are classes with no ID point in the Cetran sequences. ID points there almost never lean toward them (ID div ≤ 0.29 %), while 11–16 % of the OOD points lean toward truck, gate or overhead-bridge.
  - On Test the same classes exist as ID, and the leaders are overhead-bridge and building.
  - This is the poor transfer noted on 2026-09-15.

**Caveat.** Divided shares are ROC operating points of Group MSP, so their link to the metric deltas is partly by construction. The explanation rests on the comparison with flat MSP (selectivity, precision) and on δ95. The splits are still selected on the evaluation data, so any split must be confirmed online before it is reported.

## 2026-09-30 — Do the best splits group the classes the OOD points resemble? (feature space)

The question from 09-29: which ID classes do the OOD points look like in the features the classifier reads, and does the best split "put the classes that the OOD objects resemble on one side and everything else on the other" (the 09-15 reading)?

**Features.** `pe_features` is the 256-d per-point input of the semantic classifier (`sem_preds = pe_features @ sem_queriesᵀ`, bias-free). "Appearance" is `pe_features` minus the positional embedding added just before it. `tools/extract_point_features.py` captures both by wrapping `_P3FormerHead.init_inputs` on the model instance.
- Per frame it keeps up to 64 ID points per class and 512 OOD points, each weighted to its (frame, class) stratum.
- Cetran: 980 frames, 1,252,847 samples, 501,760 of them OOD. Test: 2,625 frames, 2,995,232 samples, 1,034,359 OOD.
- Checks: the captured features reproduce the classifier's logits (`feat @ W^T` in float64) within 0.0073 / 0.0077. At the sampled points, the run's logits, labels and OOD flags equal those of the logit dumps; the largest logit difference is 0.0.

**Measure** (`tools/ood_class_resemblance.py`): a cosine kNN (k = 10) against a class-balanced bank of ID samples. The bank holds up to 4,000 per class, drawn ∝ weight; up to 2,000 other ID points per class serve as queries.
- r_OOD(c) is the share of class c among the OOD points' neighbours; 4.17 % means no preference.
- r_ID(c) is the same share around the ID points of the other classes, and contrast = r_OOD / r_ID.
- Per split A | B, with A the smaller group:
  - R_A = Σ_{c∈A} r_OOD(c), and E_A = R_A over its no-preference value ("together": the neighbours fall in A);
  - *feature-divided*: the k neighbours hold classes of both sides ("cut through"), with its log OOD/ID ratio;
  - the *literal* test: the splits grouped by where the k most-resembled classes sit.
- **Cetran has no ID point of 10 classes**: bicycle, motorcycle, truck, rider, unpaved-road, window, overhead-bridge, gate, drain, obscurant. Against its own bank (the default) they cannot be measured and are left out; that covers truck and gate, its two best splits. `--reference test_cetran` measures every set's OOD points against the Test + Cetran bank instead.

```bash
W=work_dirs/p3former_2xb1_3x_dso_ood_dump; CFG=configs/p3former/p3former_2xb1_3x_dso_ood.py; CKPT=work_dirs/p3former_2xb1_3x_dso/epoch_36.pth
python tools/extract_point_features.py $CFG $CKPT --ann dso_infos_cetran.pkl --out-dir $W/features_cetran --check-dump $W/logits     # ~6 min, GPU
python tools/extract_point_features.py $CFG $CKPT --ann dso_infos_test.pkl --out-dir $W/features_test --check-dump $W/logits_test  # ~7 min
python tools/ood_class_resemblance.py --device cuda:0                                               # -> $W/resemblance/ (~1.5 min)
python tools/ood_class_resemblance.py --device cuda:0 --k 50 --out-dir $W/resemblance_k50
python tools/ood_class_resemblance.py --device cuda:0 --reference test_cetran --out-dir $W/resemblance_xref
python tools/plot_ood_class_resemblance.py [$W/resemblance_xref] [--space appearance]
python tools/plot_feature_tsne.py [--space appearance]
```

**Table A — what the OOD points resemble**: the top 8 by r_OOD (full space), as r_OOD % / r_ID %, with the improvement of {c} | rest in brackets.

| rank | Cetran, own bank | Cetran, T+C bank | Test | Test + Cetran |
| --- | --- | --- | --- | --- |
| 1 | person 29.8 / 0.10 (−7.1) | gate 28.3 / 0.95 (+17.5) | building 26.0 / 0.51 (+16.3) | building 19.3 / 0.54 (+13.3) |
| 2 | perimeter-barrier 28.1 / 1.14 (−11.7) | perimeter-barrier 15.5 / 1.78 (−11.7) | perimeter-barrier 15.9 / 2.78 (+5.5) | gate 15.8 / 0.95 (+10.1) |
| 3 | other-barrier 11.8 / 1.36 (−13.8) | person 13.0 / 0.08 (−7.1) | gate 9.0 / 0.75 (+6.6) | perimeter-barrier 13.4 / 1.78 (+1.2) |
| 4 | building 6.9 / 0.31 (+4.3) | other-barrier 8.1 / 0.75 (−13.8) | other-barrier 5.4 / 0.88 (−1.3) | person 6.2 / 0.08 (−46.7) |
| 5 | terrain 5.2 / 1.19 (−52.4) | building 5.0 / 0.54 (+4.3) | traffic-sign 5.1 / 0.40 (−9.1) | other-barrier 6.0 / 0.75 (−5.2) |
| 6 | sidewalk 3.3 / 9.85 (−52.2) | truck 5.0 / 0.21 (+25.2) | drain 4.1 / 1.10 (−24.2) | sidewalk 3.9 / 2.54 (−29.9) |
| 7 | pole-like-object 3.2 / 0.11 (−18.0) | sidewalk 4.2 / 2.54 (−52.2) | bus 4.0 / 0.07 (−6.1) | drain 3.7 / 0.97 (−24.5) |
| 8 | traffic-cone 3.0 / 0.32 (−29.6) | traffic-sign 2.9 / 0.18 (−5.2) | sidewalk 3.9 / 1.01 (−23.5) | bus 3.6 / 0.09 (−7.0) |

**Table B — class level**: Spearman ρ over the classes, full / appearance space. For Cetran with its own bank, ρ is over the 14 measured classes.

| run | set | r_OOD ~ improvement | r_OOD ~ OOD div % | contrast ~ improvement |
| --- | --- | --- | --- | --- |
| k = 10 | Cetran (own bank) | 0.34 / 0.46 | 0.27 / 0.23 | 0.69 / 0.82 |
| k = 10 | Cetran (T+C bank) | 0.62 / 0.67 | 0.36 / 0.37 | 0.71 / 0.73 |
| k = 10 | Test | 0.76 / 0.91 | 0.61 / 0.63 | 0.22 / 0.45 |
| k = 10 | Test + Cetran | 0.63 / 0.80 | 0.56 / 0.54 | 0.33 / 0.39 |
| k = 50 | Cetran (own bank) | 0.27 / 0.45 | 0.29 / 0.21 | 0.67 / 0.82 |
| k = 50 | Test | 0.73 / 0.91 | 0.60 / 0.63 | 0.24 / 0.43 |
| k = 50 | Test + Cetran | 0.62 / 0.79 | 0.53 / 0.54 | 0.31 / 0.45 |

**Table C — split level**: ρ with the improvement over all 503 splits, full / appearance.

| set | together: R_A | together: E_A | cut through: OOD feature-divided | cut through: log ratio |
| --- | --- | --- | --- | --- |
| Cetran (own bank) | 0.18 / 0.23 | 0.22 / 0.30 | 0.06 / 0.22 | 0.63 / 0.70 |
| Cetran (T+C bank) | 0.34 / 0.36 | 0.38 / 0.42 | 0.35 / 0.36 | 0.59 / 0.63 |
| Test | 0.43 / 0.48 | 0.49 / 0.59 | 0.34 / 0.44 | 0.50 / 0.55 |
| Test + Cetran | 0.40 / 0.44 | 0.48 / 0.56 | 0.35 / 0.41 | 0.60 / 0.65 |

**Literal reading** (`placement.tsv`, full space): the splits grouped by where the 2 (3) most-resembled classes sit, as the median improvement and the % of splits with improvement > 0.

| set | classes | small side (all in A) | large side (all in B) | apart |
| --- | --- | --- | --- | --- |
| Cetran (own bank) | person, perimeter-barrier | −12.3, 2 % (n = 41) | −15.7, 17 % (276) | −12.7, 8 % (186) |
| Cetran (T+C bank) | gate, perimeter-barrier | −15.5, 0 % (n = 44) | −20.4, 11 % (265) | −10.4, 17 % (194) |
| Test | building, perimeter-barrier | −2.7, 45 % (40) | −18.3, 12 % (269) | −2.8, 45 % (194) |
| Test | + gate | −5.8, 33 % (12) | −21.5, 11 % (194) | −4.5, 37 % (297) |
| Test + Cetran | building, gate | −2.6, 29 % (41) | −18.3, 15 % (256) | −7.3, 24 % (206) |
| Test + Cetran | + perimeter-barrier | −11.6, 0 % (12) | −25.1, 9 % (194) | −6.1, 28 % (297) |

**Table D — the Part 1 robust splits in the features** (Test + Cetran, full space; the log ratio in the appearance space in brackets).

| split | R_A % | E_A | OOD feat-div % | ID feat-div % | log ratio | imp |
| --- | --- | --- | --- | --- | --- | --- |
| `s0.1.3.5.10.16` | 14.6 | 0.59 | 33.9 | 17.3 | 0.29 (0.40) | +18.7 |
| `s2.3.4.5.6.16` | 15.6 | 0.63 | 33.5 | 4.8 | 0.84 (1.00) | +21.0 |
| `s0.2.5.16.17.18.19` | 30.6 | 1.05 | 58.5 | 14.0 | 0.62 (0.66) | +17.3 |
| `s2.3.10.12.17.18.19` | 46.5 | 1.59 | 69.2 | 27.2 | 0.41 (0.37) | +18.0 |
| `s10.12.17.18.19` | 42.4 | 2.04 | 67.1 | 26.5 | 0.40 (0.37) | +17.9 |
| `s0.1.2.3.4.6.7.10.16.17.18` | 34.6 | 0.75 | 62.6 | 23.5 | 0.43 (0.41) | +14.8 |
| `s3.16.18` | 6.5 | 0.52 | 18.5 | 5.5 | 0.53 (0.80) | +12.7 |
| `s7.13.16.17.23` | 23.8 | 1.14 | 52.7 | 12.3 | 0.63 (0.69) | +12.2 |
| `s10.12.17.19` | 40.3 | 2.42 | 65.2 | 24.6 | 0.42 (0.39) | +18.3 |
| `s2.16` | 2.3 | 0.27 | 6.4 | 3.2 | 0.29 (0.76) | +26.2 |

**Figures** (`$W/resemblance/`, and the same names in `$W/resemblance_xref/`):
- `profile_{full,appearance}`: r_OOD and r_ID per class, with the appearance-space r_OOD as rings;
- `bubble_feature_singletons_{full,appearance}`: OOD vs ID feature-divided share of the single-class splits;
- `hypothesis_{full,appearance}`: improvement against R_A, the OOD feature-divided share and the log ratio;
- `tsne_{full,appearance}`: t-SNE of the feature samples, with one fixed colour per class and OOD in black (in `$W/resemblance/` only).

**Reading.**
- **What the OOD points resemble: mostly structures.**
  - On Test: building (26 %), perimeter-barrier (16 %) and gate (9 %).
  - On Test + Cetran: building, gate and perimeter-barrier (19 / 16 / 13 %).
  - On Cetran against the full bank: gate leads (28 %), then perimeter-barrier and person (16 / 13 %). Truck is sixth (5.0 %), with a contrast of 24.

  The resemblance is spread: the top class takes 19–30 % of the neighbours, and the top three about half.
- **At the class level, resemblance matches the winning splits, with one exception.**
  - On Test, the three most-resembled classes are exactly the three positive single-class splits besides overhead-bridge.
  - Over the classes, ρ(r_OOD, improvement) is 0.76 on Test, 0.63 on Test + Cetran and 0.62 on Cetran with the full bank. It is only 0.34 over Cetran's own 14 measured classes, which lack truck and gate.
  - r_OOD also tracks Part 1's OOD divided share (ρ 0.61 / 0.56 / 0.36): OOD points that neighbour class c in the features also carry classifier mass on c.
  - **Overhead-bridge is the exception.** It is the best split on Test and Test + Cetran (+34.8 / +27.4), yet its r_OOD is only 1.4 %, below no preference. In the appearance space it rises to 6.0 %, and 21 % of the OOD points are feature-divided on it, against 2.4 % of the ID points. So the positional part of `pe_features` hides this resemblance from the kNN, while the linear classifier still puts overhead-bridge mass on 18 % of the OOD points (Part 1).
- **The appearance space**, without the positional embedding, strengthens the class-level alignment with the improvement: ρ(r_OOD, improvement) is 0.91 / 0.80 / 0.67 on Test / Test + Cetran / Cetran (full bank), against 0.76 / 0.63 / 0.62. The top classes stay the same.
- **At the split level, the literal hypothesis is not supported.**
  - Putting the two or three most-resembled classes together on the small side does no better than separating them. The share of splits with a positive improvement is:
    - Test: 45 vs 45 %;
    - Test + Cetran: 29 vs 24 % for the top 2, and 0 vs 28 % for the top 3;
    - Cetran: 0–7 vs 6–17 %.
  - What does hurt on Test and Test + Cetran is lumping those classes with everything else: the large side has a median improvement of −18 to −25.
  - An example from Test: {overhead-bridge} scores +34.8 and {building} +16.3, but {building, overhead-bridge} only +14.3, although its E_A (3.3) is ten times {overhead-bridge}'s (0.34).
  - Over all splits, the "cut through" statistic (the log OOD/ID feature-divided ratio) predicts on every set, in both spaces and with either bank: ρ 0.50–0.70. "Together" (E_A) predicts on Test and Test + Cetran (0.48–0.59) but only weakly on Cetran (0.22 with its own bank, 0.38 with the full bank).
  - The top robust splits (Table D) have E_A from 0.27 to 2.4, so "togetherness" is not what they share. What they share is a log ratio ≥ 0.29: their OOD points are feature-divided 2–7× more often than their ID points.
- **The hypothesis, refined.** The best split cuts *through* the OOD points' neighbourhoods. It puts one (or a few) of the classes the OOD points lean toward against the rest. The OOD mass, spread over several resembled classes, then straddles the boundary, while each ID class's usual confusers stay on its own side. This is Part 1's divided precision seen from the features. The feature ratio ranks the splits less well than the logit-space precision does (ρ 0.50–0.70 vs 0.72–0.84), so the neighbourhoods explain part of the effect, not all of it.
- **k = 50.** The r_OOD ranking barely moves: its rank ρ with k = 10 is 0.95–0.996, with the same top three. The headline ρ of Tables B and C move by ≤ 0.07.
- **t-SNE** (`tsne_{full,appearance}`; 300 ID points per class and 1,000 OOD points per set). This view is qualitative: t-SNE keeps neighbourhoods, not distances.
  - In the appearance space, the ID classes form clean clusters. The OOD points gather in a region of their own, which borders building, window, perimeter-barrier, gate and other-barrier on Test and Test + Cetran, and perimeter-barrier, other-barrier and person on Cetran. This is the kNN picture of Table A.
  - In the full space, position splits every class into several clusters, and the OOD points spread among them, mostly next to the barriers, building and window. On Cetran, the Stop points form small tight clusters of their own.
- **Caveats.**
  - The test is correlational only, over splits selected on the evaluation data.
  - The kNN reads neighbourhoods, while the classifier is linear on the same features, and overhead-bridge shows the two can disagree.
  - Cetran's own-bank numbers cover only its 14 measured classes.
