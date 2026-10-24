# Center or boundary: where do the robust two-group splits separate ID from OOD?

Date: 2026-10-23
Amended 2026-10-24, twice. First, the work runs on server A, from the logit dumps already
there: the re-dump on server C and its check against A (the old Part A) are dropped, and
the parts are renumbered. Second, the divided-mass and OOD-resemblance analyses are only
mentioned for context: the splits, scores, bins and checks now come from the sweep logs
and this document, not from `tools/divided_mass.py`.
Amended 2026-10-25: the side results are cut down to the distance profile.
Status: draft, under review
Branch: `ood-baselines/center-boundary`, created from `ood-baselines/grouping` at a39291e

## Goal

Test the user's hypothesis, "Point of the object: the center of the object will be stable,
but the boundary will be noisier", for the OOD score of each of the 33 robust two-group
splits. Put as a question: are ID and OOD scores better separated at object centers than
at object boundaries?

The hypothesis came with a whiteboard sketch (2026-10-23), described here so that this
document doesn't depend on the file:

- **A split.** Five bars stand on one axis, one per class. The first is labelled "overhead
  bridge", the others "road", "bus" and "…". A dashed vertical line separates the
  overhead-bridge bar from the rest, and a red ellipse circles the rest. The caption reads
  "S16 {overhead_bridge}": split s16, overhead-bridge against the other 23 classes. It is
  one of the 33 robust splits, and the one with the highest improvement on Test and on
  Test + Cetran.
- **Two objects.** The outlines of a person and of a vehicle. Small red marks sit in their
  middles: crosses on the person's torso, ticks inside the vehicle. Small dark dots run
  along part of each outline: the person's shoulder and arm, and the vehicle's side and
  underside. Read with the hypothesis, the red marks are center points, whose score should
  be stable, and the dots are boundary points, whose score should be noisier.
- **A caption**, "500 bipartitions": the random sweep the robust splits come from.

The sketch draws the boundary as the object's own outline. This work calls a point
boundary when a point of another class lies within 0.5 m instead, as the user chose (see
Decisions).

Three deliverables, each reported on Test and Cetran (and on Test + Cetran for reference):

1. **Distributions and verdict.** The OOD score distribution of ID points against OOD points,
   once for center points and once for boundary points. The overlap is measured by the KL
   divergence, next to AUROC / AP / FPR@95 per region. A verdict rule fixed in this document
   says, per split, whether the hypothesis holds.
2. **PLY samples.** Frames that contain OOD points, one folder per frame and one file per
   view:
   - ground truth: ID green, OOD blue;
   - center/boundary, to check the rule by eye;
   - one file per split: TP / FP / TN / FN at the split's FPR@95 threshold.
3. **A private website** (a claude.ai artifact) that shows the same samples as images, in
   the style of the 2026-10-23 preview (https://claude.ai/artifact/KE5f11F3LkBBuCexGQfKsU),
   together with the distributions and the verdicts. It replaces any change to
   `view_class_frames.py`.

Everything runs on server A, from the existing logit dumps. There is no retraining, no
re-dump and no change to the model code or the configs. **Nothing under `data/` is
written**: the PLY annotations and the info pkls are only read.

## Background

- **Robust splits.** The bipartition sweep (`tools/sweep_bipartitions.py`, 2026-09-15,
  commits 3253d93 and cf92bad) scored 500 random two-group splits A | B of the 24 DSO
  classes on each set. Its `--subsets singletons` mode (commit b48cbca) added the 24
  single-class splits, 503 distinct splits in all. A split is named after its smaller
  side, `s<class ids of A>`.
  - `improvement` = mean ΔAUROC + mean ΔAP − mean ΔFPR@95 over Group MSP, Group Energy and
    Group Entropy against their flat scores, as `summarize_hierarchy_ablation.py --family
    group --exclude odin` computes it from the sweep logs.
  - A split is robust when its improvement is > 0 on Cetran, Test and Test + Cetran. There
    are 33, and the tool lists them from the sweep logs itself ("Sets, labels and scores").
- **Earlier analyses, for context only.** The divided-mass analysis
  (`tools/divided_mass.py`) and the OOD-resemblance analysis
  (`tools/ood_class_resemblance.py`), both of 2026-09-30, looked at why some splits beat
  flat; `.claude/CLAUDE.md` and `DOCs.md` summarize them. The divided-mass analysis calls
  min(P_A, P_B) the divided mass m, and its `divided_mass/robust.tsv` lists the same 33
  splits. This work uses neither analysis: it reads none of their outputs and imports none
  of their code.
- **A split's score is its Group MSP.** For two groups, Group MSP = −max(P_A, P_B) = g − 1,
  with g = min(P_A, P_B) and P_A the softmax mass of side A's classes.
  - Group Entropy is a monotone function of g, so it ranks points the same way.
  - Group Energy, −max(LSE_A, LSE_B), stays within log 2 of flat Energy at every point, and
    flat Energy is one of the references.

  So "a split's score" here means g, with higher meaning more OOD.
- **Scores live on voxels.**
  - The model scores a cylindrical grid of 480 × 360 × 32 voxels over ρ 0–50 m, the full
    circle and z from −3 to 13 m. A voxel is about 0.10 m radially, 0.0175·ρ m around and
    0.5 m high.
  - Every point takes its voxel's logits.
  - Points outside that range are clamped onto the outer voxels
    (`p3former/data_preprocessors/data_preprocessor.py`).
- **Precision.** The head and the sweep compute Group MSP as −max(P_A, P_B) and flat MSP as
  −max_c p_c, in float32. Near −1 a float32 has a step of 2^-24 ≈ 6e-8, so points whose g
  or u is below about that share one score and tie. This work computes g and u in float64
  as direct sums (see Scores), which keeps those points apart, so its numbers can differ a
  little from the sweep's. The check against the sweep (A7) allows for it.
- **Logit dumps** (server A):
  - `work_dirs/p3former_2xb1_3x_dso_ood_dump/logits/` holds Cetran, 980 frames, 15 GB.
  - `logits_test/` holds Test, 2,625 frames, 55 GB. It is on `/mnt/sandisk` and
    symlinked.
  - One npz per frame holds `logits` (f16 [N, 24]), `ood` and `valid` (bool), `mapped`
    (int16) and `lidar_path`.
  - They hold no coordinates. The test pipeline drops no point, so the dump's points are the
    PLY's points in order; this is checked on every frame.
- **Preview and probe.** The preview (made on server C, 2026-10-23) drew two frames by ground
  truth and by center/boundary.
  - Its scripts stay on C, in `work_dirs/center_boundary_preview/`.
  - `.claude/memory.md` ("Point-cloud preview images") describes its frame choice, cameras,
    drawing settings and colours well enough to rebuild them.

  A geometry-only probe of seven frames found:
  - 14% of ID points and 25% of OOD points are boundary;
  - ignored points are 7–14% of a frame;
  - 1–14% of a frame lies beyond 50 m.

  OOD points carry a single dummy instance id per class per frame, so OOD objects can't be
  told apart by instance.

## Sets, labels and scores

| set | key | dump | frames | sequences (frames) |
| --- | --- | --- | --- | --- |
| Cetran | `cetran` | `logits/` | 980 | 2026-01-28 Cetran Run AM (401), 2026-01-28 Cetran Run AM-clean (202), 2026-02-04 Cetran Run AM (377) |
| Test | `test` | `logits_test/` | 2,625 | One-North Route 2 Day (1,486), Ubin Route 1 (681), Ubin Route 3 (458) |
| Test + Cetran | `test_cetran` | both | 3,605 | all six |

The dumps are under `work_dirs/p3former_2xb1_3x_dso_ood_dump/`. Test + Cetran reads both
dumps, as the sweep does; together they are exactly `dso_infos_test_cetran.pkl`.

- **Ground truth**, as in `_OODPointMetric`:
  - **OOD**: the raw id is 17 (Stop) or 28 (Others).
  - **ID**: the mapped label is not 24.
  - **Ignored**: every other point. It is left out of every metric but stays in the
    geometry.
- **Evaluation label** of a point, used only by the boundary rule:
  - its train id 0–23 if it is ID;
  - one shared OOD label for Stop and Others;
  - one ignored label.
- **Edge distance** d: the 3D Euclidean distance from a point to the nearest point of the
  same frame with a different evaluation label, capped at 3 m.
  - Every point of the frame takes part, including ignored points and points beyond 50 m.
  - Two touching instances of one class share a label, so they make no boundary.
- **Region**: boundary when d < 0.5 m, center when d ≥ 0.5 m.
- **Distance bands**, for the profile: [0, 0.25), [0.25, 0.5), [0.5, 1), [1, 2) and ≥ 2 m.
- **The 33 splits** come from the sweep logs alone. Each set has two:
  `bipartitions<sfx>/bipartitions.log` from the random sweep and
  `singletons<sfx>/bipartitions.log`, where `<sfx>` is empty for Cetran, `_test` for Test
  and `_test_cetran` for Test + Cetran.
  - Per set, read both with `parse_logs` of `tools/summarize_hierarchy_ablation.py`, the
    single-class log first, so that the random sweep's row wins for a split in both.
  - A split's improvement is `_improvement` of its Group-family mean from
    `summarise(rows, exclude={'maxlogit', 'odin'})`.
  - Keep the splits with improvement > 0 on all three sets, ordered by the worst of the
    three, highest first (ties by name). Write them to `splits.tsv`: name, side A's class
    ids and names, the improvement per set, and the worst.
  - On the current logs this gives 33 splits, from s0.1.3.5.10.16 (worst 17.87) down to
    s6.8.13.14.15.20.21.22 (0.083). This was checked on 2026-10-24, and A7 checks it again.
- **Scores**: 35 in all, every one higher for more OOD. They are computed from the float16
  logits upcast to float64.
  - g = min(P_A, P_B) for each of the 33 splits. P_A and P_B are each summed over their
    own side's softmax probabilities, never taken as 1 − the other, so that confident
    points keep their precision.
  - Two flat references, which tell an effect of the model apart from one of the grouping:
    - flat MSP, as u = 1 − max_c p_c, computed as the sum of the other classes'
      probabilities;
    - flat Energy = −logsumexp(z), T = 1, as in `p3former/utils/ood_scores.py`.
- **Score order** in every table and figure: flat MSP, flat Energy, then the 33 splits in
  `splits.tsv` order.

## Part A — the analysis (`tools/center_boundary.py`)

### A1. Per frame

The tool reads every frame of both dumps with a pool of worker processes (`--workers`). For
each frame:

1. **Read and check.** Read the dump npz, and read the PLY at `lidar_path` with
   `read_dso_ply`; the PLY is only read. Then check that the two line up. Any mismatch is
   fatal:
   - the number of points equals the PLY's vertex count;
   - `mapped` equals the label map applied to the PLY's semantic ids;
   - `ood` equals "semantic ∈ {17, 28}";
   - `valid` equals `(mapped ≠ 24) | ood`;
   - `lidar_path` is the info-pkl entry in the same order.
2. **Geometry.** Compute the evaluation labels and d. Use one `cKDTree` per label over all
   other points, with `distance_upper_bound` = 3 m. The probe took 3–9 s per frame on one
   core. Then compute the region and the distance band.
3. **Scores.** Compute the 35 scores.
4. **Accumulate.** Add the frame to the histograms (A2).

### A2. Histograms

- **Fine bins**, fixed here and the same for every set:
  - g and u: one bin [0, 1e-40); then 100 log-spaced bins per decade from 1e-40 to 0.01;
    then bins of width 1e-4 up to 0.5 for g, which never exceeds it, and up to 1.0 for u.
    The linear top resolves the highest scores, which decide AP. Every power of ten is an
    edge, 1e-7 and 1e-5 included (A7 uses them).
  - Energy: uniform bins of width 0.005 over [−200, 50]. Points outside that range are
    counted and must be zero.
- **Blocks.** A block is 50 consecutive frames of one sequence, in info-pkl order. A
  remainder shorter than 25 frames joins the previous block, so Test has 53 blocks and
  Cetran 20.
  - Per block: fine histograms of score × {ID, OOD} × {center, boundary}.
  - Sets and bootstrap resamples are sums of blocks.
- **Per set**: fine histograms of score × {ID, OOD} × distance band, for the profile (A5).
- Everything takes about 0.5 GB as int32 counts, saved in `hist.npz`.

### A3. Measures (per set, score and subset of points)

Each measure is taken on a subset of a set's points: all of them, the center points, the
boundary points, or the points of one distance band.

- **Metrics.** AUROC, AP and FPR@95 come from `metrics_from_histograms(hist_ood, hist_id)`
  on the fine bins.
- **KL divergence**, on coarse bins.
  - **Coarse bins.** There are up to 100 per score and set. They have equal mass under the
    equal-weight mixture of that set's ID and OOD distributions, with the regions pooled.
    They are merged from the fine bins at fine edges, and edges that coincide are dropped.
    For a given score and set, every subset uses the same coarse bins.
  - **KL.** Add 0.5 to every count, then normalise. Report KL(ID‖OOD), KL(OOD‖ID) and
    their sum J (Jeffreys), in nats.
  - **Scope.** KL depends on the bins, so it is compared only within one score and set.
- **τ95.** For each score and set, τ95 is the largest fine edge with at least 95% of the
  set's OOD points at or above it. That is the cut at which `metrics_from_histograms`
  reads FPR@95. A point is predicted OOD when its score is at least τ95.

### A4. Verdict (fixed here, before any score is looked at)

- **Differences.** For each score and each of Test and Cetran:
  - ΔJ = J(center) − J(boundary);
  - ΔAUROC = AUROC(center) − AUROC(boundary).
- **95% CI.** A block bootstrap with 1,000 resamples and seed 0. Blocks are drawn with
  replacement within each sequence, and each sequence keeps its block count. The interval
  is the percentile interval.
- **Per set.** "center" when both intervals lie above 0, "boundary" when both lie below 0,
  and "unclear" otherwise.
- **Per score.**
  - "supports" when the set verdict is "center" on both Test and Cetran;
  - "contradicts" when it is "boundary" on both;
  - "mixed" otherwise.

  Test + Cetran is reported but does not vote, since it contains the other two.
- **Headline.** The number of the 33 splits under each verdict, plus the verdicts of the two
  flat references.
- **What votes.** AP and FPR@95 are reported per region but do not vote. AP depends on the
  share of OOD points, and that share differs between the regions (about 25% of the OOD
  points against 14% of the ID points are boundary).

### A5. Distance profile (descriptive, no verdict)

AUROC and J per distance band, for every score and set. It shows how the separation
changes with the distance to another class, and whether the 0.5 m cut matters.

### A6. Sample frames

- **Selection.** Frames are selected from the labels only, before any score is looked at.
  - In each sequence, two frames: the one with the most OOD points, and the one at the
    median OOD count among frames with at least 1,000 OOD points. A tie goes to the lower
    frame id.
  - The two preview frames are added unless already picked:
    `(2024-04-29) One-North Route 2 Day/00000570` and
    `(2026-02-04) Cetran Run AM/00001441`.
  - That makes at most 14 frames. Their rank runs by set (Test first), then sequence in
    pkl order, then the pick (most, median, preview).
- **Close-up window.** It is centred on the frame's largest OOD object. The objects come
  from joining OOD points closer than 0.6 m, as in the preview. The half-size is
  max(8 m, half the object's horizontal extent + 2 m).
- **Sidecar.** Once τ95 is known, each sample frame gets
  `samples/<rank:02d>_<sequence>_<frame>.npz` holding:
  - xyz (f32), the evaluation label, d (f32) and the region;
  - one outcome code per prediction view, for flat MSP and the 33 splits: uint8 with
    0 ignored, 1 TN, 2 TP, 3 FP, 4 FN;
  - the 35 scores (f32, log10 for g and u), the τ95 used and the close-up window.

  The threshold is τ95 of the frame's own set, Test or Cetran.

### A7. Checks

These go to `checks.md`. A check beyond tolerance shows as FAIL and is never hidden, and a
FAIL stops the reading of the results until it is understood.

- **Alignment** of every frame (A1), with the number of frames and points checked. Test has
  2,625 frames and 1,123,003,311 points, as in its dump log.
- **Split list.** It has 33 splits, from s0.1.3.5.10.16 to s6.8.13.14.15.20.21.22.
- **Against the sweep logs.** Per set, the whole-set AUROC, AP and FPR@95 of flat MSP, flat
  Energy and each split's Group MSP, against the same rows of the set's sweep logs, which
  come from the same dumps.
  - The sweep scored in float32 on its own 2^16 bins, so it can't order points whose g or u
    is below about 1e-7. For this check only, the tool merges its fine bins below 1e-7 into
    one.
  - They must agree within 0.1 on AUROC and AP and 0.5 on FPR@95.
  - A split, or flat MSP, whose τ95 is below 1e-5 is listed with its differences but not
    held to the tolerances. Below 1e-5, the sweep's float32 rounding moves points by more
    than 0.6% of the threshold, so its numbers there depend on rounding.
- **Range.** No Energy value falls outside its fine bins, and under 1% of any score's OOD
  points fall in the bottom bin [0, 1e-40).

### A8. Outputs (`center_boundary/`)

- `hist.npz`;
- `splits.tsv`: the 33 splits, as listed from the sweep logs;
- `meta.json`: frames, blocks, bins, τ95, sample frames;
- `metrics.tsv`: long format, with set, score, subset, AUROC, AP, FPR@95, both KLs, J and
  the counts;
- `bootstrap.tsv` and `verdict.tsv`;
- `checks.md`;
- `samples/`.

## Part B — figures and summary (`tools/plot_center_boundary.py`)

The figures use the Agg backend, in the style of `plot_ood_score_distributions.py`. They
are written as PNG at 150 dpi for the website and as PDF for the record.

- **Per score and set**, `plots/<set>/<NN>_<score>`: two panels, center and boundary. Each
  shows the ID and OOD densities over log10 g or log10 u, or over Energy. The panel title
  gives J, both KLs, AUROC, AP and FPR@95.
- **Summary per set**: one row per score, with a dumbbell of AUROC center against boundary
  and its CI, and a second panel for J.
- **Profile per set**: AUROC against distance band, one line per score, with the two
  references drawn bold.
- **`summary.md`**: the checks, the verdict counts, the per-set tables, the distance
  profile and a short reading.

## Part C — PLY samples (`tools/export_center_boundary.py ply`)

- **Layout.** One folder per frame and one file per view:
  `center_boundary/ply/<rank:02d>_<sequence>_<frame>/<NNN>_<view>.ply`. This was chosen
  with the user after the review.
  - The views are `000_gt`, `001_region`, `002_flat-msp`, then `003_…` to `035_…` for the 33
    splits in `splits.tsv` order, named after the split, e.g. `003_s0.1.3.5.10.16.ply`.
  - The sequence name is made filesystem-safe.
  - Folders match the viewer's `^(\d{1,3})_(.+)$`, and the view number comes first in each
    file name. So the current `view_class_frames.py` opens them unchanged:
    - N / P step through the views of one scan and keep the camera;
    - ] / [ move to the next frame, which opens on `000_gt`.

    The other layout, one folder per view, was dropped: there ] / [ jump back to frame 1,
    so comparing views of one scan would take many key presses.
- **Format.** Each file is the source PLY, header and 27-byte records, with only red, green
  and blue replaced. It is written from the sidecar and the source file, which is only read.
- **Colours** (as in the preview):

  | point | colour |
  | --- | --- |
  | ID, TN | #2CA02C |
  | OOD, TP | #1A5FD0 |
  | ignored | #BDBDBD |
  | ID boundary | #FF8C00 |
  | OOD boundary | #D61AD6 |
  | FP | #E41A1C |
  | FN | #E6AB02 |

  A correct prediction keeps its ground-truth colour.
- **`ply/views.tsv`**: one row per view, with the split's classes, τ95 per set and the
  TP / FP / TN / FN counts of every frame.
- **`ply/folders.csv`**: the frame folders' names (set, sequence, frame) in the viewer's
  `class_frames.csv` format (`class_id,name,color`). Run the viewer with
  `--csv <ply>/folders.csv`, so that it shows these names instead of the DSO class names its
  default CSV gives folders 1–28.
- **Size**: about 4.5 GB.

## Part D — the private website (`tools/build_center_boundary_site.py`)

- **Why the page draws the views itself.** Pre-rendered images, as in the preview, would be
  about 14 frames × 36 colourings × 3 views ≈ 1,500 PNGs, or 0.5–1 GB. An artifact version
  holds at most 511 files and 256 MB, and one publish carries at most 64 MB. So the page
  draws the views on a canvas from compact per-frame point data. It uses the preview's
  cameras and colours, described in `.claude/memory.md`.
- **Build.** The tool builds `center_boundary/site/` from the sidecars and the outputs of
  Parts A and B:
  - `index.html`;
  - `manifest.json`: frames, windows, views with each split's classes, τ95, numbers and
    verdict, plus the colours;
  - `frames/<rank>.bin`;
  - `plots/`.
- **Frame data.** Only the points inside the whole-frame view (±45 m) are kept, sorted by z
  so that the top views draw higher points last. Each point stores:
  - x, y, z as int16 centimetres;
  - one byte for ignored / ID center / ID boundary / OOD center / OOD boundary;
  - 2 bits per prediction view (TN, TP, FP, FN) for the 34 views.

  That is about 16 bytes per point, about 6 MB per frame and about 90 MB in all, spread
  over 2–3 publishes. A Python decoder mirrors the page's decoder for the tests.
- **Page.** It looks like the preview: header, rule, legend, light and dark themes, and
  white panels behind the points.
  - **Results**: the verdict counts, the per-set tables and the summary figures.
  - **Samples**:
    - a frame picker and a colouring picker. The colourings are ground truth,
      center/boundary, flat MSP and the 33 splits, each shown by name and side-A classes.
    - Three canvases per frame: the 3D close-up (orthographic, from the sensor toward the
      object, 25° down), the top close-up and the whole frame.
    - Ground truth or center/boundary on the left, the chosen prediction on the right.
    - Click a canvas to enlarge it. The keys follow the viewer with the PLY layout of
      Part C: N / P for the colouring and ] / [ for the frame.
  - **Per split**: its distribution figure for the frame's set, its τ95 and its numbers.
- **Publish.** Claude publishes the site with the Artifact tool as a private page. The link
  goes into `docs/experiments/` and `.claude/memory.md`.

## Running on server A

- **Inputs**, all only read:
  - the two dumps;
  - the PLY annotations under `data/dso/` and the info pkls;
  - the six sweep logs, `bipartitions{,_test,_test_cetran}/bipartitions.log` and
    `singletons{,_test,_test_cetran}/bipartitions.log`.
- **Outputs.** `work_dirs/p3former_2xb1_3x_dso_ood_dump/center_boundary/` is a symlink to
  `/mnt/sandisk/khoadv/P3Former_OOD/p3former_2xb1_3x_dso_ood_dump/center_boundary/`, since
  A's `/` is about 95% full. It holds about 6 GB:
  - `ply/`, about 4.5 GB;
  - `hist.npz`, up to about 0.5 GB;
  - `samples/`, under 1 GB;
  - `site/`, about 0.1 GB.
- **Smoke run** first, on about 20 frames (`--limit 20`).
- **The full run** is one detached CPU job (`nohup setsid`, see `machine-resources.md`). It
  needs no GPU. It runs Part A, then Parts B and C and Part D's build.
  - A is shared with another user, so `--workers` follows the free cores.
  - The edge distances take about 3,605 frames × 6 s / workers, about 25 min with 16
    workers. Reading the 70 GB of dumps adds a few minutes.
  - RAM is about 0.5 GB per worker plus about 0.5 GB of histograms.
- The commands and the run log go into the experiment record.

## Testing and verification

**Unit tests.** They use synthetic data, as pytest functions plus a `__main__` runner, and
add to the existing 120 (~70 s).

- **`tests/test_center_boundary.py`**:
  - d against brute force on random labelled clouds: ignored counts as another label, the
    same label never counts, the cap, and duplicate points;
  - region and band edges, e.g. d = 0.5 is center;
  - the alignment check raising on each kind of mismatch;
  - g and u against brute force in float64 with confident points, so there is no
    cancellation; and Energy against `ood_scores.py`;
  - the split list from small synthetic sweep logs: improvement > 0 required on every
    set, the random sweep's row winning over the single-class one, and the order;
  - the fine bins, with every power of ten an edge;
  - the sweep check: the merge below 1e-7 and the exemption when τ95 is below 1e-5;
  - block assignment: 50 frames per block, the remainder rule, never across sequences;
  - block sums equal direct histograms, and metrics match `binary_ood_metrics`;
  - equal-mass bins, KL and J against a hand-computed case, J = 0 for equal histograms,
    and the smoothing;
  - τ95: at least 95% of OOD at or above it, and the next edge fails;
  - bootstrap determinism with a fixed seed, resampling within sequences, and a CI that
    covers a known difference;
  - every case of the verdict rule;
  - sample selection: most, median, ties, the preview extras, at most 14;
  - outcome codes.
- **`tests/test_export_center_boundary.py`**:
  - the output reads with `read_dso_ply`;
  - every field but the colours is byte-identical to the source;
  - colours by code;
  - folder and file names match the viewer's regex and rank order;
  - `views.tsv`, and `folders.csv` read back with the viewer's `load_csv_meta` rules.
- **`tests/test_build_center_boundary_site.py`**:
  - encoder → Python decoder round trip: xyz within 1 cm, the class codes and all 34
    outcomes;
  - the manifest schema;
  - points beyond ±45 m dropped;
  - the z order.
- **`tests/test_plot_center_boundary.py`**: a smoke run on synthetic outputs.

**Smoke run** of the analysis on about 20 real frames before the full run.

**Review.** An agent reviews the new tools for correctness and maintainability. Its
findings are addressed before the results are reported.

**The website.** One look at the published page: every frame draws, the pickers and keys
work, and both themes read.

## Documentation and git

- **Branch.** This branch starts from `ood-baselines/grouping`.
  - Its `.claude/CLAUDE.md` gets a fresh, short progress log: what it inherits from
    grouping, in a few lines, and this work's status. Grouping's long log stays on
    grouping.
  - The `memory.md` note on the preview images is already on this branch.
- **Shared files.** `.claude/rules/branches.md` gets a row for this branch, and
  `.claude/memory.md` gets the website link. Both change on this branch only. Moving them
  to `ood-baselines/flat`, and so to the other branches, is left for later. Nothing is
  committed on `ood-baselines/occuq`.
- **Rules.** `.claude/rules/offline-tools.md` gets the new tools, and `testing.md` the new
  test count.
- **Log.** A `DOCs.md` entry holds the summary and links to the full record,
  `docs/experiments/<date>-center-boundary.md`: commands, the run log, check results and
  tables.
- **Git.** Commits happen on this branch only, one per plan task. Nothing is pushed unless
  asked.

## Out of scope

- Changing `view_class_frames.py`, which the website replaces.
- Re-dumping the logits; the existing dumps on A are used.
- Re-running the sweep, changing the rule that picks the 33 splits, new scores and
  retraining.
- The divided-mass and OOD-resemblance analyses: mentioned for context, not used.
- Boundaries between instances, since OOD points have no real instance ids.
- Splitting boundaries by the class on the other side (decided: center against boundary
  only).

## Risks and caveats

- **The regions differ in what they hold.** Boundary points come from other classes and
  other ranges than center points, so the comparison is descriptive, not causal.
- **Voxel sharing.** A boundary point often shares a voxel, and so a score, with points of
  another label. That is a likely mechanism of any effect, not a bug.
- **Ground contact.** Anything standing on the ground gets a 0.5 m boundary strip, and an
  object under about 1 m tall is mostly boundary. The preview shows this.
- **Far points.** They are clamped onto the outer voxels and are sparse, so most come out
  center. They stay in every result.
- **Bins.** KL depends on the bins, so it is compared only within a score and set, on bins
  fixed by rule.
- **Correlated frames.** Neighbouring frames are correlated, hence the block bootstrap.
  Cetran has only 20 blocks, so its intervals are wide.
- **Splits near saturation.** For some splits many OOD points have g within a few float32
  steps of 0. The sweep tied those points and this work orders them, so whole-set
  numbers, mostly FPR@95, can differ from the sweep's. A7 lists these splits.
- **Selection bias.** The 33 splits were chosen on these same sets, so the analysis
  describes them. A claim beyond these sets needs other data. Nothing here is tuned on the
  results: the 0.5 m rule, the bins, the blocks and the verdict rule are all fixed above.

## Decisions

**Decided with the user (2026-10-23 to 2026-10-25):**

- The boundary means "near another class": a point with another label within 0.5 m.
- Ignored points count as another label.
- Stop and Others form one OOD label, and instances of one class make no boundary.
- The work runs on server A from the existing dumps, with no re-dump. This replaced the
  first plan of re-dumping on C and checking against A.
- The divided-mass and OOD-resemblance analyses are not used. This document mentions them
  for context only, and the 33 splits come straight from the sweep logs.
- The PLY files go one folder per frame and one file per view, in the dataset's 27-byte
  format, and open in the current viewer. This replaced one folder per view after the
  review.
- The analysis compares center and boundary only.
- The only side result is the distance profile. The inside-grid, per-sequence, spread,
  grouping-gain and composition tables were dropped on 2026-10-25.
- A private website replaces any change to the viewer.
- A new branch starts from grouping, and the `memory.md` note moves to it.
- `data/` stays untouched.

**Calls made in this document** (open to change while reviewing):

- **Scores and precision**: the two flat references, and g and u in float64 from direct
  sums.
- **Verdict**: J and AUROC vote, with CIs, on Test and Cetran separately.
- **Statistics**:
  - the fine bins: the 1e-40 floor, 100 per decade and the linear top;
  - equal-mass coarse bins for KL;
  - the bootstrap, with 50-frame blocks, 1,000 resamples and seed 0;
  - the check against the sweep logs: the merge below 1e-7, the exemption when τ95 is
    below 1e-5, and the tolerances.
- **Samples**: two frames per sequence plus the two preview frames.
- **Colours** and the sidecars.
- **Extra outputs**:
  - the PLY files are kept although the website replaces the viewer change;
  - the website draws its views on the page.
- **Housekeeping**: the output location on `/mnt/sandisk` and the short progress log.
