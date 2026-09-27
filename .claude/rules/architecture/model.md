---
paths:
  - "p3former/**/*.py"
  - "configs/_base_/models/*.py"
---

# Model: segmentor and P3Former head

`_P3Former` (`p3former/segmentors/p3former.py`, extends mmdet3d `Cylinder3D`) is a thin orchestrator: `SegVFE` → `_Asymm3DSpconv` sparse-conv U-Net (`p3former/backbones/cylinder3d.py`) → `_P3FormerHead`.

The head (`p3former/decode_heads/p3former_head.py`):

- **MPE**: polar and cartesian positional embeddings of the voxel coordinates, fused into the backbone features (`pe_type='mpe'`).
- **Queries**: 128 learnable thing queries plus per-class stuff queries from `sem_queries`. An auxiliary semantic branch (`use_sem_loss`) trains with CrossEntropy + Lovasz on voxel labels; its logits (`sem_preds`) are what the OOD scores use.
- **Decoder**: 6 `_Transformer_Decoder` layers, each running `_Masked_Focal_Attention` (gated masked cross-attention over voxels where the previous mask is >0.5), then self-attention, then an FFN.
- **PA-Seg**: masks are predicted from the features and, separately, from the positional embeddings (`fc_coor_mask`, dice loss, `pa_seg_weight=0.2`).
- **Training**: per-layer Hungarian matching on thing queries only. Layer 0 uses a mask-only assigner. Stuff queries are matched by fixed class index. `_MaskPseudoSampler` wraps the assignment results.
- **Inference** (`generate_panoptic_results`): drop queries below score 0.4, take a per-voxel argmax over the score-weighted masks, and drop masks whose surviving/original area is below `iou_thr` (0.8).
