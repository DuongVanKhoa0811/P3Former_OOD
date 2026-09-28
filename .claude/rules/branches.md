# OOD branches

On 2026-09-29 the old `ood-baselines` branch was split into stacked branches, one per line of research:

| Branch | Content | Built on |
| --- | --- | --- |
| `ood-baselines/flat` | flat point-level OOD baselines (MSP, MaxLogit, ODIN, Energy, Entropy), the OOD metric and the `_ood` configs | `dso-dataset` (merged in) |
| `ood-baselines/grouping` | Group / GN scores, the class-hierarchy ablation, per-point logit dumps, the bipartition sweep | `ood-baselines/flat` |
| `ood-baselines/occuq` | OCCUQ-style feature-density uncertainty (planned) | `ood-baselines/flat` |
| `duy/ood-baselines` | a collaborator's feature-distance (Mahalanobis) work; not merged | the pre-split history at `90af8ea` |
| `archive/ood-baselines-pre-split` | the old `ood-baselines` as it was before the split | — |

- Fix shared code on `ood-baselines/flat` first: scores, metric, head and segmentor plumbing, shared configs and shared rules. Then bring it into the other branches with `git merge ood-baselines/flat`. Never merge `grouping` or `occuq` into `flat`.
- `DOCs.md` and the progress log in `.claude/CLAUDE.md` are per branch: log an experiment on the branch that ran it.
- A plain `ood-baselines` branch can't exist while any `ood-baselines/*` does: git stores the ref `ood-baselines` as a file and `ood-baselines/*` inside a directory of the same name.
- On GitHub, the ruleset "Restriction for merge and delete" (no deletion, no force-push, pull request required, owner bypass) should target `refs/heads/ood-baselines/**` and `refs/heads/archive/**`.
