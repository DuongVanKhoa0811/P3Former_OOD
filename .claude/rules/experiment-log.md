# Experiment log and reference documents

- `DOCs.md` is the dated working log, newest at the bottom. Each experiment gets a concise entry, `## YYYY-MM-DD — <title>`: what was run and why, the key command, the headline result table and the conclusion. Read the relevant entry before re-running an experiment.
- When the full record is long (more than about 40 lines: every command, run scripts, job or crash history, per-set or per-seed tables), it goes to `docs/experiments/<entry date>-<slug>.md`, and the `DOCs.md` entry keeps the summary and links to it.
- Design specs and plans are in `docs/superpowers/`.
- The baseline papers are in `docs/others/baselines/`, and the OOD hyperparameters come from `OOD_Baseline.pdf`.
