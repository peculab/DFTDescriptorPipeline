# Results Folder Guide (All 4 Case Studies)

This `results/` directory is organized by **case study**:

- `results/azoarene/`
- `results/heck_boronic_acids/`
- `results/indigo_aryl_alkyl/`
- `results/indigo_diaryl/`

Each case folder contains the same set of outputs.

## What to check first (fastest path)

1) `*_top5_models.csv`  
   The best **five** regression models (formula + R² + Q² + plot filename). Start here.

2) `top1_*_Regression_Plot.html` (and `top2...top5`)  
   Interactive regression plots (open in browser). Use these for slides/papers.

## What each file is for

- `*_features.csv` — modeling-ready dataset used for regression.
- `*_subset_search.csv` — full record of all candidate subsets evaluated.
- `*_top_models_by_k.csv` — ranked models (often grouped by `k`).
- `*_regression_report.json` — machine-readable summary for reproducibility.
- `*_Regression_Plot.html` — (if present) the “best overall” plot.

## Reproducibility note
If you re-run the pipeline, these files may be overwritten. If you want to keep a snapshot,
copy the case folder to a timestamped directory, e.g. `results/azoarene_2026-01-17/`.
