# Results Guide — azoarene

This folder contains the **regression outputs for the `azoarene` case**.

## Quick start: what to open first

1) **Top 5 best models (recommended entry point)**  
   - `azoarene_top5_models.csv`  
   What you get:
   - the best **five** regression formulas ranked by (typically) **Q² then R²**
   - the selected feature set for each model
   - the corresponding **R²** and **Q²**
   - the **plot filename** you can open to see the regression fit

2) **Regression plots for the top models**  
   - `top1_*_Regression_Plot.html`, `top2_*_Regression_Plot.html`, …  
   Open in a browser. These show **predicted vs observed** (and the displayed metrics).

## Deeper dive: all artifacts and what they mean

### A) Modeling-ready table
- `azoarene_features.csv`  
  The merged, modeling-ready dataset used for regression.

### B) Subset search log
- `azoarene_subset_search.csv`  
  Every candidate feature subset that was evaluated (useful for auditing / reproducibility).

### C) Per-k ranking table
- `azoarene_top_models_by_k.csv`  
  Ranked models (often one file that aggregates multiple `k` values, or multiple files by `k`).

### D) Machine-readable report
- `azoarene_regression_report.json`  
  A structured summary for programmatic reading (best model, coefficients, metrics, settings).

### E) Best-model plot (sometimes produced by the regression script)
- `azoarene_Regression_Plot.html`  
  If present, this is typically the **single best** plot produced by the regression module.

## How to cite / report the final numbers
When you write up results, use:
- **Top model metrics** from `azoarene_top5_models.csv` (row 1)  
- The **exact formula** from the same row (equation column)

## Tips
- If a plot doesn’t open correctly in your browser, try Chrome/Edge.
- CSVs are easiest to inspect in Excel; JSON is easiest in VS Code.
