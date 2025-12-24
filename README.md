# DFTDescriptorPipeline

A lightweight, reproducible pipeline to:

1) parse **Gaussian** `.log` files,  
2) identify a consistent set of atom labels (a–g) around a carboxyl motif,  
3) extract electronic descriptors (energies, HOMO/LUMO, dipole, Mulliken charges),  
4) join with experimental data from Excel, and  
5) run **subset-search regression** (k = 3/4/5) with interactive **Plotly** reports.

This repo is designed for fast iteration across multiple reaction/property datasets.

---

## Features

- **Atom labeling (a–g) from Gaussian logs**  
  Automatically finds the carboxyl carbon and assigns (a, b, c, d, e, f, g) using bonding inferred from geometry.

- **Descriptor extraction**
  - SCF energy, Gibbs free energy, enthalpy
  - dipole (total)
  - HOMO / LUMO / gap (Hartree)
  - Mulliken charges at labeled atoms (q_a … q_g), plus charge mean/std

- **Regression + reporting**
  - subset-search linear models for k = 3/4/5 features
  - metrics: R², Q² (CV), RMSE
  - outputs: interactive Plotly HTML (Actual vs Predicted), coefficient plots, CSVs, and JSON report

---

## Repo layout (expected)

The pipeline expects the following structure under your project root (`--base`):

```text
DFTDescriptorPipeline/
  batch_runner.py
  extract_abcefg_from_logs.py
  extractor_regr.py
  requirements.txt

  azoarene/
    logfiles/*.log
    azoarene_data.xlsx

  heck_boronic_acids/
    logfiles/*.log
    heck_boronic_acids_data.xlsx

  indigo_aryl_alkyl/
    logfiles/*.log
    indigo_aryl_alkyl_data.xlsx

  indigo_diaryl/
    logfiles/*.log
    indigo_diaryl_data.xlsx
```

> Each dataset folder contains:
> - `logfiles/` : Gaussian output logs  
> - `*_data.xlsx` : experimental table (target property column specified in `batch_runner.py`)

---

## Installation

### 1) Create an environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quickstart

### Run the full pipeline (mapping + regression)

From the repo root:

```bash
python batch_runner.py --base .
```

Or provide an absolute path (recommended if you run from elsewhere):

```bash
python batch_runner.py --base "C:\path\to\DFTDescriptorPipeline"
```

### Optional flags

- Skip mapping step:

```bash
python batch_runner.py --base . --skip-mapping
```

- Skip regression step:

```bash
python batch_runner.py --base . --skip-regression
```

- Auto-open Plotly HTML in browser:

```bash
python batch_runner.py --base . --open-plots
```

---

## What gets generated

For each dataset (e.g., `azoarene`) the pipeline generates:

### Step 1) Atom mapping CSV

- `{category}_mapping.csv`  
  Contains: `logfile, a, b, c, d, e, f, g, ok, error, elapsed`

### Step 2) Regression outputs (per k)

For `k ∈ {3,4,5}`:

- `{category}_k{k}_features.csv`  
  The merged modeling table (descriptors + target), for inspection.

- `{category}_k{k}_subset_search.csv`  
  All passing models (if any), sorted by Q²/R²/RMSE.

- `{category}_k{k}_top_models_by_k.csv`  
  Top-ranked models for that k (even if no model meets thresholds).

- `{category}_k{k}_Regression_Plot.html`  
  Interactive Actual vs Predicted plot with metrics & equation.

- `{category}_k{k}_Coef_Plot.html`  
  Coefficient bar chart (original feature units).

- `{category}_k{k}_regression_report.json`  
  Machine-readable summary (best features, metrics, join mode, settings).

---

## Configuration & tuning

The main entry is `batch_runner.py`. It defines:

- dataset list (category, log folder, xlsx path, target column)
- global behavior via environment variables

### Key environment variables

**Cross-validation**

- `CV_MODE=loocv` (default) or `CV_MODE=kfold`
- `N_SPLITS=5` (only for kfold)

**Search speed / breadth**

- `CAND_POOL=25` : candidate pool size
- `MAX_COMBOS=250000` : full enumeration threshold
- `SAMPLE_COMBOS=80000` : random sampling if combinations explode
- `PREFILTER_TOP=0` : set >0 for 2-stage speedup (quick-fit then CV on top N)

**Model choice**

- `REGRESSOR=ols` or `REGRESSOR=ridge`
- `RIDGE_ALPHA=1.0`

**Model selection thresholds**

- `MIN_R2=0.75`
- `MIN_Q2=0.75`

**Plot behavior**

- `OPEN_PLOTS=1` to auto-open generated HTML plots

### Example (Windows PowerShell)

```powershell
$env:CV_MODE="kfold"
$env:N_SPLITS="5"
$env:REGRESSOR="ridge"
$env:RIDGE_ALPHA="10.0"
$env:CAND_POOL="30"
python batch_runner.py --base .
```

### Example (macOS/Linux)

```bash
export CV_MODE=kfold
export N_SPLITS=5
export REGRESSOR=ridge
export RIDGE_ALPHA=10.0
export CAND_POOL=30
python batch_runner.py --base .
```

---

## Notes on pair datasets (Ar1 / Ar2)

If your Excel sheet contains **both** `Ar1` and `Ar2`, the regression script can switch to **pair mode**:

- builds descriptors for `ar1_*` and `ar2_*`
- can optionally add engineered features: `sum_*`, `diff_*`, `absdiff_*`, `prod_*`

This repo defaults to a conservative setting (only Ar1/Ar2 descriptors, no engineered terms).  
If you want engineered terms, set:

```bash
export PAIR_ENGINEERED=1
```

---

## Adding a new dataset

1. Create a new folder:

```text
my_dataset/
  logfiles/*.log
  my_dataset_data.xlsx
```

2. Add a new entry in `CASES` inside `batch_runner.py`:

```python
{
  "category": "my_dataset",
  "log_subdir": r"my_dataset\logfiles",
  "xlsx": r"my_dataset\my_dataset_data.xlsx",
  "target": "your_target_column_name",
}
```

3. Run:

```bash
python batch_runner.py --base .
```

---

## Requirements

See `requirements.txt` (NumPy, Pandas, scikit-learn, Plotly, OpenPyXL, Kaleido).

---

## Citation

If you use this pipeline in academic work, please cite it as:

> DFTDescriptorPipeline, GitHub repository, (https://github.com/peculab/DFTDescriptorPipeline/), accessed YYYY-MM-DD.

---

## License

Choose a license for open-source distribution (MIT is common for research tooling).  
Add a `LICENSE` file at the repo root.
