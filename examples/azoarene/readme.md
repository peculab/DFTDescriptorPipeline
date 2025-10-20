# Azoarene Descriptor Extraction & Regression Modeling 🧪📈

This project provides a complete pipeline for **extracting quantum chemical descriptors from Gaussian log files** and performing **regression analysis** to predict reaction rates (e.g., `ln(kobs)`) of azoarene compounds.

It includes:
- A feature extraction engine (`extractor_regr.py`)
- A demonstration notebook (`azoarene.ipynb`) to visualize and run the full workflow
- Automated Sterimol, NBO, HOMO/LUMO, dipole, and vibrational feature extraction
- Model selection and evaluation using LOOCV

---

## 🧠 What This Project Does

- Automated extraction from Gaussian `.log` files  
- Sterimol descriptors via `morfeus-ml`  
- NBO charge parsing (C1–C2/O atoms)  
- Dipole, HOMO-LUMO, polarizability, vibrational features  
- Regression modeling using LOOCV (Q², R², RMSE)  
- Visual regression plot and result export  
- Compatibility with Ar1–Ar2 substituted azoarene systems  

## ⚙️ How to Use

### 1. Install dependencies

```bash
pip install -r requirements.txt
# or manually:
pip install pandas numpy scikit-learn matplotlib morfeus-ml
````

> The script will auto-install `morfeus-ml` if missing and prompt you to restart the kernel.

---

### 2. Prepare input files

* Gaussian `.log` files in `logfiles/`
* An Excel file (`data.xlsx`) with columns:

  * `Compound`, `Ar1`, `Ar2`, `ln(kobs)`

---

### 3. Run the pipeline

#### Option 1: Script (recommended for batch mode)

```python
from extractor_regr import run_full_pipeline

run_full_pipeline(
    log_folder='logfiles',
    xlsx_path='data.xlsx',
    target='ln(kobs)',
    output_path='final_output.xlsx',
    plot_path='Regression_Plot.png',
    auto_pairing=True  # Automatically match Ar1 and Ar2 features
)
```

#### Option 2: Jupyter Notebook

Open `azoarene.ipynb` to:

* Step through data loading
* Visualize intermediate results
* Customize regression parameters

---

## 📦 Output Files

* `final_output.xlsx`: Clean dataset with descriptors and predictions
* `regression_search_results.csv`: All tested regression models with Q², R², RMSE
* `Regression_Plot.png`: Experimental vs predicted values
* `problem_index_report.xlsx`: List of molecules with incomplete or failed extractions
* `unique_ar_features.xlsx`: Extracted features per unique Ar group

---

## 🧠 Methods

### Extracted Descriptors

* **HOMO/LUMO** from SCF section
* **Dipole Moment** (Debye)
* **Polarizability** tensor (averaged)
* **Sterimol Parameters**: L, B1, B5 via morfeus
* **NBO Charges**: C1, C2, O atoms
* **Bond Distances & Vibration**: C=O vibrational frequency and intensity

### Regression

* Linear model with Leave-One-Out Cross-Validation (LOOCV)
* R² (fitting), Q² (cross-validation), RMSE
* Optional: Force Ar1–Ar2 feature balancing

---

## 📜 License

MIT License. Feel free to adapt and cite.
