# Methodology: Group‑Balanced Feature Pairing and LOOCV Regression

## Overview

We design a reproducible pipeline to (i) extract quantum‑chemistry and geometry descriptors from Gaussian log files, (ii) align them to multiple substituent groups (Ar1, Ar2, …) listed in a reaction/activity spreadsheet, and (iii) perform group‑balanced feature selection with linear regression evaluated by leave‑one‑out cross‑validation (LOOCV). The implementation is provided in `extractor_regr.py`.

## Inputs

* **Log corpus**: Gaussian `.log` files per substituent (e.g., `101.log`).
* **Sheet**: Excel file with columns `Ar1`, `Ar2`, … and target `ln(kobs)`.

## Step 1: Key normalization & entity resolution

1. **Canonical column detection.** Columns matching `[Aa][Rr]\s*\d+` are renamed to canonical `Ar1`, `Ar2`, …
2. **Key normalization.** All Ar cell values are normalized as strings (trim spaces; convert `123.0 → 123`; map `"nan"/"None"/empty` to missing). This guarantees type‑stable joins between the sheet and log features.
3. **Unique entity set.** We build the union of all non‑missing Ar values across `Ar1..ArN`, forming the set of substituents to parse from logs.

## Step 2: Descriptor extraction from logs

For each unique Ar entity with a matching log file, we parse and compute:

* **Frontier orbitals**: HOMO and LUMO energies from SCF population blocks.
* **Dipole moment**: total Debye from the dipole summary.
* **Polarizability**: diagonal terms averaged from the “Exact polarizability” block.
* **NBO summary**: locate the O–H bond(s), identify the C1–C2 scaffold (single/single/double neighborhood criterion), then extract

  * NBO occupancies/energies for C1–O and C1–C2 bonds,
  * Atomic charges (C1, C2, O=, O−) from the Natural Population Analysis summary.
* **Vibrations**: IR intensity and frequency for the C=O stretch by finding modes in 1800–1900 cm⁻¹ with maximal C–D relative displacement.
* **Geometry**: C1–C2 distance from the last “Standard orientation”.
* **Sterimol** (via *morfeus*): Using a filtered XYZ (excluding {a,b,d} atoms), compute L, B1, B5 along the C1→C2 axis with Bondi radii (H radius set to 1.09 Å).

The result is a per‑Ar feature table containing, e.g., `Ar_NBO_C2`, `Ar_NBO_=O`, `Ar_NBO_-O`, `Ar_v_C=O`, `Ar_I_C=O`, `Ar_dp`, `Ar_polar`, `Ar_LUMO`, `Ar_HOMO`, `L_C1_C2`, `Ar_Ster_L`, `Ar_Ster_B1`, `Ar_Ster_B5`, plus index bookkeeping (`Ar_c`, `Ar_e`, …) for diagnostics.

## Step 3: Quality filters & diagnostics

* We drop Ar rows with missing **essential descriptors** to ensure downstream stability.
* We emit a diagnostic report enumerating molecules whose index inference was incomplete (for manual inspection).

## Step 4: Prefix join to build per‑pair design matrix

For each Ar column in the sheet (Ar1, Ar2, …):

1. **Value normalization** (same as Step 1) to guarantee a stable key.
2. **Prefix merge** the per‑Ar table with the current Ar column, adding a prefix `ArK_` (e.g., `Ar2_Ar_NBO_C2`).
3. **Meta cleanup**: drop join helper fields (`*_Ar`, `*_Ar_key`, `*_log_path`, `*_log_exists`).

The final feature set **F** consists of all columns matching `^Ar\d+_` except the helper suffixes above. The target is `y = ln(kobs)`.

## Step 5: Group‑balanced combinatorial feature search

Let the Ar columns define **groups** G = {Ar1, Ar2, …}. Each group g∈G owns the subset F\_g = {features with prefix g\_}. We search linear models of size k∈{1…K\_max} subject to group balance constraints:

* **Bounds model** (default): for each g, enforce `L_g ≤ |S ∩ F_g| ≤ U_g` with L\_g≥1 by default (each Ar contributes at least one descriptor). If k < |G|, such allocations are skipped.
* **Equal model** (optional): approximately equal allocation per group (`floor(k/|G|)` to `ceil(k/|G|)`).

### Allocation enumeration

We enumerate integer allocations a = (a\_1,…,a\_m) with sum(a\_i)=k and per‑group bounds using a depth‑first generator. For each feasible a, we build the Cartesian product of `C(|F_g|, a_g)` choices and concatenate the selected columns to form a candidate set S of size k. A cap on the number of combinations per k avoids combinatorial explosion via randomized thinning.

## Step 6: Model fitting & LOOCV evaluation

For each candidate feature set S:

1. **Fit** ordinary least squares with a tiny ridge (1e‑8 I) or pseudo‑inverse fallback for numerical stability.
2. **Compute** the hat matrix H and the LOOCV predictions using the closed‑form formula `y^(−i) = (ŷ_i − h_i y_i)/(1−h_i)` without refitting.
3. **Report** performance metrics: in‑sample R², LOOCV Q², and RMSE.
4. **Filter**: keep only models with R² ≥ τ (default τ=0.7) for efficiency; finally rank by **Q²**.

## Step 7: Best model selection & visualization

* **Selection**: choose the model with maximum Q²; report coefficients, intercept, R², Q², RMSE, and the exact column names (which record the group attribution via their prefix).
* **Plot**: generate a parity plot (Predicted vs. Experimental) with the regression equation, R², Q²(LOO), RMSE, and sample count annotated.

## Robustness & fallbacks

* If no prefixed groups are detected after merging (e.g., empty feature set due to missing logs), the pipeline **falls back** to an ungrouped exhaustive search over F.
* All joins and entity matching are **string‑normalized**, ensuring compatibility across `int`, `float`, and `str` identifiers (e.g., `101`, `"101"`, `101.0`).
* Numerical stability is ensured with ridge regularization and pseudo‑inverse fallback when `XᵀX` is near‑singular.

## Advantages of the method

1. **Chemically faithful pairing**: Enforces that each substituent group (Ar1/Ar2/…) contributes descriptors, aligning with mechanistic symmetry in paired systems.
2. **Efficient LOOCV**: Closed‑form LOO avoids k‑fold refitting and enables exhaustive (or near‑exhaustive) search under bounds.
3. **End‑to‑end reproducibility**: From raw logs to ranked models with diagnostics, all steps are scripted and versionable.

## Key hyperparameters (defaults)

* `K_max = 5` (maximum features per model)
* `τ = 0.7` (minimum in‑sample R² to retain a candidate)
* `L_g = 1, U_g = min(3, |F_g|)` (per‑group bounds)
* `max_combinations_per_k = 20000` (cap with random thinning)
