# Group‑Balanced Pairing (How every Ar is always selected)

This project enforces **paired substituent selection** when building regression models from multi‑Ar spreadsheets (e.g., `Ar1`, `Ar2`, ...). The core idea: **every Ar group must contribute at least one descriptor** to any candidate model, so the final model is chemically “paired” across the substituent positions.

## Why we need it

In paired systems (e.g., azoarenes), each row contains multiple substituent groups (Ar1, Ar2, ...). If we allow the model to cherry‑pick from only one position, we lose symmetry and interpretability. Group‑balanced pairing restores mechanistic symmetry by forcing each Ar position to participate.

## How it works (high‑level)

1. **Prefix grouping** — After feature extraction from Gaussian logs, descriptors are merged back to the sheet with prefixes: `Ar1_*`, `Ar2_*`, ... We then group features by these prefixes: `F_Ar1, F_Ar2, ...`.
2. **Lower‑bound constraint** — During combinatorial search, each group `g` has bounds `L_g ≤ a_g ≤ U_g` with the **default** `L_g = 1`. This guarantees **each Ar contributes at least one feature**. If the desired model size `k` is smaller than the number of groups, that `k` is skipped because it cannot satisfy the constraints.
3. **Feasible allocations** — We enumerate all integer allocations `a = (a_1, ..., a_m)` such that `∑ a_g = k` and bounds hold. For each allocation, we take Cartesian products of per‑group combinations to build candidate feature sets `S` of size `k`.
4. **Evaluation via closed‑form LOOCV** — For each candidate set `S`, we fit OLS with a tiny ridge for stability and compute LOOCV predictions using the hat‑matrix formula. We rank models by **Q²** (with an `R²` pre‑filter for speed).
5. **Best model** — The top‑Q² model reports coefficients, intercept, `R²`, `Q²`, RMSE, and feature names that encode their Ar origin (e.g., `Ar2_Ar_NBO_C2`).

## Exact constraint used in code

* Groups: `G = {Ar1, Ar2, ...}` from feature prefixes.
* Feature pools: `F_g = { f | f starts with g_ }`.
* For a model size `k` and group counts `a_g`:

  * Bounds: `L_g ≤ a_g ≤ U_g` with defaults `L_g = 1`, `U_g = min(3, |F_g|)`.
  * Sum: `∑_{g∈G} a_g = k`.
  * If `k < |G|`, skip (cannot meet `L_g = 1`).

This is implemented in `search_best_models_general(...)` with a **bounded integer composition** generator and per‑group combination enumeration. A cap `max_combinations_per_k` prevents explosion; when exceeded, randomized thinning keeps a diverse sample without violating bounds.

## Practical notes

* **Normalization** — All Ar IDs are string‑normalized (e.g., `101`, `"101"`, `101.0` → `"101"`) so merges never fail on types.
* **Missing logs** — If a group has no valid features (e.g., missing logs), it won’t be formed; if no groups exist at all, the pipeline falls back to an ungrouped search.
* **Defaults** — `K_max = 5`, `R²` filter `≥ 0.7`, per‑group bounds `(1, min(3, |F_g|))`.

## Minimal example

Suppose groups `G = {Ar1, Ar2, Ar3}` and `k = 5`. Feasible allocations include `(1,2,2)`, `(2,1,2)`, `(2,2,1)`, `(1,1,3)`, etc. Because `L_g = 1`, **every allocation includes each group at least once** → every candidate model is paired across all Ar positions.

## Where to look in code

* Grouped search: `search_best_models_general(...)`
* LOOCV metrics: `compute_loocv_metrics(...)`
* Group construction from prefixes: `run_full_pipeline(...): STEP 4`
* Key normalization and prefix merge: `run_full_pipeline(...): STEP 2–3`
