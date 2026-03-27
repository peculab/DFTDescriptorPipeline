from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ============================================================
# 1) Project root (override by --base)
# ============================================================
BASE = r""

# ============================================================
# 2) Four systems configuration
# ============================================================
CASES: List[Dict[str, str]] = [
    {
        "category": "azoarene",
        "log_subdir": r"azoarene\logfiles",
        "xlsx": r"azoarene\azoarene_data.xlsx",
        "target": "ln(kobs)",
    },
    {
        "category": "heck_boronic_acids",
        "log_subdir": r"heck_boronic_acids\logfiles",
        "xlsx": r"heck_boronic_acids\heck_boronic_acids_data.xlsx",
        "target": "ddG",
    },
    {
        "category": "indigo_aryl_alkyl",
        "log_subdir": r"indigo_aryl_alkyl\logfiles",
        "xlsx": r"indigo_aryl_alkyl\indigo_aryl_alkyl_data.xlsx",
        "target": "ln(kobs)_MeCN",
    },
    {
        "category": "indigo_diaryl",
        "log_subdir": r"indigo_diaryl\logfiles",
        "xlsx": r"indigo_diaryl\indigo_diaryl_data.xlsx",
        "target": "ln(kobs)",
    },
]

# ============================================================
# Global constraints (apply to ALL cases)
# - Only allow Ar1/Ar2 descriptors in PAIR MODE
# - Do NOT allow engineered sum_/diff_/prod_ unless you flip PAIR_ENGINEERED=1
# ============================================================
GLOBAL_ENV = {
    "PAIR_ONLY_AR": "1",
    "PAIR_ENGINEERED": "0",
}

# Per-dataset overrides (kept minimal to avoid affecting other stable cases)
# Easy model swapping:
# - OLS         : REGRESSOR=ols
# - Ridge       : REGRESSOR=ridge      + RIDGE_ALPHA=...
# - SVM (RBF)   : REGRESSOR=svr_rbf    + SVR_C / SVR_EPSILON / SVR_GAMMA
# - SVM (Linear): REGRESSOR=svr_linear + SVR_C / SVR_EPSILON
CASE_ENV_OVERRIDES = {
    "azoarene": {
        "GAUSS_LUMO_MODE": "lastblock_first",  # more robust LUMO when virt blocks split
        "REGRESSOR": "svr_rbf",
        "SVR_C": "10.0",
        "SVR_EPSILON": "0.1",
        "SVR_GAMMA": "scale",
        # "OUTLIER_DROP_TOP": "2",
    }
}

# We want plots for k = 3/4/5
K_LIST = [3, 4, 5]

# Results layout
RESULTS_DIRNAME = "results"
TOP_N_MODELS = 5


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _norm_rel(path_str: str) -> Path:
    """Normalize a relative path that might use Windows backslashes.

    This keeps the config human-readable while making it portable on macOS/Linux.
    """
    s = str(path_str).strip()
    if not s:
        return Path("")
    # Treat both "\\" and "/" as separators
    s = s.replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    return Path(*parts)


def _ensure_import_local_modules(base: Path) -> None:
    """
    Make sure we can import extractor_regr.py even if BASE != current cwd.
    Priority:
      1) BASE (your project root)
      2) this script folder
    """
    base_str = str(base)
    script_str = str(_script_dir())
    if base_str not in sys.path:
        sys.path.insert(0, base_str)
    if script_str not in sys.path:
        sys.path.insert(0, script_str)


def run_mapping_extraction(base: Path) -> None:
    """
    Call extract_abcefg_from_logs.py to generate:
      BASE/{category}_mapping.csv
    """
    extractor = base / "extract_abcefg_from_logs.py"
    if not extractor.exists():
        extractor = _script_dir() / "extract_abcefg_from_logs.py"

    if not extractor.exists():
        raise FileNotFoundError(
            f"Cannot find extract_abcefg_from_logs.py under BASE or script dir.\n"
            f"  BASE      : {base}\n"
            f"  script dir: {_script_dir()}"
        )

    overall_rows = []

    for case in CASES:
        category = case["category"]
        logdir = base / _norm_rel(case["log_subdir"])
        out_csv = base / f"{category}_mapping.csv"

        print("=" * 80)
        print(f"=== Running mapping for {category} ===")
        print(f"  logdir : {logdir}")
        print(f"  output : {out_csv}")

        if not logdir.is_dir():
            print(f"  [WARN] logdir not found, skip: {logdir}")
            continue

        out_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(out_csv, "w", encoding="utf-8") as f:
            subprocess.run(
                [sys.executable, str(extractor), str(logdir), category],
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"  -> mapping CSV written: {out_csv}")

    print("\n=== Mapping for all categories finished. ===\n")


def _load_top_models(top_models_csv: Path, top_n: int) -> List[Dict[str, str]]:
    """Load top models (across all k) from *_top_models_by_k.csv."""
    import pandas as pd

    if not top_models_csv.exists():
        return []

    df = pd.read_csv(top_models_csv)
    if df.empty:
        return []

    df = df.sort_values(["Q2", "R2", "RMSE"], ascending=[False, False, True]).head(top_n)
    return [r._asdict() for r in df.itertuples(index=False)]


def _make_and_save_plot(
    *,
    merged_csv: Path,
    target: str,
    category: str,
    features: List[str],
    out_html: Path,
    title_suffix: str,
) -> Tuple[float, float, str, str]:
    """Refit a model for a given feature set and write a Plotly HTML regression plot.

    Returns: (r2, q2, cv_name, equation_pretty)
    """
    import json
    import math

    import numpy as np
    import pandas as pd

    try:
        import plotly.graph_objects as go
    except Exception as e:
        raise ImportError("Plotly is required: pip install plotly") from e

    from sklearn.metrics import mean_squared_error, r2_score

    # Re-use the exact same helpers as extractor_regr.py to avoid drift.
    from extractor_regr import (  # type: ignore
        _compute_q2_cv,
        _equation_pretty,
        _fit_pipeline,
        _orig_unit_coeffs,
    )

    df = pd.read_csv(merged_csv)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {merged_csv}")

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in merged table: {missing}")

    X = df[features]
    y = pd.to_numeric(df[target], errors="coerce")

    model = _fit_pipeline(X, y)
    yhat = model.predict(X)

    r2 = float(r2_score(y, yhat))
    rmse = float(math.sqrt(mean_squared_error(y, yhat)))
    q2, cv_name = _compute_q2_cv(model, X, y)

    coef_orig, intercept_orig = _orig_unit_coeffs(model)
    decimals = int(os.environ.get("FORMULA_DECIMALS", "2"))
    equation_pretty = _equation_pretty(features, coef_orig, intercept_orig, target, decimals=decimals)

    # Plot (Actual vs Predicted)
    x = np.asarray(y, dtype=float)
    ypred = np.asarray(yhat, dtype=float)
    mask = np.isfinite(x) & np.isfinite(ypred)
    x = x[mask]
    ypred = ypred[mask]

    if len(x) < 2:
        raise ValueError("Not enough valid points to plot.")

    m_line, b_line = np.polyfit(x, ypred, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    ys = m_line * xs + b_line

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ypred, mode="markers", name="Points",
                             hovertemplate="Actual=%{x}<br>Pred=%{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Fit line (Pred vs Actual)"))
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="y = x"))

    fig.update_layout(
        title=f"{category} | Actual vs Predicted | {title_suffix}",
        xaxis_title="Actual",
        yaxis_title="Predicted",
    )

    best_model_str = "[" + ", ".join(features) + "]"
    metrics_text = (
        f"Model: {best_model_str} | Q²={q2:.3f} | R²={r2:.3f}<br>"
        f"Regression Formula: {equation_pretty}<br>"
        f"RMSE={rmse:.3g} | CV={cv_name}"
    )
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=metrics_text,
        showarrow=False,
        align="left",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.85)",
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")

    # Sidecar JSON for quick programmatic use
    sidecar = {
        "category": category,
        "target": target,
        "features": features,
        "metrics": {"r2": r2, "q2": q2, "cv": cv_name, "rmse": rmse},
        "equation_pretty": equation_pretty,
        "plot_html": str(out_html.name),
    }
    out_html.with_suffix(".json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    return r2, q2, cv_name, equation_pretty


def run_all_regressions(base: Path, open_plots: bool = False) -> None:
    """Run all 4 cases and write *results/{case}/...*.

    What you'll get per case (inside results/<case>/):
      - <case>_features.csv
      - <case>_subset_search.csv
      - <case>_top_models_by_k.csv
      - <case>_regression_report.json
      - <case>_Regression_Plot.html (best)
      - <case>_top5_models.csv (summary)
      - top1_..._Regression_Plot.html ... top5_..._Regression_Plot.html
    """
    _ensure_import_local_modules(base)

    from extractor_regr import run_regression_from_mapping  # type: ignore

    # Control whether Plotly HTML auto-opens
    os.environ["OPEN_PLOTS"] = "1" if open_plots else "0"
    os.environ["ATOM_ONLY"] = "1"
    os.environ["PAIR_ONLY_AR"] = "1"
    os.environ["PAIR_ENGINEERED"] = "0"

    # Apply global env guards (all datasets)
    for k_env, v_env in GLOBAL_ENV.items():
        os.environ[k_env] = str(v_env)

    results_root = base / RESULTS_DIRNAME
    results_root.mkdir(parents=True, exist_ok=True)

    # Ask extractor_regr to evaluate exactly k=3/4/5 in one run (so we can rank across k).
    # If extractor_regr doesn't support list-form FORCE_K, it will safely ignore it.
    prev_force_k = os.environ.get("FORCE_K")
    os.environ["FORCE_K"] = ",".join(str(k) for k in K_LIST)

    overall_rows = []

    for case in CASES:
        category = case["category"]
        log_folder = base / _norm_rel(case["log_subdir"])
        mapping_csv = base / f"{category}_mapping.csv"
        xlsx_path = base / _norm_rel(case["xlsx"])
        target = case["target"]

        case_dir = results_root / category
        case_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = case_dir / category

        print("=" * 80)
        print(f"=== Running regressions for {category} ===")
        print(f"  mapping : {mapping_csv}")
        print(f"  log     : {log_folder}")
        print(f"  xlsx    : {xlsx_path}")
        print(f"  target  : {target}")
        print(f"  results : {case_dir}")
        case_t0 = time.time()

        # Apply per-category env overrides (restore after finishing this category)
        prev_env: Dict[str, str] = {}
        overrides = CASE_ENV_OVERRIDES.get(category, {})
        for k_env, v_env in overrides.items():
            if k_env in os.environ:
                prev_env[k_env] = os.environ[k_env]
            os.environ[k_env] = str(v_env)

        try:
            if not mapping_csv.exists():
                print(f"  [WARN] mapping CSV not found, skip regression: {mapping_csv}")
                continue
            if not xlsx_path.exists():
                print(f"  [WARN] kinetics xlsx not found, skip regression: {xlsx_path}")
                continue

            # One run per case: evaluate candidate models, write the usual outputs.
            run_regression_from_mapping(
                mapping_csv=str(mapping_csv),
                log_folder=str(log_folder),
                xlsx_path=str(xlsx_path),
                target=target,
                max_features=max(3, max(K_LIST)),
                output_prefix=str(out_prefix),
                category_name=category,
            )

            # Rank + record top N models and generate plots for each.
            top_models_csv = Path(f"{out_prefix}_top_models_by_k.csv")
            merged_csv = Path(f"{out_prefix}_features.csv")
            report_json = Path(f"{out_prefix}_regression_report.json")

            top_rows = _load_top_models(top_models_csv, TOP_N_MODELS)
            if not top_rows:
                print(f"  [WARN] No rows found in: {top_models_csv}")
                continue

            summary_rows = []
            for rank, row in enumerate(top_rows, start=1):
                k = int(row.get("k"))
                feats = str(row.get("features", "")).split("|")
                feats = [f for f in feats if f]

                plot_name = f"top{rank}_k{k}_Regression_Plot.html"
                plot_path = case_dir / plot_name

                r2, q2, cv_name, eq = _make_and_save_plot(
                    merged_csv=merged_csv,
                    target=target,
                    category=category,
                    features=feats,
                    out_html=plot_path,
                    title_suffix=f"Top {rank} (k={k})",
                )

                summary_rows.append(
                    {
                        "rank": rank,
                        "k": k,
                        "features": "|".join(feats),
                        "R2": r2,
                        "Q2": q2,
                        "CV": cv_name,
                        "equation": eq,
                        "plot_html": plot_name,
                    }
                )

            # Write a compact summary CSV (easy to paste into papers / tables).
            try:
                import pandas as pd

                pd.DataFrame(summary_rows).to_csv(case_dir / f"{category}_top{TOP_N_MODELS}_models.csv", index=False)
            except Exception as e:
                print(f"  [WARN] Failed to write top-N summary CSV: {e}")

            # Case-level timing + summary row
            case_elapsed = round(time.time() - case_t0, 6)
            case_model = overrides.get("REGRESSOR", os.environ.get("REGRESSOR", "ols"))
            summary = {
                "category": category,
                "target": target,
                "model": case_model,
                "elapsed_seconds_batch_runner": case_elapsed,
                "top_models_csv": str(top_models_csv),
                "regression_report_json": str(report_json),
            }
            if report_json.exists():
                try:
                    report = json.loads(report_json.read_text(encoding="utf-8"))
                    summary.update({
                        "best_r2": report.get("metrics", {}).get("r2"),
                        "best_q2": report.get("metrics", {}).get("q2"),
                        "best_rmse": report.get("metrics", {}).get("rmse"),
                        "best_features": "|".join(report.get("best_features", [])),
                        "total_case_seconds_pipeline": report.get("timing_seconds", {}).get("total_case_seconds"),
                    })
                except Exception as e:
                    print(f"  [WARN] Failed to read report JSON for summary: {e}")
            overall_rows.append(summary)
            Path(case_dir / f"{category}_timing_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        finally:
            # Restore per-category overrides
            for k_env in overrides.keys():
                if k_env in prev_env:
                    os.environ[k_env] = prev_env[k_env]
                else:
                    os.environ.pop(k_env, None)

        print()

    # Restore FORCE_K to avoid surprising callers
    if prev_force_k is None:
        os.environ.pop("FORCE_K", None)
    else:
        os.environ["FORCE_K"] = prev_force_k

    if overall_rows:
        try:
            import pandas as pd
            pd.DataFrame(overall_rows).to_csv(results_root / "all_cases_summary.csv", index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"[WARN] Failed to write all_cases_summary.csv: {e}")

    print("=== All regressions finished. ===")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE, help="Project root folder")
    parser.add_argument("--skip-mapping", action="store_true", help="Skip Step 1 mapping extraction")
    parser.add_argument("--skip-regression", action="store_true", help="Skip Step 2 regression/plotting")
    parser.add_argument("--open-plots", action="store_true", help="Auto-open Plotly HTML in browser")
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    print(f"[INFO] BASE = {base}")

    if not args.skip_mapping:
        run_mapping_extraction(base)

    if not args.skip_regression:
        run_all_regressions(base, open_plots=args.open_plots)

    print("\n>>> All done. <<<")


if __name__ == "__main__":
    main()
