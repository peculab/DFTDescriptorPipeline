from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

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
CASE_ENV_OVERRIDES = {
    "azoarene": {
        "GAUSS_LUMO_MODE": "lastblock_first",  # more robust LUMO when virt blocks split
        "REGRESSOR": "ridge",
        "RIDGE_ALPHA": "10.0",
        # "OUTLIER_DROP_TOP": "2",
    }
}

# We want plots for k = 3/4/5
K_LIST = [3, 4, 5]


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


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

    for case in CASES:
        category = case["category"]
        logdir = base / case["log_subdir"]
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


def run_all_regressions(base: Path, open_plots: bool = False) -> None:
    """
    For each category:
      - Run regression + Plotly plot for k = 3/4/5 features.
      - Output files:
          {base}/{category}_k{K}_Regression_Plot.html
          {base}/{category}_k{K}_features.csv
          {base}/{category}_k{K}_subset_search.csv
          {base}/{category}_k{K}_top_models_by_k.csv
          {base}/{category}_k{K}_regression_report.json
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

    for case in CASES:
        category = case["category"]
        log_folder = base / case["log_subdir"]
        mapping_csv = base / f"{category}_mapping.csv"
        xlsx_path = base / case["xlsx"]
        target = case["target"]

        print("=" * 80)
        print(f"=== Running regressions for {category} ===")
        print(f"  mapping : {mapping_csv}")
        print(f"  log     : {log_folder}")
        print(f"  xlsx    : {xlsx_path}")
        print(f"  target  : {target}")

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

            for k in K_LIST:
                print("-" * 80)
                print(f"  -> k = {k} features")

                prev_force_k = os.environ.get("FORCE_K")
                os.environ["FORCE_K"] = str(k)

                out_prefix = base / f"{category}_k{k}"
                try:
                    run_regression_from_mapping(
                        mapping_csv=str(mapping_csv),
                        log_folder=str(log_folder),
                        xlsx_path=str(xlsx_path),
                        target=target,
                        max_features=max(3, k),
                        output_prefix=str(out_prefix),
                        category_name=category,
                    )
                    print(f"     plot : {out_prefix}_Regression_Plot.html")
                finally:
                    if prev_force_k is None:
                        os.environ.pop("FORCE_K", None)
                    else:
                        os.environ["FORCE_K"] = prev_force_k

        finally:
            # Restore per-category overrides
            for k_env in overrides.keys():
                if k_env in prev_env:
                    os.environ[k_env] = prev_env[k_env]
                else:
                    os.environ.pop(k_env, None)

        print()

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
