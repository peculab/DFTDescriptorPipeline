from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

BASE_DEFAULT = ""
RESULTS_DIRNAME = "results"
AZO_CATEGORY = "azoarene"
AZO_LOG_SUBDIR = r"azoarene\logfiles"
AZO_XLSX = r"azoarene\azoarene_data.xlsx"
AZO_TARGET = "ln(kobs)"

GLOBAL_ENV = {
    "PAIR_ONLY_AR": "1",
    "PAIR_ENGINEERED": "0",
}

# Only vary the Azoarene override block.
EXPERIMENTS: List[Dict[str, str]] = [
    #{
    #    "name": "ols_baseline",
    #    "GAUSS_LUMO_MODE": "lastblock_first",
    #    "REGRESSOR": "ols",
    #},
    #{
    #    "name": "ridge_a0.3",
    #    "GAUSS_LUMO_MODE": "lastblock_first",
    #    "REGRESSOR": "ridge",
    #    "RIDGE_ALPHA": "0.3",
    #},
    #{
    #    "name": "ridge_a1.0",
    #    "GAUSS_LUMO_MODE": "lastblock_first",
    #    "REGRESSOR": "ridge",
    #    "RIDGE_ALPHA": "1.0",
    #},
    #{
    #    "name": "svr_linear_c1_eps01",
    #    "GAUSS_LUMO_MODE": "lastblock_first",
    #    "REGRESSOR": "svr_linear",
    #    "SVR_C": "1.0",
    #    "SVR_EPSILON": "0.1",
    #},
    #{
    #    "name": "svr_linear_c10_eps01",
    #    "GAUSS_LUMO_MODE": "lastblock_first",
    #    "REGRESSOR": "svr_linear",
    #    "SVR_C": "10.0",
    #    "SVR_EPSILON": "0.1",
    #},
    #{
    #    "name": "svr_rbf_c3_eps02_scale",
    #    "GAUSS_LUMO_MODE": "lastblock_first",
    #    "REGRESSOR": "svr_rbf",
    #    "SVR_C": "3.0",
    #    "SVR_EPSILON": "0.2",
    #    "SVR_GAMMA": "scale",
    #},
    #{
    #    "name": "svr_rbf_c10_eps01_scale",
    #    "GAUSS_LUMO_MODE": "lastblock_first",
    #    "REGRESSOR": "svr_rbf",
    #    "SVR_C": "10.0",
    #    "SVR_EPSILON": "0.1",
    #    "SVR_GAMMA": "scale",
    #},
    {
        "name": "svr_rbf_c10_eps01_g003",
        "GAUSS_LUMO_MODE": "lastblock_first",
        "REGRESSOR": "svr_rbf",
        "SVR_C": "10.0",
        "SVR_EPSILON": "0.1",
        "SVR_GAMMA": "0.03",
    },
    {
        "name": "svr_rbf_c10_eps01_g03",
        "GAUSS_LUMO_MODE": "lastblock_first",
        "REGRESSOR": "svr_rbf",
        "SVR_C": "10.0",
        "SVR_EPSILON": "0.1",
        "SVR_GAMMA": "0.3",
    },
    {
        "name": "svr_rbf_c30_eps005_scale",
        "GAUSS_LUMO_MODE": "lastblock_first",
        "REGRESSOR": "svr_rbf",
        "SVR_C": "30.0",
        "SVR_EPSILON": "0.05",
        "SVR_GAMMA": "scale",
    },
]


def _norm_rel(path_str: str) -> Path:
    s = str(path_str).strip().replace("\\", "/")
    return Path(*[p for p in s.split("/") if p])


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _ensure_import_local_modules(base: Path) -> None:
    for p in [str(base), str(_script_dir())]:
        if p not in sys.path:
            sys.path.insert(0, p)


def _ensure_mapping(base: Path) -> Path:
    mapping_csv = base / f"{AZO_CATEGORY}_mapping.csv"
    if mapping_csv.exists():
        return mapping_csv

    extractor = base / "extract_abcefg_from_logs.py"
    if not extractor.exists():
        extractor = _script_dir() / "extract_abcefg_from_logs.py"
    if not extractor.exists():
        raise FileNotFoundError("Cannot find extract_abcefg_from_logs.py")

    logdir = base / _norm_rel(AZO_LOG_SUBDIR)
    with open(mapping_csv, "w", encoding="utf-8") as f:
        subprocess.run(
            [sys.executable, str(extractor), str(logdir), AZO_CATEGORY],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return mapping_csv


def _load_report(report_json: Path) -> Dict:
    if not report_json.exists():
        return {}
    try:
        return json.loads(report_json.read_text(encoding="utf-8"))
    except Exception:
        return {}


def run_experiments(base: Path, open_plots: bool = False) -> Path:
    _ensure_import_local_modules(base)
    from extractor_regr_paper_descriptors import run_regression_from_mapping  # type: ignore

    mapping_csv = _ensure_mapping(base)
    log_folder = base / _norm_rel(AZO_LOG_SUBDIR)
    xlsx_path = base / _norm_rel(AZO_XLSX)

    results_root = base / RESULTS_DIRNAME / "azoarene_experiments"
    results_root.mkdir(parents=True, exist_ok=True)

    prev_force_k = os.environ.get("FORCE_K")
    os.environ["FORCE_K"] = "3,4,5"
    os.environ["OPEN_PLOTS"] = "1" if open_plots else "0"
    for k, v in GLOBAL_ENV.items():
        os.environ[k] = str(v)

    summary_rows = []

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        case_dir = results_root / exp_name
        case_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = case_dir / AZO_CATEGORY

        print("=" * 88)
        print(f"=== Azoarene experiment: {exp_name} ===")

        prev_env = {k: os.environ.get(k) for k in exp.keys() if k != "name"}
        t0 = time.time()
        try:
            for k, v in exp.items():
                if k == "name":
                    continue
                os.environ[k] = str(v)

            run_regression_from_mapping(
                mapping_csv=str(mapping_csv),
                log_folder=str(log_folder),
                xlsx_path=str(xlsx_path),
                target=AZO_TARGET,
                max_features=5,
                output_prefix=str(out_prefix),
                category_name=f"{AZO_CATEGORY}_{exp_name}",
                open_browser=open_plots,
            )

            report_json = case_dir / f"{AZO_CATEGORY}_regression_report.json"
            top_models_csv = case_dir / f"{AZO_CATEGORY}_top_models_by_k.csv"
            report = _load_report(report_json)
            metrics = report.get("metrics", {})
            best_features = "|".join(report.get("best_features", []))

            row = {
                "experiment": exp_name,
                "regressor": exp.get("REGRESSOR", "ols"),
                "GAUSS_LUMO_MODE": exp.get("GAUSS_LUMO_MODE", ""),
                "SVR_C": exp.get("SVR_C", ""),
                "SVR_EPSILON": exp.get("SVR_EPSILON", ""),
                "SVR_GAMMA": exp.get("SVR_GAMMA", ""),
                "RIDGE_ALPHA": exp.get("RIDGE_ALPHA", ""),
                "best_r2": metrics.get("r2"),
                "best_q2": metrics.get("q2"),
                "best_rmse": metrics.get("rmse"),
                "cv": metrics.get("cv"),
                "best_features": best_features,
                "equation_pretty": report.get("equation_pretty", ""),
                "elapsed_seconds": round(time.time() - t0, 6),
                "report_json": str(report_json),
                "top_models_csv": str(top_models_csv),
            }
            summary_rows.append(row)
        finally:
            for k, old in prev_env.items():
                if old is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old

    if prev_force_k is None:
        os.environ.pop("FORCE_K", None)
    else:
        os.environ["FORCE_K"] = prev_force_k

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["best_q2", "best_r2", "best_rmse"], ascending=[False, False, True])
    summary_csv = results_root / "azoarene_experiment_summary.csv"
    summary_json = results_root / "azoarene_experiment_summary.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    summary_json.write_text(summary_df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Finished Azoarene experiment sweep ===")
    print(f"Summary CSV : {summary_csv}")
    print(f"Summary JSON: {summary_json}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=BASE_DEFAULT, help="Project root folder")
    parser.add_argument("--open-plots", action="store_true", help="Open generated plot HTML files")
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    print(f"[INFO] BASE = {base}")
    run_experiments(base, open_plots=args.open_plots)


if __name__ == "__main__":
    main()
