# -*- coding: utf-8 -*-
"""
extractor_regr.py

What this script does
- Parse Gaussian log files -> numeric descriptors (SCF energy, GFE, dipole, HOMO/LUMO/gap, Mulliken charges).
- Join descriptors with kinetics data from an Excel sheet.
- Run subset-search linear regression for k = 3/4/5 features (or FORCE_K) and report:
    R2, Q2 (CV), RMSE, N points, and the regression formula.
- Write Plotly HTML:
    1) XY scatter (Actual vs Predicted) + fit line + y=x + metrics + pretty formula
    2) Coefficient bar chart (optional)

Speed knobs (environment variables)
- CV_MODE="kfold"  (fast) or "loocv" (slow, default)
- N_SPLITS="5"
- CAND_POOL="18"       (candidate feature pool size)
- MAX_COMBOS="250000"  (max combos to fully enumerate before sampling)
- SAMPLE_COMBOS="40000" (if combinations explode, randomly sample this many)
- PREFILTER_TOP="1500" (0 disables; >0 = 2-stage speedup: quick-fit all -> CV only top N)
- MIN_R2="0.75"  MIN_Q2="0.75"
- FORCE_K="3,4,5" (force only these ks)
- REGRESSOR="ols" or "ridge"  (RIDGE_ALPHA="1.0")
- VERBOSE="1"  PROGRESS_EVERY="500"
- FORMULA_DECIMALS="2"
- OPEN_PLOTS="1"

Azoarene note
- If the kinetics sheet has BOTH columns 'Ar1' and 'Ar2' (and you did not manually pass join_col),
  we auto-switch to PAIR MODE:
    build ar1_* and ar2_* descriptors, plus engineered features:
      sum_*, diff_*, absdiff_*, prod_*
  This fixes the classic "single join collisions" problem for bi-aryl datasets.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import random
import re
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception as e:
    raise ImportError("Plotly is required: pip install plotly") from e

try:
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
except Exception as e:
    raise ImportError("scikit-learn is required: pip install scikit-learn") from e


# -------------------- logging / progress --------------------

_VERBOSE = os.environ.get("VERBOSE", "1").strip().lower() not in ("0", "false", "no")
_PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "2000"))


def _log(msg: str) -> None:
    if _VERBOSE:
        print(msg, flush=True)


ATOM_INDEX_COLS_DEFAULT = ["a", "b", "c", "d", "e", "f", "g"]


# -------------------- regressor --------------------

def _get_model_name() -> str:
    return os.environ.get("REGRESSOR", "ols").strip().lower()


def _make_regressor():
    reg = _get_model_name()
    if reg == "ridge":
        alpha = float(os.environ.get("RIDGE_ALPHA", "1.0"))
        return Ridge(alpha=alpha)
    if reg in ("svm", "svr", "svr_rbf", "svm_rbf"):
        c = float(os.environ.get("SVR_C", "10.0"))
        epsilon = float(os.environ.get("SVR_EPSILON", "0.1"))
        gamma = os.environ.get("SVR_GAMMA", "scale").strip() or "scale"
        return SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma)
    if reg in ("svr_linear", "svm_linear"):
        c = float(os.environ.get("SVR_C", "10.0"))
        epsilon = float(os.environ.get("SVR_EPSILON", "0.1"))
        return SVR(kernel="linear", C=c, epsilon=epsilon)
    return LinearRegression()


def _is_linear_model(model: Pipeline) -> bool:
    lr = model.named_steps["lr"]
    return hasattr(lr, "coef_") and hasattr(lr, "intercept_")


# -------------------- log parsing --------------------

def _read_text(path) -> str:
    """Read text from a path-like (Path or str)."""
    path = Path(path)
    return path.read_text(errors="ignore")


def _find_last_float(pattern: str, text: str) -> Optional[float]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        return None
    return float(matches[-1].group(1))


def _find_last_line_values(pattern: str, text: str) -> Optional[List[float]]:
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    if not matches:
        return None
    line = matches[-1].group(1)
    vals: List[float] = []
    for tok in line.strip().split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals if vals else None


def _parse_mulliken_charges(text: str) -> Dict[int, float]:
    charges: Dict[int, float] = {}

    anchors = []
    for pat in [
        r"^\s*Mulliken charges:\s*$",
        r"^\s*Mulliken charges and spin densities:\s*$",
    ]:
        anchors.extend([m.start() for m in re.finditer(pat, text, flags=re.MULTILINE)])
    if not anchors:
        return charges

    start = max(anchors)
    chunk = text[start:].splitlines()

    row_re = re.compile(r"^\s*(\d+)\s+\w+\s+(-?\d+\.\d+)\s*$")
    for line in chunk:
        if "Sum of Mulliken charges" in line:
            break
        m = row_re.match(line)
        if m:
            idx = int(m.group(1))
            chg = float(m.group(2))
            charges[idx] = chg
    return charges


@dataclass
class LogFeatures:
    logfile: str
    scf_energy_hartree: Optional[float] = None
    gibbs_free_energy_hartree: Optional[float] = None
    enthalpy_hartree: Optional[float] = None
    dipole_total_debye: Optional[float] = None
    homo_hartree: Optional[float] = None
    lumo_hartree: Optional[float] = None
    gap_hartree: Optional[float] = None
    mulliken: Optional[Dict[int, float]] = None


def parse_gaussian_log(log_path: Path) -> LogFeatures:
    txt = _read_text(log_path)

    scf = _find_last_float(r"SCF Done:\s+E\([RU]?\w+\)\s*=\s*(-?\d+\.\d+)", txt)
    gfe = _find_last_float(r"Sum of electronic and thermal Free Energies=\s*(-?\d+\.\d+)", txt)
    enthalpy = _find_last_float(r"Sum of electronic and thermal Enthalpies=\s*(-?\d+\.\d+)", txt)
    dipole = _find_last_float(r"Dipole moment.*?Tot=\s*(-?\d+\.\d+)", txt)

    occ = _find_last_line_values(r"Alpha\s+occ\.\s+eigenvalues\s+--\s+(.+)$", txt)
    virt = _find_last_line_values(r"Alpha\s+virt\.\s+eigenvalues\s+--\s+(.+)$", txt)

    homo = occ[-1] if occ else None
    lumo = virt[0] if virt else None
    gap = (lumo - homo) if (homo is not None and lumo is not None) else None

    mulliken = _parse_mulliken_charges(txt)

    return LogFeatures(
        logfile=log_path.name,
        scf_energy_hartree=scf,
        gibbs_free_energy_hartree=gfe,
        enthalpy_hartree=enthalpy,
        dipole_total_debye=dipole,
        homo_hartree=homo,
        lumo_hartree=lumo,
        gap_hartree=gap,
        mulliken=mulliken if mulliken else None,
    )


# -------------------- join helpers --------------------

def _smart_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if float(x).is_integer():
            return str(int(x))
        return str(x)
    return str(x).strip()


def _to_log_key(val) -> str:
    s = _smart_str(val)
    if not s:
        return ""
    s = s.replace("\\", "/").split("/")[-1].strip().lower()
    for ext in (".log", ".out", ".txt"):
        if s.endswith(ext):
            s = s[: -len(ext)]
            break
    return s


def _infer_best_join_strategy(
    kin_df: pd.DataFrame,
    mapping_keys: List[str],
    max_pair_cols: int = 12,
) -> Tuple[str, Optional[str], Optional[str]]:
    mset = set([k for k in mapping_keys if k])

    def score_single(col: str) -> int:
        vals = kin_df[col].dropna().tolist()
        keys = [_to_log_key(v) for v in vals]
        return len(set(keys) & mset)

    cols = list(kin_df.columns)

    best_col = None
    best_score = -1
    for col in cols:
        sc = score_single(col)
        if sc > best_score:
            best_score = sc
            best_col = col

    if best_col is not None and best_score > 0:
        return ("single", best_col, None)

    cand_cols = cols[:max_pair_cols]

    def pair_key(a, b, sep: str) -> str:
        ka = _to_log_key(a)
        kb = _to_log_key(b)
        if not ka and not kb:
            return ""
        return f"{ka}{sep}{kb}".strip(sep)

    best_pair = None
    best_pair_score = -1
    best_sep = None
    seps = ["-", "_", ""]
    for i in range(len(cand_cols)):
        for j in range(i + 1, len(cand_cols)):
            c1, c2 = cand_cols[i], cand_cols[j]
            v1 = kin_df[c1].tolist()
            v2 = kin_df[c2].tolist()
            for sep in seps:
                keys = [pair_key(a, b, sep) for a, b in zip(v1, v2)]
                sc = len(set([k for k in keys if k]) & mset)
                if sc > best_pair_score:
                    best_pair_score = sc
                    best_pair = (c1, c2)
                    best_sep = sep

    if best_pair and best_pair_score > 0:
        return (f"pair:{best_sep}", best_pair[0], best_pair[1])

    raise ValueError("Cannot infer join key column(s) between xlsx and mapping.csv.")


# -------------------- numeric coercion / leakage guards --------------------

def _coerce_numeric_columns(df: pd.DataFrame, skip: Optional[set] = None, min_valid_ratio: float = 0.6) -> pd.DataFrame:
    """Convert object columns that look numeric into float.
    Only converts if enough values are convertible (>= min_valid_ratio).
    """
    skip = skip or set()
    for c in list(df.columns):
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if df[c].dtype == object:
            s = df[c].astype(str).str.strip()
            num = pd.to_numeric(s, errors="coerce")
            valid_ratio = float(num.notna().mean()) if len(num) else 0.0
            if valid_ratio >= min_valid_ratio and num.nunique(dropna=True) > 1:
                df[c] = num
    return df


def _is_leak_feature(col: str, target: str) -> bool:
    """Block obvious target leakage features (e.g., using 'rate (kobs)' to predict ln(kobs))."""
    if col == target:
        return False
    lc = col.lower()
    lt = target.lower()

    if "kobs" in lt and "kobs" in lc:
        return True
    if "ddg" in lt and "ddg" in lc:
        return True

    norm = re.sub(r"[^a-z0-9]+", "", lc)
    norm_t = re.sub(r"[^a-z0-9]+", "", lt)
    if norm == norm_t:
        return True

    return False


# -------------------- feature table from logs --------------------

def _build_feature_table(mapping_df: pd.DataFrame, log_folder: Path, atom_index_cols: List[str]) -> pd.DataFrame:
    rows = []
    total = int(mapping_df.shape[0])
    t0 = time.time()
    _log(f"  [STAGE] Parsing Gaussian logs -> features (n_logs={total})")
    for i, (_, r) in enumerate(mapping_df.iterrows(), start=1):
        logfile = str(r["logfile"]).strip()
        if _VERBOSE and (i == 1 or i == total or i % 10 == 0):
            dt = time.time() - t0
            _log(f"    - log {i}/{total}: {logfile} (elapsed {dt:.1f}s)")

        log_path = log_folder / logfile
        if not log_path.exists():
            if not logfile.lower().endswith(".log") and (log_folder / (logfile + ".log")).exists():
                log_path = log_folder / (logfile + ".log")
                logfile = logfile + ".log"
            else:
                raise FileNotFoundError(f"Log file not found: {log_path}")

        lf = parse_gaussian_log(log_path)

        feat = {
            "logfile": logfile,
            "logkey": _to_log_key(logfile),
            "scf_energy_hartree": lf.scf_energy_hartree,
            "gibbs_free_energy_hartree": lf.gibbs_free_energy_hartree,
            "enthalpy_hartree": lf.enthalpy_hartree,
            "dipole_total_debye": lf.dipole_total_debye,
            "homo_hartree": lf.homo_hartree,
            "lumo_hartree": lf.lumo_hartree,
            "gap_hartree": lf.gap_hartree,
        }

        # atomic charges
        if lf.mulliken:
            for col in atom_index_cols:
                if col in r and pd.notna(r[col]):
                    try:
                        idx = int(r[col])
                    except Exception:
                        idx = None
                    feat[f"q_{col}"] = lf.mulliken.get(idx) if idx is not None else np.nan
                else:
                    feat[f"q_{col}"] = np.nan
        else:
            for col in atom_index_cols:
                feat[f"q_{col}"] = np.nan

        qcols = [f"q_{c}" for c in atom_index_cols]
        qvals = pd.to_numeric(pd.Series([feat[c] for c in qcols]), errors="coerce")
        feat["q_mean"] = float(qvals.mean(skipna=True)) if qvals.notna().any() else np.nan
        feat["q_std"] = float(qvals.std(skipna=True)) if qvals.notna().any() else np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


# -------------------- model / metrics --------------------

def _compute_q2_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[float, str]:
    n = int(len(y))
    if n <= 2:
        return (float("nan"), "CV:NA")

    cv_mode = os.environ.get("CV_MODE", "loocv").strip().lower()
    if cv_mode == "kfold":
        n_splits = int(os.environ.get("N_SPLITS", "5"))
        n_splits = max(2, min(n_splits, n))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_name = f"{n_splits}-fold"
    else:
        cv = LeaveOneOut()
        cv_name = "LOOCV"

    y_cv_pred = cross_val_predict(model, X, y, cv=cv)
    q2 = float(r2_score(y, y_cv_pred))
    return q2, cv_name


def _corr_rank_features(df: pd.DataFrame, target: str) -> List[Tuple[str, float]]:
    y = pd.to_numeric(df[target], errors="coerce")
    ranked: List[Tuple[str, float]] = []
    for c in df.columns:
        if c == target:
            continue
        if _is_leak_feature(c, target):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() < 3 or x.nunique(dropna=True) <= 1:
            continue
        corr = x.corr(y)
        if pd.notna(corr):
            ranked.append((c, abs(float(corr))))
    ranked.sort(key=lambda t: t[1], reverse=True)
    return ranked


def _dedup_high_corr(df: pd.DataFrame, features: List[str], thr: float) -> List[str]:
    if len(features) <= 1:
        return features
    kept: List[str] = []
    for f in features:
        ok = True
        for g in kept:
            c = df[f].corr(df[g])
            if pd.notna(c) and abs(float(c)) >= thr:
                ok = False
                break
        if ok:
            kept.append(f)
    return kept


def _fit_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", _make_regressor()),
        ]
    ).fit(X, y)


def _orig_unit_coeffs(model: Pipeline) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not _is_linear_model(model):
        return None, None

    scaler: StandardScaler = model.named_steps["scaler"]
    lr = model.named_steps["lr"]

    coef_scaled = np.asarray(lr.coef_, dtype=float).reshape(-1)
    intercept_scaled = float(np.asarray(lr.intercept_, dtype=float).reshape(-1)[0])

    scale = np.asarray(scaler.scale_, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)

    coef_orig = coef_scaled / scale
    intercept_orig = intercept_scaled - np.sum((mean / scale) * coef_scaled)
    return coef_orig, intercept_orig


def _equation_pretty(feature_names: List[str], coef_orig: Optional[np.ndarray], intercept_orig: Optional[float], y_name: str, decimals: int = 2) -> str:
    if coef_orig is None or intercept_orig is None:
        return f"{y_name} = f({', '.join(feature_names)}) [nonlinear model]"

    parts: List[str] = []
    for f, c in zip(feature_names, coef_orig):
        term = f"{abs(float(c)):.{decimals}f}({f})"
        if not parts:
            parts.append(f"-{term}" if float(c) < 0 else term)
        else:
            parts.append(("- " if float(c) < 0 else "+ ") + term)

    b = float(intercept_orig)
    parts.append(("- " if b < 0 else "+ ") + f"{abs(b):.{decimals}f}")
    return f"{y_name} = " + " ".join(parts)


# -------------------- subset search --------------------



def _compute_feature_importance(model: Pipeline, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=20, random_state=random_state, scoring="r2")
    imp = pd.DataFrame({
        "feature": list(X.columns),
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False, ignore_index=True)
    return imp


def _save_descriptor_correlations(df: pd.DataFrame, target: str, output_prefix: str, category_name: str) -> str:
    rows: List[Dict[str, float]] = []
    y = pd.to_numeric(df[target], errors="coerce")
    for c in df.columns:
        if c == target or _is_leak_feature(c, target):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        valid = pd.concat([x, y], axis=1).dropna()
        if valid.shape[0] < 3 or valid.iloc[:, 0].nunique() <= 1:
            continue
        corr = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
        rows.append({
            "feature": c,
            "pearson_corr": corr,
            "abs_pearson_corr": abs(corr),
            "n_valid": int(valid.shape[0]),
        })

    corr_df = pd.DataFrame(rows).sort_values("abs_pearson_corr", ascending=False, ignore_index=True)
    out_csv = output_prefix + "_descriptor_correlations.csv"
    corr_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if not corr_df.empty:
        top_df = corr_df.head(20).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_df["pearson_corr"], y=top_df["feature"], orientation="h", name="Pearson r"))
        fig.update_layout(
            title=f"{category_name} | Descriptor Correlations with {target} (Top 20)",
            xaxis_title="Pearson correlation",
            yaxis_title="Descriptor",
        )
        fig.write_html(output_prefix + "_descriptor_correlations.html", include_plotlyjs="cdn")
    return out_csv


def _save_feature_importance_outputs(model: Pipeline, X: pd.DataFrame, y: pd.Series, output_prefix: str, category_name: str) -> str:
    imp_df = _compute_feature_importance(model, X, y)
    out_csv = output_prefix + "_feature_importance.csv"
    imp_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    if not imp_df.empty:
        top_df = imp_df.head(20).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_df["importance_mean"],
            y=top_df["feature"],
            orientation="h",
            error_x=dict(type="data", array=top_df["importance_std"]),
            name="Permutation importance",
        ))
        fig.update_layout(
            title=f"{category_name} | Feature Importance (Permutation, Top 20)",
            xaxis_title="Mean importance (R² drop)",
            yaxis_title="Feature",
        )
        fig.write_html(output_prefix + "_feature_importance.html", include_plotlyjs="cdn")
    return out_csv

def _combo_iter(features: List[str], k: int, max_combos: int, sample_combos: int) -> Tuple[bool, List[Tuple[str, ...]]]:
    n = len(features)
    total = math.comb(n, k) if n >= k else 0
    if total <= max_combos:
        return False, list(itertools.combinations(features, k))

    rng = random.Random(42)
    sampled = set()
    attempts = 0
    cap = sample_combos * 50
    while len(sampled) < sample_combos and attempts < cap:
        comb = tuple(sorted(rng.sample(features, k)))
        sampled.add(comb)
        attempts += 1
    return True, sorted(sampled)


def _subset_search(
    df: pd.DataFrame,
    target: str,
    ks: List[int],
    min_r2: float,
    min_q2: float,
    cand_pool: int,
    max_combos: int,
    sample_combos: int,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    ranked = _corr_rank_features(df, target)
    candidates = [f for f, _ in ranked[:cand_pool]]

    dedup_thr = float(os.environ.get("DEDUP_CORR", "0.9995"))
    candidates = _dedup_high_corr(df, candidates, thr=dedup_thr)

    if len(candidates) < min(ks):
        raise ValueError(f"Not enough candidate features ({len(candidates)}) for ks={ks}. Increase CAND_POOL or lower DEDUP_CORR.")

    y = pd.to_numeric(df[target], errors="coerce")
    eval_rows: List[Dict] = []
    pass_rows: List[Dict] = []
    cv_name_used = None

    seen = 0
    best_so_far: Optional[Dict] = None
    t_search0 = time.time()

    prefilter_top = int(os.environ.get("PREFILTER_TOP", "0"))

    for k in ks:
        sampled, combos = _combo_iter(candidates, k, max_combos=max_combos, sample_combos=sample_combos)
        total_here = len(combos)

        # 2-stage speedup (optional)
        if prefilter_top > 0 and total_here > prefilter_top:
            _log(f"    - k={k}: prefiltering {total_here} combos -> top {prefilter_top} (no-CV)")
            quick = []
            for comb in combos:
                Xq = df[list(comb)]
                model_q = _fit_pipeline(Xq, y)
                yhat_q = model_q.predict(Xq)
                r2_q = float(r2_score(y, yhat_q))
                rmse_q = float(math.sqrt(mean_squared_error(y, yhat_q)))
                quick.append((r2_q, -rmse_q, comb))
            quick.sort(reverse=True)
            combos = [t[2] for t in quick[:prefilter_top]]
            total_here = len(combos)
            _log(f"      -> CV stage will evaluate {total_here} combos" + (" (SAMPLED)" if sampled else ""))

        _log(f"    - k={k}: evaluating {total_here} combos" + (" (SAMPLED)" if sampled else ""))

        for comb in combos:
            seen += 1
            if _VERBOSE and _PROGRESS_EVERY > 0 and (seen % _PROGRESS_EVERY == 0):
                dt = time.time() - t_search0
                if best_so_far is not None:
                    _log(
                        f"      progress: {seen} models | elapsed {dt:.1f}s | "
                        f"best_so_far Q2={best_so_far['Q2']:.3f} R2={best_so_far['R2']:.3f} "
                        f"RMSE={best_so_far['RMSE']:.3g} k={best_so_far['k']}"
                    )
                else:
                    _log(f"      progress: {seen} models | elapsed {dt:.1f}s")

            X = df[list(comb)]
            model = _fit_pipeline(X, y)
            yhat = model.predict(X)

            r2 = float(r2_score(y, yhat))
            rmse = float(math.sqrt(mean_squared_error(y, yhat)))
            q2, cv_name = _compute_q2_cv(model, X, y)
            cv_name_used = cv_name

            row = {"k": k, "features": "|".join(comb), "R2": r2, "Q2": q2, "RMSE": rmse, "sampled": int(sampled)}
            eval_rows.append(row)

            if best_so_far is None or (row["Q2"], row["R2"], -row["RMSE"]) > (best_so_far["Q2"], best_so_far["R2"], -best_so_far["RMSE"]):
                best_so_far = row

            if (r2 >= min_r2) and (q2 >= min_q2):
                pass_rows.append(row)

    eval_df = pd.DataFrame(eval_rows)
    if eval_df.empty:
        raise ValueError("No models evaluated in subset search (unexpected).")

    def sort_df(d: pd.DataFrame) -> pd.DataFrame:
        return d.sort_values(["Q2", "R2", "RMSE"], ascending=[False, False, True])

    if pass_rows:
        pass_df = sort_df(pd.DataFrame(pass_rows))
        best = pass_df.iloc[0].to_dict()
        best["status"] = "PASS"
    else:
        pass_df = pd.DataFrame(columns=eval_df.columns)
        best = sort_df(eval_df).iloc[0].to_dict()
        best["status"] = "BEST_NO_PASS"

    best["cv"] = cv_name_used
    best["candidate_pool"] = candidates
    best["min_r2"] = min_r2
    best["min_q2"] = min_q2
    best["n_evaluated"] = int(eval_df.shape[0])
    best["n_passing"] = int(len(pass_rows))

    return best, pass_df, eval_df


# -------------------- pair-feature builder (for Ar1+Ar2 datasets) --------------------

def _build_pair_features(
    kin: pd.DataFrame,
    feat_df: pd.DataFrame,
    target: str,
    col1: str = "Ar1",
    col2: str = "Ar2",
) -> Tuple[pd.DataFrame, str]:
    if col1 not in kin.columns or col2 not in kin.columns:
        raise ValueError(f"Pair mode requested but columns not found: {col1}, {col2}")

    base_feat_cols = [c for c in feat_df.columns if c not in ["logfile", "logkey"]]
    f1 = feat_df[["logkey"] + base_feat_cols].copy()
    f2 = feat_df[["logkey"] + base_feat_cols].copy()

    f1 = f1.rename(columns={c: f"ar1_{c}" for c in base_feat_cols})
    f2 = f2.rename(columns={c: f"ar2_{c}" for c in base_feat_cols})

    kin2 = kin.copy()
    kin2["logkey1"] = kin2[col1].map(_to_log_key)
    kin2["logkey2"] = kin2[col2].map(_to_log_key)

    m = kin2.merge(f1, left_on="logkey1", right_on="logkey", how="left").drop(columns=["logkey"])
    m = m.merge(f2, left_on="logkey2", right_on="logkey", how="left").drop(columns=["logkey"])

    # drop rows where one side is entirely missing (optional)
    if os.environ.get("PAIR_REQUIRE_BOTH", "1").strip().lower() in ("1", "true", "yes"):
        ar1_cols = [c for c in m.columns if c.startswith("ar1_")]
        ar2_cols = [c for c in m.columns if c.startswith("ar2_")]
        before = int(m.shape[0])
        m = m[~(m[ar1_cols].isna().all(axis=1) | m[ar2_cols].isna().all(axis=1))].copy()
        after = int(m.shape[0])
        if after != before:
            _log(f"  [INFO] PAIR_REQUIRE_BOTH dropped {before - after} rows due to missing Ar1/Ar2 descriptors")

    # engineered features
    for c in base_feat_cols:
        a = f"ar1_{c}"
        b = f"ar2_{c}"
        va = pd.to_numeric(m[a], errors="coerce")
        vb = pd.to_numeric(m[b], errors="coerce")
        m[f"sum_{c}"] = va + vb
        m[f"diff_{c}"] = va - vb
        m[f"absdiff_{c}"] = (va - vb).abs()
        m[f"prod_{c}"] = va * vb

    m[target] = pd.to_numeric(m[target], errors="coerce")
    return m, f"pair:{col1}+{col2}"


# -------------------- public entry --------------------

def run_regression_from_mapping(
    mapping_csv: str,
    log_folder: str,
    xlsx_path: str,
    target: str,
    max_features: int,
    output_prefix: str,
    category_name: str = "",
    atom_index_cols: Optional[List[str]] = None,
    join_col: Optional[str] = None,
    open_browser: Optional[bool] = None,
) -> None:
    atom_index_cols = atom_index_cols or ATOM_INDEX_COLS_DEFAULT
    if open_browser is None:
        open_browser = os.environ.get("OPEN_PLOTS", "0").strip() == "1"

    min_r2 = float(os.environ.get("MIN_R2", "0.75"))
    min_q2 = float(os.environ.get("MIN_Q2", "0.75"))
    cand_pool = int(os.environ.get("CAND_POOL", "25"))
    max_combos = int(os.environ.get("MAX_COMBOS", "250000"))
    sample_combos = int(os.environ.get("SAMPLE_COMBOS", "80000"))

    t_case0 = time.time()
    stage_times: Dict[str, float] = {}
    model_name = _get_model_name()

    _log(f"[START] {category_name} regression")
    _log(f"  mapping_csv : {mapping_csv}")
    _log(f"  log_folder  : {log_folder}")
    _log(f"  xlsx_path   : {xlsx_path}")
    _log(f"  target      : {target}")
    _log(f"  max_features: {max_features}")

    log_folder_p = Path(log_folder)

    t0 = time.time()
    # mapping
    mdf = pd.read_csv(mapping_csv)
    if "ok" in mdf.columns:
        mdf = mdf[mdf["ok"] == 1].copy()
    mdf["logfile"] = mdf["logfile"].astype(str).str.strip()
    mapping_keys = mdf["logfile"].map(_to_log_key).tolist()
    stage_times["load_mapping_seconds"] = round(time.time() - t0, 6)

    t0 = time.time()
    # kinetics
    kin = pd.read_excel(xlsx_path)
    kin = _coerce_numeric_columns(kin, skip={'Ar1','Ar2','Ar','Compound'})
    if target not in kin.columns:
        raise ValueError(f"Target column '{target}' not found in xlsx.")
    stage_times["load_kinetics_seconds"] = round(time.time() - t0, 6)

    t0 = time.time()
    # parse logs -> per-log descriptors
    feat_df = _build_feature_table(mdf, log_folder_p, atom_index_cols)
    stage_times["parse_logs_seconds"] = round(time.time() - t0, 6)

    # decide join mode
    join_mode = "unknown"
    use_pair = ("Ar1" in kin.columns and "Ar2" in kin.columns and join_col is None)

    t0 = time.time()
    if use_pair:
        merged, join_mode = _build_pair_features(kin, feat_df, target=target, col1="Ar1", col2="Ar2")
        merged = merged.dropna(subset=[target]).copy()
        merged = _coerce_numeric_columns(merged, skip={'logfile','logkey','logkey1','logkey2','Ar1','Ar2','Ar','Compound', target})
    else:
        kin2 = kin.copy()
        if join_col is not None:
            kin2["logkey"] = kin2[join_col].map(_to_log_key)
            join_mode = f"single:{join_col}"
        else:
            mode, c1, c2 = _infer_best_join_strategy(kin2, mapping_keys)
            if mode == "single":
                kin2["logkey"] = kin2[c1].map(_to_log_key)
                join_mode = f"single:{c1}"
            else:
                sep = mode.split(":", 1)[1]
                kin2["logkey"] = [
                    f"{_to_log_key(a)}{sep}{_to_log_key(b)}".strip(sep)
                    for a, b in zip(kin2[c1].tolist(), kin2[c2].tolist())
                ]
                join_mode = f"pair:{c1}{sep}{c2}"

        merged = feat_df.merge(kin2[["logkey", target]], on="logkey", how="inner")
        merged[target] = pd.to_numeric(merged[target], errors="coerce")
        merged = merged.dropna(subset=[target]).copy()
        merged = _coerce_numeric_columns(merged, skip={'logfile','logkey','logkey1','logkey2','Ar1','Ar2','Ar','Compound', target})

    stage_times["build_merged_dataset_seconds"] = round(time.time() - t0, 6)

    n = int(merged.shape[0])
    if n < 5:
        raise ValueError(f"Not enough rows for regression: n={n}")

    # save merged table for inspection
    out_features_csv = output_prefix + "_features.csv"
    merged.to_csv(out_features_csv, index=False, encoding="utf-8-sig")

    t0 = time.time()
    descriptor_corr_csv = _save_descriptor_correlations(merged, target, output_prefix, category_name)
    stage_times["descriptor_correlation_seconds"] = round(time.time() - t0, 6)

    # ks to search
    if max_features < 3:
        raise ValueError("max_features must be >= 3")
    ks = list(range(3, max_features + 1))
    force_k = os.environ.get("FORCE_K", "").strip()
    if force_k:
        ks = sorted({int(x.strip()) for x in force_k.split(",") if x.strip()})
        _log(f"  [INFO] FORCE_K active -> ks={ks}")

    _log(f"  [STAGE] Subset search k={ks} | MIN_R2={min_r2} MIN_Q2={min_q2} | CAND_POOL={cand_pool}")
    _log(f"          CV_MODE={os.environ.get('CV_MODE','loocv')} N_SPLITS={os.environ.get('N_SPLITS','5')} | MAX_COMBOS={max_combos} SAMPLE_COMBOS={sample_combos}")

    t0 = time.time()
    best, passing_df, eval_df = _subset_search(
        merged,
        target=target,
        ks=ks,
        min_r2=min_r2,
        min_q2=min_q2,
        cand_pool=cand_pool,
        max_combos=max_combos,
        sample_combos=sample_combos,
    )
    stage_times["subset_search_seconds"] = round(time.time() - t0, 6)

    # outputs
    out_search_csv = output_prefix + "_subset_search.csv"
    if not passing_df.empty:
        passing_df.to_csv(out_search_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["k", "features", "R2", "Q2", "RMSE", "sampled"]).to_csv(out_search_csv, index=False, encoding="utf-8-sig")

    # top models by k (even if nothing passes)
    top_per_k = int(os.environ.get("TOP_PER_K", "25"))
    top_rows = []
    for kk in sorted(eval_df["k"].unique().tolist()):
        sub = eval_df[eval_df["k"] == kk].copy()
        sub = sub.sort_values(["Q2", "R2", "RMSE"], ascending=[False, False, True]).head(top_per_k)
        top_rows.append(sub)
    top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    out_top_csv = output_prefix + "_top_models_by_k.csv"
    top_df.to_csv(out_top_csv, index=False, encoding="utf-8-sig")

    # fit best model
    best_features = best["features"].split("|")
    X_best = merged[best_features]
    y = pd.to_numeric(merged[target], errors="coerce")

    t0 = time.time()
    model = _fit_pipeline(X_best, y)
    yhat = model.predict(X_best)
    stage_times["fit_best_model_seconds"] = round(time.time() - t0, 6)

    r2 = float(r2_score(y, yhat))
    rmse = float(math.sqrt(mean_squared_error(y, yhat)))
    
    q2, cv_name = _compute_q2_cv(model, X_best, y)

    # Optional diagnostics: per-point CV residuals
    if os.environ.get("DIAG_CV", "0").strip().lower() in ("1", "true", "yes", "y"):
        try:
            n_pts = int(len(y))
            cv_mode = os.environ.get("CV_MODE", "loocv").strip().lower()
            if cv_mode == "kfold":
                n_splits = int(os.environ.get("N_SPLITS", "5"))
                n_splits = max(2, min(n_splits, n_pts))
                cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            else:
                cv = LeaveOneOut()

            y_cv = cross_val_predict(model, X_best, y, cv=cv)

            diag_df = merged.reset_index(drop=True).copy()
            diag_df["_y"] = y.values
            diag_df["_y_cv_pred"] = y_cv
            diag_df["_cv_resid"] = diag_df["_y"] - diag_df["_y_cv_pred"]
            diag_df["_abs_cv_resid"] = diag_df["_cv_resid"].abs()

            keep_id_cols = [c for c in ["Compound", "Ar1", "Ar2", "logfile", "logkey"] if c in diag_df.columns]
            out_cols = keep_id_cols + ["_y", "_y_cv_pred", "_cv_resid", "_abs_cv_resid"]
            diag_df[out_cols].sort_values("_abs_cv_resid", ascending=False).to_csv(
                output_prefix + "_cv_residuals.csv", index=False, encoding="utf-8-sig"
            )

            if "Ar1" in diag_df.columns:
                g = diag_df.groupby("Ar1")["_abs_cv_resid"].mean().sort_values(ascending=False)
                g.to_csv(output_prefix + "_Ar1_mean_abs_cv_resid.csv", header=["mean_abs_cv_resid"], encoding="utf-8-sig")
        except Exception as e:
            _log(f"  [WARN] DIAG_CV failed: {e}")

    t0 = time.time()
    coef_orig, intercept_orig = _orig_unit_coeffs(model)
    decimals = int(os.environ.get("FORMULA_DECIMALS", "2"))
    formula_pretty = _equation_pretty(best_features, coef_orig, intercept_orig, target, decimals=decimals)
    feature_importance_csv = _save_feature_importance_outputs(model, X_best, y, output_prefix, category_name)
    stage_times["feature_importance_seconds"] = round(time.time() - t0, 6)

    # Plotly XY
    x = np.asarray(y, dtype=float)
    ypred = np.asarray(yhat, dtype=float)
    mask = np.isfinite(x) & np.isfinite(ypred)
    x = x[mask]
    ypred = ypred[mask]

    m_line, b_line = np.polyfit(x, ypred, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    ys = m_line * xs + b_line

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ypred, mode="markers", name="Points",
                             hovertemplate="Actual=%{x}<br>Pred=%{y}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Fit line (Pred vs Actual)"))
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="y = x"))

    best_list_str = ", ".join(best_features)
    fig.update_layout(
        title=f"{category_name} | Actual vs Predicted (k={len(best_features)}, n={n})",
        xaxis_title="Actual",
        yaxis_title="Predicted",
    )
    best_model_str = "[" + ", ".join(best_features) + "]"

    metrics_text = (
        f"Best model: {best_model_str} | Q²={q2:.3f} | R²={r2:.3f}<br>"
        f"Regression Formula: {formula_pretty}<br>"
        f"RMSE={rmse:.3g} | join={join_mode} | CV={cv_name}"
    )
    fig.add_annotation(
        x=0.02, y=0.98, xref="paper", yref="paper",
        text=metrics_text, showarrow=False, align="left",
        bordercolor="rgba(0,0,0,0.2)", borderwidth=1,
        bgcolor="rgba(255,255,255,0.85)"
    )

    t0 = time.time()
    out_plot = output_prefix + "_Regression_Plot.html"
    fig.write_html(out_plot, include_plotlyjs="cdn")

    # Coef / importance bar
    fig2 = go.Figure()
    if coef_orig is not None:
        coef_order = np.argsort(coef_orig)[::-1]
        feat_ordered = [best_features[i] for i in coef_order]
        coef_ordered = [float(coef_orig[i]) for i in coef_order]
        fig2.add_trace(go.Bar(x=feat_ordered, y=coef_ordered, name="Coef (orig units)"))
        fig2.update_layout(title=f"{category_name} Coefficients (original feature units)",
                           xaxis_title="Feature", yaxis_title="Coefficient")
    else:
        imp_df = pd.read_csv(feature_importance_csv)
        fig2.add_trace(go.Bar(x=imp_df["feature"], y=imp_df["importance_mean"], name="Permutation importance"))
        fig2.update_layout(title=f"{category_name} Feature Importance (nonlinear model)",
                           xaxis_title="Feature", yaxis_title="Permutation importance")
    out_coef_plot = output_prefix + "_Coef_Plot.html"
    fig2.write_html(out_coef_plot, include_plotlyjs="cdn")
    stage_times["plot_write_seconds"] = round(time.time() - t0, 6)

    stage_times["total_case_seconds"] = round(time.time() - t_case0, 6)

    # report json
    report = {
        "category": category_name,
        "target": target,
        "join_mode": join_mode,
        "n_points": n,
        "model": {"name": model_name},
        "metrics": {"r2": r2, "q2": q2, "q2_cv": cv_name, "rmse": rmse},
        "best_features": best_features,
        "equation_pretty": formula_pretty,
        "search": {
            "ks": ks,
            "min_r2": min_r2,
            "min_q2": min_q2,
            "cand_pool": cand_pool,
            "max_combos": max_combos,
            "sample_combos": sample_combos,
            "prefilter_top": int(os.environ.get("PREFILTER_TOP", "0")),
            "cv_mode": os.environ.get("CV_MODE", "loocv"),
            "n_splits": os.environ.get("N_SPLITS", "5"),
            "dedup_corr": float(os.environ.get("DEDUP_CORR", "0.9995")),
            "status": best["status"],
            "n_evaluated": best["n_evaluated"],
            "n_passing": best["n_passing"],
        },
        "outputs": {
            "features_csv": out_features_csv,
            "descriptor_correlations_csv": descriptor_corr_csv,
            "feature_importance_csv": feature_importance_csv,
            "subset_search_csv": out_search_csv,
            "top_models_by_k_csv": out_top_csv,
            "plot_html": out_plot,
            "coef_plot_html": out_coef_plot,
        },
        "timing_seconds": stage_times,
    }
    out_report = output_prefix + "_regression_report.json"
    Path(out_report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"  [OK] {category_name} subset-search regression | model={model_name} | status={best['status']} | "
        f"n={n} | k={len(best_features)} | R2={r2:.3f} | Q2({cv_name})={q2:.3f} | RMSE={rmse:.3g}"
    )
    print(f"       join_mode = {join_mode}")
    print(f"       best_features = {best_features}")
    print(f"       equation_pretty = {formula_pretty}")
    print(f"       descriptor_correlations_csv = {descriptor_corr_csv}")
    print(f"       feature_importance_csv = {feature_importance_csv}")
    print(f"       passing_models_csv = {out_search_csv}")
    print(f"       top_models_by_k_csv = {out_top_csv}")
    print(f"       - {out_plot}")
    print(f"       - {out_coef_plot}")
    print(f"       - {out_report}")
    print(f"       timing_seconds = {stage_times}")

    if open_browser:
        try:
            webbrowser.open(f"file:///{out_plot}")
            webbrowser.open(f"file:///{out_coef_plot}")
        except Exception:
            pass
