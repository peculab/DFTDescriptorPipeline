# -*- coding: utf-8 -*-
"""
extractor_regr_paper_descriptors.py

53-feature schema version:
- 40 base features
- +13 paper-friendly aliases (Ar_*)

This file supports:
1) build_features_from_mapping(...)
2) run_regression_from_features_csv(...)
3) run_regression_from_mapping(..., features_only=True/False)
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except Exception as e:
    raise ImportError("Plotly is required: pip install plotly") from e

try:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
except Exception as e:
    raise ImportError("scikit-learn is required: pip install scikit-learn") from e

try:
    from morfeus import Sterimol  # type: ignore
    _HAS_MORFEUS = True
except Exception:
    _HAS_MORFEUS = False


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
        gamma_raw = os.environ.get("SVR_GAMMA", "scale").strip() or "scale"
        if gamma_raw.lower() in {"scale", "auto"}:
            gamma = gamma_raw.lower()
        else:
            gamma = float(gamma_raw)
        return SVR(kernel="rbf", C=c, epsilon=epsilon, gamma=gamma)
    if reg in ("svr_linear", "svm_linear"):
        c = float(os.environ.get("SVR_C", "10.0"))
        epsilon = float(os.environ.get("SVR_EPSILON", "0.1"))
        return SVR(kernel="linear", C=c, epsilon=epsilon)
    return LinearRegression()


def _is_linear_model(model: Pipeline) -> bool:
    lr = model.named_steps["lr"]
    return hasattr(lr, "coef_") and hasattr(lr, "intercept_")


# -------------------- text parsing helpers --------------------

def _read_text(path) -> str:
    path = Path(path)
    return path.read_text(errors="ignore")


def _find_last_float(pattern: str, text: str) -> Optional[float]:
    vals = re.findall(pattern, text, flags=re.MULTILINE)
    if not vals:
        return None
    try:
        return float(vals[-1])
    except Exception:
        try:
            return float(vals[-1][-1])
        except Exception:
            return None


def _find_last_line(pattern: str, text: str) -> Optional[str]:
    lines = re.findall(pattern, text, flags=re.MULTILINE)
    return lines[-1] if lines else None


def _floats_from_line(line: str) -> List[float]:
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
    out = []
    for x in nums:
        try:
            out.append(float(x))
        except Exception:
            pass
    return out


def _find_last_block(header_regex: str, text: str, span: int = 20000) -> str:
    ms = list(re.finditer(header_regex, text, flags=re.MULTILINE))
    if not ms:
        return ""
    start = ms[-1].start()
    return text[start:start + span]


# -------------------- Gaussian parsers --------------------

def _parse_mulliken_charges(text: str) -> Dict[int, float]:
    block_starts = list(re.finditer(r"^\s*Mulliken charges(?: and spin densities)?:\s*$", text, flags=re.MULTILINE))
    if not block_starts:
        return {}
    start = block_starts[-1].end()

    tail = text[start:]
    end_match = re.search(r"^\s*Sum of Mulliken charges", tail, flags=re.MULTILINE)
    block = tail[: end_match.start()] if end_match else tail[:20000]

    charges = {}
    row_re = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+([-+]?\d*\.\d+|[-+]?\d+)\s*$", re.MULTILINE)
    for m in row_re.finditer(block):
        idx = int(m.group(1))
        q = float(m.group(3))
        charges[idx] = q
    return charges


def _parse_exact_polarizability(text: str) -> Optional[float]:
    block = _find_last_block(r"^\s*Exact polarizability:", text, span=1200)
    if not block:
        return None

    m = re.search(
        r"Isotropic\s*=\s*([-+]?\d+(?:\.\d+)?(?:[DEde][-+]?\d+)?)",
        block,
        flags=re.IGNORECASE,
    )
    if m:
        try:
            return float(m.group(1).replace("D", "E").replace("d", "e"))
        except Exception:
            pass

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[DEde][-+]?\d+)?", block)
    if nums:
        try:
            return float(nums[0].replace("D", "E").replace("d", "e"))
        except Exception:
            return None
    return None


def _parse_npa_charges(text: str) -> Dict[int, float]:
    """
    Parse NPA charges from Gaussian/NBO output.
    Expected row format:
        C    1   -0.18103   1.99900 ...
    """
    charges: Dict[int, float] = {}

    block = _find_last_block(r"^\s*Summary of Natural Population Analysis:\s*$", text, span=20000)
    if not block:
        return charges

    row_re = re.compile(
        r"^\s*([A-Za-z]{1,2})\s+(\d+)\s+(-?\d+\.\d+)\b",
        re.MULTILINE
    )

    for m in row_re.finditer(block):
        try:
            atom_idx = int(m.group(2))
            charge = float(m.group(3))
            charges[atom_idx] = charge
        except Exception:
            pass

    return charges


def _parse_standard_orientation_coords(text: str) -> Dict[int, Tuple[str, float, float, float]]:
    starts = list(re.finditer(r"^\s*Standard orientation:\s*$", text, flags=re.MULTILINE))
    if not starts:
        return {}

    start = starts[-1].end()
    tail = text[start:]
    lines = tail.splitlines()

    rows_started = False
    coords: Dict[int, Tuple[str, float, float, float]] = {}
    atomic_num_to_symbol = {
        1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 14: "Si", 15: "P",
        16: "S", 17: "Cl", 35: "Br", 53: "I"
    }

    dash_count = 0
    for line in lines:
        if re.match(r"^\s*-{5,}\s*$", line):
            dash_count += 1
            if dash_count >= 2 and not rows_started:
                rows_started = True
                continue
            elif dash_count >= 3 and rows_started:
                break

        if not rows_started:
            continue

        m = re.match(
            r"^\s*(\d+)\s+(\d+)\s+\d+\s+"
            r"([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s*$",
            line
        )
        if m:
            center_idx = int(m.group(1))
            atomic_num = int(m.group(2))
            x = float(m.group(3))
            y = float(m.group(4))
            z = float(m.group(5))
            sym = atomic_num_to_symbol.get(atomic_num, str(atomic_num))
            coords[center_idx] = (sym, x, y, z)

    return coords


def _distance(coords: Dict[int, Tuple[str, float, float, float]], i: int, j: int) -> Optional[float]:
    if i not in coords or j not in coords:
        return None
    _, xi, yi, zi = coords[i]
    _, xj, yj, zj = coords[j]
    return float(((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2) ** 0.5)


def _parse_ir_c_o_band(text: str, lo: float = 1500.0, hi: float = 1900.0) -> Tuple[Optional[float], Optional[float]]:
    freq_matches = list(re.finditer(r"Frequencies --\s+([^\n]+)", text))
    inten_matches = list(re.finditer(r"IR Inten\s+--\s+([^\n]+)", text))

    pairs: List[Tuple[float, float]] = []
    for fm, im in zip(freq_matches, inten_matches):
        fvals = _floats_from_line(fm.group(1))
        ivals = _floats_from_line(im.group(1))
        for f, inten in zip(fvals, ivals):
            if lo <= f <= hi:
                pairs.append((float(f), float(inten)))

    if not pairs:
        return None, None

    f_best, i_best = max(pairs, key=lambda t: t[1])
    return f_best, i_best


def _parse_nbo_bond_summary(text: str) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    occ: Dict[Tuple[int, int], float] = {}
    ene: Dict[Tuple[int, int], float] = {}

    block = _find_last_block(r"Natural Bond Orbitals \(Summary\)", text, span=50000)
    if not block:
        block = _find_last_block(r"NATURAL BOND ORBITALS \(Summary\)", text, span=50000)
    if not block:
        return occ, ene

    patt = re.compile(
        r"\(\s*([0-9]+)\)\s*([A-Za-z]{1,2})\s+([0-9]+)\s*-\s*\(\s*([0-9]+)\)\s*([A-Za-z]{1,2})\s+([0-9]+).*?"
        r"Occ\.\s*=\s*([-+]?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE
    )

    for m in patt.finditer(block):
        i = int(m.group(3))
        j = int(m.group(6))
        val = float(m.group(7))
        key = tuple(sorted((i, j)))
        occ[key] = val

    pert_block = _find_last_block(r"SECOND ORDER PERTURBATION THEORY ANALYSIS OF FOCK MATRIX IN NBO BASIS", text, span=80000)
    if pert_block:
        energy_line = re.compile(
            r"\(\s*([0-9]+)\)\s*([A-Za-z]{1,2})\s*([0-9]+).*?"
            r"\(\s*([0-9]+)\)\s*([A-Za-z]{1,2})\s*([0-9]+).*?"
            r"([-+]?\d+(?:\.\d+)?)\s*$",
            flags=re.MULTILINE
        )
        for m in energy_line.finditer(pert_block):
            i = int(m.group(3))
            j = int(m.group(6))
            try:
                e2 = float(m.group(7))
            except Exception:
                continue
            key = tuple(sorted((i, j)))
            if key not in ene:
                ene[key] = e2

    return occ, ene


def _sterimol_fallback(
    coords: Dict[int, Tuple[str, float, float, float]],
    atom1: int,
    atom2: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if atom1 not in coords or atom2 not in coords:
        return None, None, None

    _, x1, y1, z1 = coords[atom1]
    _, x2, y2, z2 = coords[atom2]
    axis = np.array([x2 - x1, y2 - y1, z2 - z1], dtype=float)
    norm = np.linalg.norm(axis)
    if norm == 0:
        return None, None, None
    u = axis / norm
    origin = np.array([x1, y1, z1], dtype=float)

    proj = []
    rad = []
    for idx, (_, x, y, z) in coords.items():
        if idx == atom1:
            continue
        p = np.array([x, y, z], dtype=float) - origin
        t = float(np.dot(p, u))
        perp = p - t * u
        r = float(np.linalg.norm(perp))
        proj.append(t)
        rad.append(r)

    if not proj or not rad:
        return None, None, None

    L = max(proj)
    B1 = min(rad)
    B5 = max(rad)
    return float(L), float(B1), float(B5)


def _compute_sterimol(
    coords: Dict[int, Tuple[str, float, float, float]],
    atom1: int,
    atom2: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if atom1 not in coords or atom2 not in coords:
        return None, None, None

    if _HAS_MORFEUS:
        try:
            elements = []
            xyz = []
            max_idx = max(coords.keys())
            for idx in range(1, max_idx + 1):
                if idx not in coords:
                    continue
                sym, x, y, z = coords[idx]
                elements.append(sym)
                xyz.append([x, y, z])
            xyz = np.array(xyz, dtype=float)
            st = Sterimol(elements, xyz, atom1, atom2)
            return float(st.L_value), float(st.B_1_value), float(st.B_5_value)
        except Exception:
            pass

    return _sterimol_fallback(coords, atom1, atom2)


# -------------------- parsed feature object --------------------

@dataclass
class LogFeaturesFull:
    logfile: str
    scf_energy_hartree: Optional[float]
    gibbs_free_energy_hartree: Optional[float]
    enthalpy_hartree: Optional[float]
    dipole_total_debye: Optional[float]
    homo_hartree: Optional[float]
    lumo_hartree: Optional[float]
    gap_hartree: Optional[float]
    mulliken: Dict[int, float]

    isotropic_polarizability_au: Optional[float]
    npa_charge: Dict[int, float]

    ir_c_o_freq_cm1: Optional[float]
    ir_c_o_intensity_km_mol: Optional[float]

    coords: Dict[int, Tuple[str, float, float, float]]

    nbo_bond_occ: Dict[Tuple[int, int], float]
    nbo_bond_energy: Dict[Tuple[int, int], float]


def parse_gaussian_log(log_path: Path) -> LogFeaturesFull:
    txt = _read_text(log_path)

    scf = _find_last_float(r"^\s*SCF Done:\s+E\([RU]?[A-Za-z0-9]+\)\s*=\s*([-+]?\d+\.\d+)", txt)
    gfe = _find_last_float(r"^\s*Sum of electronic and thermal Free Energies=\s*([-+]?\d+\.\d+)", txt)
    enth = _find_last_float(r"^\s*Sum of electronic and thermal Enthalpies=\s*([-+]?\d+\.\d+)", txt)

    dip = _find_last_float(r"^\s*Tot=\s*([-+]?\d+\.\d+)\s*$", txt)
    if dip is None:
        dip = _find_last_float(r"Dipole moment .*?Tot=\s*([-+]?\d+\.\d+)", txt)

    occ_line = _find_last_line(r"^\s*Alpha\s+occ\.\s+eigenvalues -- .*$", txt)
    virt_line = _find_last_line(r"^\s*Alpha\s+virt\.\s+eigenvalues -- .*$", txt)

    homo = None
    lumo = None
    if occ_line:
        vals = _floats_from_line(occ_line)
        if vals:
            homo = vals[-1]
    if virt_line:
        vals = _floats_from_line(virt_line)
        if vals:
            mode = os.environ.get("GAUSS_LUMO_MODE", "").strip().lower()
            lumo = vals[-1] if mode == "lastblock_last" else vals[0]

    gap = None
    if homo is not None and lumo is not None:
        gap = lumo - homo

    mull = _parse_mulliken_charges(txt)
    pol = _parse_exact_polarizability(txt)
    npa_charge = _parse_npa_charges(txt)
    ir_f, ir_i = _parse_ir_c_o_band(txt)
    coords = _parse_standard_orientation_coords(txt)
    nbo_occ, nbo_ene = _parse_nbo_bond_summary(txt)

    return LogFeaturesFull(
        logfile=log_path.name,
        scf_energy_hartree=scf,
        gibbs_free_energy_hartree=gfe,
        enthalpy_hartree=enth,
        dipole_total_debye=dip,
        homo_hartree=homo,
        lumo_hartree=lumo,
        gap_hartree=gap,
        mulliken=mull,
        isotropic_polarizability_au=pol,
        npa_charge=npa_charge,
        ir_c_o_freq_cm1=ir_f,
        ir_c_o_intensity_km_mol=ir_i,
        coords=coords,
        nbo_bond_occ=nbo_occ,
        nbo_bond_energy=nbo_ene,
    )


# -------------------- feature table building --------------------

def _safe_get_charge(charges: Dict[int, float], idx: Optional[float]) -> Optional[float]:
    if idx is None or (isinstance(idx, float) and np.isnan(idx)):
        return None
    try:
        return charges.get(int(idx), None)
    except Exception:
        return None


def _pair_lookup(d: Dict[Tuple[int, int], float], i: Optional[float], j: Optional[float]) -> Optional[float]:
    if i is None or j is None or (isinstance(i, float) and np.isnan(i)) or (isinstance(j, float) and np.isnan(j)):
        return None
    key = tuple(sorted((int(i), int(j))))
    return d.get(key, None)


def _normalize_join_key(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.log$", "", regex=True)
    return s


def _normalize_name_key(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\.log$", "", s, flags=re.IGNORECASE)
    return s


def _build_feature_table(mapping_df: pd.DataFrame, log_folder: Path, atom_index_cols: List[str]) -> pd.DataFrame:
    rows = []

    # auto-detect filename column
    filename_col = None
    preferred = ["filename", "logfile", "file", "name"]

    for c in preferred:
        if c in mapping_df.columns:
            filename_col = c
            break

    # fallback: choose the first column that looks like a log filename column
    if filename_col is None:
        for c in mapping_df.columns:
            vals = mapping_df[c].astype(str)
            if vals.str.contains(r"\.log$", case=False, regex=True).any():
                filename_col = c
                break

    if filename_col is None:
        raise ValueError(
            f"Cannot find a filename column in mapping CSV. "
            f"Available columns: {list(mapping_df.columns)}"
        )

    rows = []
    for _, r in mapping_df.iterrows():
        lf = parse_gaussian_log(log_folder / str(r[filename_col]))

        feat: Dict[str, Any] = {
            "logfile": lf.logfile,
            "logkey": _normalize_name_key(lf.logfile),

            "scf_energy_hartree": lf.scf_energy_hartree,
            "gibbs_free_energy_hartree": lf.gibbs_free_energy_hartree,
            "enthalpy_hartree": lf.enthalpy_hartree,
            "dipole_total_debye": lf.dipole_total_debye,
            "isotropic_polarizability_au": lf.isotropic_polarizability_au,
            "homo_hartree": lf.homo_hartree,
            "lumo_hartree": lf.lumo_hartree,
            "gap_hartree": lf.gap_hartree,

            "ir_c_o_freq_cm1": lf.ir_c_o_freq_cm1,
            "ir_c_o_intensity_km_mol": lf.ir_c_o_intensity_km_mol,
        }

        q_vals = []
        nbo_q_vals = []
        idx_map: Dict[str, Optional[float]] = {}
        for col in atom_index_cols:
            idx = r[col] if col in r.index else np.nan
            idx_map[col] = idx

            q = _safe_get_charge(lf.mulliken, idx)
            feat[f"q_{col}"] = q
            if q is not None:
                q_vals.append(q)

            nq = _safe_get_charge(lf.npa_charge, idx)
            feat[f"nbo_q_{col}"] = nq
            if nq is not None:
                nbo_q_vals.append(nq)

        feat["q_mean"] = float(np.mean(q_vals)) if q_vals else np.nan
        feat["q_std"] = float(np.std(q_vals, ddof=0)) if q_vals else np.nan

        feat["nbo_q_mean"] = float(np.mean(nbo_q_vals)) if nbo_q_vals else np.nan
        feat["nbo_q_std"] = float(np.std(nbo_q_vals, ddof=0)) if nbo_q_vals else np.nan

        c_idx = idx_map.get("c", np.nan)
        a_idx = idx_map.get("a", np.nan)
        d_idx = idx_map.get("d", np.nan)
        e_idx = idx_map.get("e", np.nan)

        feat["nbo_bond_occ_c_a"] = _pair_lookup(lf.nbo_bond_occ, c_idx, a_idx)
        feat["nbo_bond_occ_c_d"] = _pair_lookup(lf.nbo_bond_occ, c_idx, d_idx)
        feat["nbo_bond_occ_c_e"] = _pair_lookup(lf.nbo_bond_occ, c_idx, e_idx)

        feat["nbo_bond_energy_c_a"] = _pair_lookup(lf.nbo_bond_energy, c_idx, a_idx)
        feat["nbo_bond_energy_c_d"] = _pair_lookup(lf.nbo_bond_energy, c_idx, d_idx)
        feat["nbo_bond_energy_c_e"] = _pair_lookup(lf.nbo_bond_energy, c_idx, e_idx)

        feat["bond_length_c_a"] = _distance(lf.coords, int(c_idx), int(a_idx)) if pd.notna(c_idx) and pd.notna(a_idx) else np.nan
        feat["bond_length_c_d"] = _distance(lf.coords, int(c_idx), int(d_idx)) if pd.notna(c_idx) and pd.notna(d_idx) else np.nan
        feat["bond_length_c_e"] = _distance(lf.coords, int(c_idx), int(e_idx)) if pd.notna(c_idx) and pd.notna(e_idx) else np.nan

        ster_L, ster_B1, ster_B5 = _compute_sterimol(
            lf.coords,
            int(c_idx) if pd.notna(c_idx) else -1,
            int(e_idx) if pd.notna(e_idx) else -1,
        )
        feat["sterimol_L"] = ster_L
        feat["sterimol_B1"] = ster_B1
        feat["sterimol_B5"] = ster_B5

        # 13 paper-friendly aliases
        feat["Ar_NBO_C1"] = feat.get("nbo_q_c", np.nan)
        feat["Ar_NBO_C2"] = feat.get("nbo_q_e", np.nan)
        feat["Ar_NBO_OH"] = feat.get("nbo_q_a", np.nan)
        feat["Ar_NBO_CO"] = feat.get("nbo_q_d", np.nan)

        feat["Ar_v_CeqO"] = feat.get("ir_c_o_freq_cm1", np.nan)
        feat["Ar_I_CeqO"] = feat.get("ir_c_o_intensity_km_mol", np.nan)

        feat["Ar_Ster_L"] = feat.get("sterimol_L", np.nan)
        feat["Ar_Ster_B1"] = feat.get("sterimol_B1", np.nan)
        feat["Ar_Ster_B5"] = feat.get("sterimol_B5", np.nan)

        feat["Ar_dp"] = feat.get("dipole_total_debye", np.nan)
        feat["Ar_polar"] = feat.get("isotropic_polarizability_au", np.nan)
        feat["Ar_HOMO"] = feat.get("homo_hartree", np.nan)
        feat["Ar_LUMO"] = feat.get("lumo_hartree", np.nan)

        rows.append(feat)

    return pd.DataFrame(rows)

# -------------------- regression helpers --------------------

def _as_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _fit_pipeline(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", _make_regressor()),
        ]
    )
    model.fit(X, y)
    return model


def _get_cv():
    mode = os.environ.get("CV_MODE", "loocv").strip().lower()
    if mode == "kfold":
        n_splits = int(os.environ.get("N_SPLITS", "5"))
        shuffle = os.environ.get("KFOLD_SHUFFLE", "1").strip().lower() not in ("0", "false", "no")
        seed = int(os.environ.get("RANDOM_SEED", "42"))
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed), f"{n_splits}-fold"
    return LeaveOneOut(), "LOOCV"


def _compute_q2_cv(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[float, str]:
    cv, name = _get_cv()
    try:
        ycv = cross_val_predict(model, X, y, cv=cv)
        q2 = float(r2_score(y, ycv))
    except Exception:
        q2 = float("nan")
    return q2, name


def _orig_unit_coeffs(model: Pipeline) -> Tuple[np.ndarray, float]:
    if not _is_linear_model(model):
        return np.array([]), float("nan")

    sc = model.named_steps["sc"]
    lr = model.named_steps["lr"]

    coef_scaled = np.asarray(lr.coef_).reshape(-1)
    intercept_scaled = float(np.asarray(lr.intercept_).reshape(-1)[0]) if np.ndim(lr.intercept_) else float(lr.intercept_)

    scales = np.asarray(sc.scale_)
    means = np.asarray(sc.mean_)

    coef_orig = coef_scaled / scales
    intercept_orig = intercept_scaled - np.sum(coef_scaled * means / scales)
    return coef_orig, float(intercept_orig)


def _equation_pretty(features: List[str], coef_orig: np.ndarray, intercept_orig: float, target_name: str, decimals: int = 2) -> str:
    if coef_orig.size == 0 or not np.isfinite(intercept_orig):
        return f"{target_name} = <nonlinear model; no closed-form linear equation>"
    parts = [f"{target_name} = {intercept_orig:.{decimals}f}"]
    for f, c in zip(features, coef_orig):
        sign = "+" if c >= 0 else "-"
        parts.append(f" {sign} {abs(c):.{decimals}f}*{f}")
    return "".join(parts)


def _make_xy_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_html: Path,
    metrics_text: str,
) -> None:
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="Points",
            hovertemplate="Actual=%{x}<br>Pred=%{y}<extra></extra>",
        )
    )

    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 200)
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Fit line"))
        fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", name="y = x"))

    fig.update_layout(title=title, xaxis_title="Actual", yaxis_title="Predicted")
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

    if os.environ.get("OPEN_PLOTS", "0").strip().lower() not in ("0", "false", "no"):
        try:
            webbrowser.open_new_tab(out_html.resolve().as_uri())
        except Exception:
            pass


def _make_coef_bar_plot(features: List[str], coef_orig: np.ndarray, out_html: Path, title: str) -> None:
    if coef_orig.size == 0:
        return
    fig = go.Figure([go.Bar(x=features, y=coef_orig)])
    fig.update_layout(title=title, xaxis_title="Feature", yaxis_title="Coefficient (original units)")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="cdn")


def _candidate_pool_rank(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    ranks = {}
    for c in X.columns:
        xv = pd.to_numeric(X[c], errors="coerce")
        ok = xv.notna() & y.notna()
        if ok.sum() < 3:
            ranks[c] = 0.0
            continue
        try:
            corr = np.corrcoef(xv[ok].astype(float), y[ok].astype(float))[0, 1]
            ranks[c] = abs(float(corr)) if np.isfinite(corr) else 0.0
        except Exception:
            ranks[c] = 0.0
    return pd.Series(ranks).sort_values(ascending=False)


def _iter_feature_combos(features: List[str], k: int):
    max_combos = int(os.environ.get("MAX_COMBOS", "250000"))
    sample_combos = int(os.environ.get("SAMPLE_COMBOS", "40000"))
    all_combos = list(itertools.combinations(features, k))
    if len(all_combos) <= max_combos:
        return all_combos
    seed = int(os.environ.get("RANDOM_SEED", "42"))
    rng = random.Random(seed)
    idx = list(range(len(all_combos)))
    rng.shuffle(idx)
    idx = idx[:sample_combos]
    return [all_combos[i] for i in idx]


def _subset_search(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    max_features: int,
    category_name: str = "",
) -> Tuple[dict, pd.DataFrame]:
    cands_rank = _candidate_pool_rank(X, y)
    cand_pool = int(os.environ.get("CAND_POOL", "18"))
    cand_feats = list(cands_rank.head(min(cand_pool, len(cands_rank))).index)

    force_k = os.environ.get("FORCE_K", "").strip()
    if force_k:
        ks = [int(x) for x in force_k.split(",") if x.strip()]
    else:
        ks = list(range(1, max_features + 1))

    prefilter_top = int(os.environ.get("PREFILTER_TOP", "1500"))
    min_r2 = float(os.environ.get("MIN_R2", "0.75"))
    min_q2 = float(os.environ.get("MIN_Q2", "0.75"))

    tested_rows = []
    best = None
    best_key = (-np.inf, -np.inf, np.inf)

    for k in ks:
        combos = _iter_feature_combos(cand_feats, k)
        _log(f"[subset] k={k}, candidate features={len(cand_feats)}, combos={len(combos)}")

        quick_rows = []
        for i, comb in enumerate(combos, start=1):
            feats = list(comb)
            Xi = X[feats]
            model = _fit_pipeline(Xi, y)
            yhat = model.predict(Xi)
            r2 = float(r2_score(y, yhat))
            rmse = float(math.sqrt(mean_squared_error(y, yhat)))
            quick_rows.append((feats, r2, rmse))
            if _VERBOSE and _PROGRESS_EVERY > 0 and i % _PROGRESS_EVERY == 0:
                _log(f"  k={k}: quick-fit {i}/{len(combos)}")

        quick_rows.sort(key=lambda t: (t[1], -t[2]), reverse=True)
        quick_rows = quick_rows[:prefilter_top] if prefilter_top > 0 else quick_rows

        for feats, r2, rmse in quick_rows:
            Xi = X[feats]
            model = _fit_pipeline(Xi, y)
            q2, cv_name = _compute_q2_cv(model, Xi, y)

            coef_orig, intercept_orig = _orig_unit_coeffs(model)
            eq_pretty = _equation_pretty(
                feats, coef_orig, intercept_orig, target, decimals=int(os.environ.get("FORMULA_DECIMALS", "2"))
            )

            row = {
                "category": category_name,
                "target": target,
                "k": k,
                "features": "|".join(feats),
                "R2": r2,
                "Q2": q2,
                "RMSE": rmse,
                "CV": cv_name,
                "equation": eq_pretty,
            }
            tested_rows.append(row)

            key = (q2 if np.isfinite(q2) else -np.inf, r2, -rmse)
            if key > best_key and r2 >= min_r2 and (q2 >= min_q2 or not np.isfinite(q2)):
                best_key = key
                best = {
                    "best_features": feats,
                    "metrics": {"r2": r2, "q2": q2, "rmse": rmse, "cv": cv_name},
                    "equation_pretty": eq_pretty,
                }

    tested_df = pd.DataFrame(tested_rows)
    if best is None and not tested_df.empty:
        tested_df = tested_df.sort_values(["Q2", "R2", "RMSE"], ascending=[False, False, True])
        top = tested_df.iloc[0]
        best = {
            "best_features": str(top["features"]).split("|"),
            "metrics": {
                "r2": float(top["R2"]),
                "q2": float(top["Q2"]) if pd.notna(top["Q2"]) else float("nan"),
                "rmse": float(top["RMSE"]),
                "cv": str(top["CV"]),
            },
            "equation_pretty": str(top["equation"]),
        }

    return best if best is not None else {}, tested_df


# -------------------- merging mono/pair --------------------

def _merge_single(
    xlsx_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    target: str,
    join_col: Optional[str] = None,
) -> pd.DataFrame:
    df = xlsx_df.copy()

    # 1) try common column names first
    if join_col is None:
        candidates = [
            "compound", "substituent", "name", "filename", "logfile", "log",
            "aryl", "ar", "r", "x", "sub", "id"
        ]
        lower_map = {str(c).strip().lower(): c for c in df.columns}
        for c in candidates:
            if c.lower() in lower_map:
                join_col = lower_map[c.lower()]
                break

    # 2) fallback: find a column whose values overlap most with feat_df["logkey"]
    if join_col is None:
        feat_keys = set(feat_df["logkey"].astype(str).str.strip().str.lower())

        best_col = None
        best_score = -1

        for c in df.columns:
            vals = (
                df[c]
                .astype(str)
                .str.strip()
                .str.replace(r"\.log$", "", regex=True)
                .str.lower()
            )

            score = vals.isin(feat_keys).sum()
            if score > best_score:
                best_score = score
                best_col = c

        # require at least one actual overlap
        if best_col is not None and best_score > 0:
            join_col = best_col

    if join_col is None:
        raise ValueError(
            "Could not infer join column for mono-substituent dataset. "
            f"Available Excel columns: {list(df.columns)}"
        )

    print(f"[INFO] mono join column inferred: {join_col}")

    df["_join_key"] = (
        df[join_col]
        .astype(str)
        .str.strip()
        .str.replace(r"\.log$", "", regex=True)
    )

    feat_df = feat_df.copy()
    feat_df["_join_key"] = feat_df["logkey"].astype(str).str.strip()

    merged = df.merge(
        feat_df.drop(columns=["logfile", "logkey"]),
        on="_join_key",
        how="left"
    )

    if target not in merged.columns:
        raise ValueError(f"Target column '{target}' not found in Excel sheet.")

    return merged

def _build_pair_features(
    xlsx_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    if "Ar1" not in xlsx_df.columns or "Ar2" not in xlsx_df.columns:
        raise ValueError("Pair mode requires 'Ar1' and 'Ar2' columns in Excel sheet.")

    base = xlsx_df.copy()
    base["_join1"] = _normalize_join_key(base["Ar1"])
    base["_join2"] = _normalize_join_key(base["Ar2"])

    F = feat_df.copy()
    F["_join_key"] = F["logkey"]

    fcols = [c for c in F.columns if c not in ("logfile", "logkey", "_join_key")]

    left = F[["_join_key"] + fcols].copy()
    left = left.rename(columns={c: f"ar1_{c}" for c in fcols})
    right = F[["_join_key"] + fcols].copy()
    right = right.rename(columns={c: f"ar2_{c}" for c in fcols})

    m = base.merge(left, left_on="_join1", right_on="_join_key", how="left").drop(columns=["_join_key"])
    m = m.merge(right, left_on="_join2", right_on="_join_key", how="left").drop(columns=["_join_key"])

    # only add engineered columns if explicitly enabled
    if os.environ.get("PAIR_ENGINEERED", "0").strip().lower() not in ("0", "false", "no"):
        new_cols = {}
        for c in fcols:
            a = pd.to_numeric(m[f"ar1_{c}"], errors="coerce")
            b = pd.to_numeric(m[f"ar2_{c}"], errors="coerce")
            new_cols[f"sum_{c}"] = a + b
            new_cols[f"diff_{c}"] = a - b
            new_cols[f"absdiff_{c}"] = (a - b).abs()
            new_cols[f"prod_{c}"] = a * b
        m = pd.concat([m, pd.DataFrame(new_cols, index=m.index)], axis=1)

    if target not in m.columns:
        raise ValueError(f"Target column '{target}' not found in Excel sheet.")
    return m


def _read_excel_flexible(xlsx_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    for s in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=s)
        if df is not None and not df.empty:
            return df
    raise ValueError(f"No non-empty sheet found in {xlsx_path}")


# -------------------- public APIs --------------------

def build_features_from_mapping(
    mapping_csv: str,
    log_folder: str,
    xlsx_path: str,
    target: str,
    output_prefix: str,
    category_name: str = "",
    atom_index_cols: Optional[List[str]] = None,
    join_col: Optional[str] = None,
) -> str:
    mapping_csv = str(mapping_csv)
    log_folder = Path(log_folder)
    xlsx_path = Path(xlsx_path)
    output_prefix = Path(output_prefix)

    if atom_index_cols is None:
        atom_index_cols = ATOM_INDEX_COLS_DEFAULT

    mapping_df = pd.read_csv(mapping_csv)
    feat_df = _build_feature_table(mapping_df, log_folder, atom_index_cols)
    xlsx_df = _read_excel_flexible(xlsx_path)

    pair_mode = (join_col is None) and ("Ar1" in xlsx_df.columns) and ("Ar2" in xlsx_df.columns)
    if pair_mode:
        merged = _build_pair_features(xlsx_df, feat_df, target)
    else:
        merged = _merge_single(xlsx_df, feat_df, target, join_col=join_col)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    features_csv = Path(f"{output_prefix}_features.csv")
    merged_csv = Path(f"{output_prefix}_merged.csv")

    feat_df.to_csv(features_csv, index=False)
    merged.to_csv(merged_csv, index=False)

    meta = {
        "category": category_name,
        "target": target,
        "pair_mode": pair_mode,
        "features_csv": str(features_csv),
        "merged_csv": str(merged_csv),
        "n_rows_features": int(len(feat_df)),
        "n_rows_merged": int(len(merged)),
        "n_cols_features": int(feat_df.shape[1]),
        "n_cols_merged": int(merged.shape[1]),
    }
    Path(f"{output_prefix}_feature_build_report.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return str(features_csv)


def run_regression_from_features_csv(
    features_csv: str,
    target: str,
    max_features: int,
    output_prefix: str,
    category_name: str = "",
    open_browser: Optional[bool] = None,
) -> None:
    t0 = time.time()

    features_csv = str(features_csv)
    output_prefix = Path(output_prefix)

    merged = pd.read_csv(features_csv)

    if target not in merged.columns:
        merged_csv = Path(features_csv.replace("_features.csv", "_merged.csv"))
        if merged_csv.exists():
            merged = pd.read_csv(merged_csv)
        else:
            raise ValueError(
                f"Target column '{target}' not found in {features_csv}, and no merged CSV found."
            )

    def _normalize_col_name(col: str) -> str:
        return str(col).strip().lower().replace(" ", "")

    def _is_identifier_like(col: str) -> bool:
        c = _normalize_col_name(col)
        identifier_keywords = {
            "ar1", "ar2", "_join1", "_join2", "_join_key",
            "compound", "substituent", "name", "filename", "log", "logfile", "logkey"
        }
        return c in {_normalize_col_name(x) for x in identifier_keywords}

    def _is_leaky_feature(col: str, target: str) -> bool:
        c_raw = str(col).strip()
        c = _normalize_col_name(col)
        t = _normalize_col_name(target)

        # 1) exact target
        if c == t:
            return True

        # 2) explicit response / kinetic / experimental outcome columns
        leakage_keywords = [
            "kobs",
            "ln(kobs)",
            "log(kobs)",
            "l nkobs",
            "lnkobs",
            "rate",
            "rateconstant",
            "reactionrate",
            "yield",
            "conversion",
            "selectivity",
            "ee",
            "er",
            "dr",
        ]

        if any(k.replace(" ", "") in c for k in leakage_keywords):
            return True

        # 3) target-family leakage:
        #    if target itself is kinetic-like, exclude other columns from same family too
        target_family_keywords = [
            "kobs",
            "ln(kobs)",
            "log(kobs)",
            "lnkobs",
        ]
        if any(k.replace(" ", "") in t for k in target_family_keywords):
            if any(k.replace(" ", "") in c for k in target_family_keywords):
                return True

        # 4) obvious solvent-specific response columns often mixed into descriptor table
        #    e.g. kobs_MeCN, ln(kobs)_toluene
        solvent_like_tokens = ["mecn", "toluene", "dmso", "thf", "acn", "meoh", "etoh"]
        if any(tok in c for tok in solvent_like_tokens) and any(
            k.replace(" ", "") in c for k in ["kobs", "ln(kobs)", "log(kobs)", "lnkobs", "rate"]
        ):
            return True

        return False

    exclude_cols = []
    leaky_cols = []
    kept_cols = []

    for col in merged.columns:
        if _is_identifier_like(col):
            exclude_cols.append(col)
        elif _is_leaky_feature(col, target):
            leaky_cols.append(col)
        else:
            kept_cols.append(col)

    feature_cols = kept_cols

    print(f"\n[INFO] Category: {category_name}")
    print(f"[INFO] Target: {target}")
    print(f"[INFO] Total columns in merged table: {len(merged.columns)}")
    print(f"[INFO] Identifier-like excluded columns ({len(exclude_cols)}): {exclude_cols}")
    print(f"[INFO] Leaky/response-like excluded columns ({len(leaky_cols)}): {leaky_cols}")
    print(f"[INFO] Final feature columns count: {len(feature_cols)}")

    if len(feature_cols) == 0:
        raise ValueError(
            f"No valid feature columns remain after excluding identifiers and leakage columns for target '{target}'."
        )

    X = merged[feature_cols].copy()
    y = pd.to_numeric(merged[target], errors="coerce")
    X = _as_numeric_df(X, feature_cols)

    valid = y.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)

    keep_cols = [c for c in X.columns if X[c].notna().sum() >= 3]
    dropped_sparse_cols = [c for c in X.columns if c not in keep_cols]
    X = X[keep_cols]

    print(f"[INFO] Feature columns retained after numeric/sparsity filtering: {len(keep_cols)}")
    if dropped_sparse_cols:
        print(f"[INFO] Dropped sparse columns ({len(dropped_sparse_cols)}): {dropped_sparse_cols}")

    if X.shape[1] == 0:
        raise ValueError(
            f"All candidate feature columns were removed after numeric/sparsity filtering for target '{target}'."
        )

    best, tested_df = _subset_search(
        X, y, target, max_features=max_features, category_name=category_name
    )

    top_models_csv = Path(f"{output_prefix}_top_models_by_k.csv")
    tested_df.sort_values(["Q2", "R2", "RMSE"], ascending=[False, False, True]).to_csv(
        top_models_csv, index=False
    )

    report = {
        "category": category_name,
        "target": target,
        "excluded_identifier_cols": exclude_cols,
        "excluded_leaky_cols": leaky_cols,
        "final_feature_cols": list(X.columns),
        "best_features": best.get("best_features", []),
        "metrics": best.get("metrics", {}),
        "equation_pretty": best.get("equation_pretty", ""),
        "timing_seconds": {"total_case_seconds": round(time.time() - t0, 6)},
        "files": {
            "features_csv": str(features_csv),
            "top_models_csv": str(top_models_csv),
        },
    }

    report_json = Path(f"{output_prefix}_regression_report.json")
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if best.get("best_features"):
        feats = best["best_features"]
        Xi = X[feats]
        model = _fit_pipeline(Xi, y)
        yhat = model.predict(Xi)
        coef_orig, intercept_orig = _orig_unit_coeffs(model)
        eq_pretty = _equation_pretty(
            feats,
            coef_orig,
            intercept_orig,
            target,
            decimals=int(os.environ.get("FORMULA_DECIMALS", "2")),
        )

        metrics = report["metrics"]
        metrics_text = (
            f"Model: [{', '.join(feats)}] | "
            f"Q²={metrics.get('q2', float('nan')):.3f} | "
            f"R²={metrics.get('r2', float('nan')):.3f}<br>"
            f"Regression Formula: {eq_pretty}<br>"
            f"RMSE={metrics.get('rmse', float('nan')):.3g} | "
            f"CV={metrics.get('cv', '')}"
        )

        xy_html = Path(f"{output_prefix}_Regression_Plot.html")
        _make_xy_plot(
            y_true=np.asarray(y, dtype=float),
            y_pred=np.asarray(yhat, dtype=float),
            title=f"{category_name} | Actual vs Predicted",
            out_html=xy_html,
            metrics_text=metrics_text,
        )

        coef_html = Path(f"{output_prefix}_Coefficient_Bar.html")
        _make_coef_bar_plot(
            feats, coef_orig, coef_html, title=f"{category_name} | Coefficients"
        )

        if open_browser is None:
            open_browser = os.environ.get("OPEN_PLOTS", "0").strip().lower() not in (
                "0", "false", "no"
            )
        if open_browser:
            try:
                webbrowser.open_new_tab(xy_html.resolve().as_uri())
            except Exception:
                pass
            try:
                webbrowser.open_new_tab(coef_html.resolve().as_uri())
            except Exception:
                pass
        else:
            print(f"[INFO] Plots saved to: {xy_html}, {coef_html}")
    else:
        print(f"[INFO] No valid feature subset found that meets the criteria for target '{target}'.")   
        
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
    features_only: bool = False,
) -> None:
    features_csv = build_features_from_mapping(
        mapping_csv=mapping_csv,
        log_folder=log_folder,
        xlsx_path=xlsx_path,
        target=target,
        output_prefix=output_prefix,
        category_name=category_name,
        atom_index_cols=atom_index_cols,
        join_col=join_col,
    )
    if features_only:
        return
    run_regression_from_features_csv(
        features_csv=features_csv,
        target=target,
        max_features=max_features,
        output_prefix=output_prefix,
        category_name=category_name,
        open_browser=open_browser,
    )