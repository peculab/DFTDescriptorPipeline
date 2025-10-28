# -*- coding: utf-8 -*-
"""
extractor_regr.py (robust, Colab-ready)

What you get:
- NBO BD parser tolerant to "BD(1)" and "BD ( 1 )" formats
- OH detection via NBO; fallback to geometry distance if NBO missing
- F/G derivation via BD graph; fallback with geometry adjacency
- Strict "no last-value" policy across files
- Safe numeric coercions for Ar/Ar_* columns
- Optional Sterimol with morfeus-ml (skipped if unavailable)
- Model search requiring BOTH R² >= 0.70 and Q²(LOO) >= 0.70
"""

import os, re, glob, math, random
from typing import Any, Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ---------------- Utilities ----------------

_INT_FULL = re.compile(r'^\d+$')
def safe_int(tok: Any) -> Optional[int]:
    if tok is None: return None
    tok = str(tok).strip()
    if _INT_FULL.match(tok):
        try: return int(tok)
        except Exception: return None
    return None

# BD line tolerant to spaces: "BD(1) C 12 - O 3" or "BD ( 1 ) C 12 - O 3"
BD_LINE = re.compile(r'^\s*BD\s*\(\s*\d+\s*\)\s+([A-Z][a-z]?)\s+(\d+)\s*-\s*([A-Z][a-z]?)\s+(\d+)\s*$', re.IGNORECASE)

# Coord line from orientation tables
COORD_LINE = re.compile(r'^\s*(\d+)\s+([A-Z][a-z]?|\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$')

_Z2E = {1:"H",6:"C",7:"N",8:"O",9:"F",15:"P",16:"S",17:"Cl",35:"Br",53:"I"}

def _extract_bd_bonds_anywhere(text: str) -> List[Tuple[int,str,int,str,int]]:
    """Return list of tuples: (order, symA, idxA, symB, idxB). Order defaults to 1 here."""
    bonds = []
    for line in text.splitlines():
        m = BD_LINE.match(line)
        if not m: continue
        a_sym, a_idx, b_sym, b_idx = m.groups()
        ai, bi = safe_int(a_idx), safe_int(b_idx)
        if ai is None or bi is None: continue
        bonds.append((1, a_sym.upper(), ai, b_sym.upper(), bi))
    return bonds

def _bond_graph_from_bd(bonds):
    """Graph as dict: node -> list of (neighbor, order, neighbor_symbol)."""
    g = {}
    for order,xs,xi,ys,yj in bonds:
        g.setdefault(xi, []).append((yj, order, ys))
        g.setdefault(yj, []).append((xi, order, xs))
    return g

def _parse_center_coords(text: str):
    """
    Parse the last Standard/Input orientation block strictly.
    Returns (elems: dict idx->sym, coords: dict idx->(x,y,z)).
    """
    head_re = re.compile(r"(Standard|Input)\s+orientation\s*:\s*", re.IGNORECASE)
    hits = [m.start() for m in head_re.finditer(text)]
    if not hits: return {}, {}
    start = hits[-1]
    lines = text[start:].splitlines()

    header_idx = None
    for i,ln in enumerate(lines[:300]):
        s = ln.strip()
        if ("X" in s and "Y" in s and "Z" in s) and (("Center" in s) or ("Atomic" in s) or ("Coordinates" in s)):
            header_idx = i; break
    if header_idx is None: return {}, {}

    elems, coords, seen = {}, {}, set()
    for ln in lines[header_idx+1:header_idx+1+2000]:
        s = ln.strip()
        if not s:
            if coords: break
            else: continue
        if set(s) <= set("-= "):
            if coords: break
            else: continue
        m = COORD_LINE.match(s)
        if not m:
            # stop if a new table header appears
            if s.lower().startswith(("standard","input")) and "orientation" in s.lower(): break
            continue
        idx_s, sym_or_Z, x_s, y_s, z_s = m.groups()
        idx = safe_int(idx_s)
        if idx is None or idx in seen: continue
        try:
            x,y,z = float(x_s), float(y_s), float(z_s)
        except Exception:
            continue
        if _INT_FULL.match(sym_or_Z):
            sym = _Z2E.get(int(sym_or_Z), "X")
        else:
            sym = sym_or_Z
        elems[idx] = sym
        coords[idx] = (x,y,z)
        seen.add(idx)
    return elems, coords

# ---------------- OH detection (NBO + geometry fallback) ----------------

_OH_LINE = re.compile(r'BD\s*\(\s*1\s*\)\s*O\s*(\d+)\s*-\s*H\s*(\d+)', re.IGNORECASE)

def find_oh_bonds(text: str):
    pairs = []
    for ln in text.splitlines():
        if "BD" not in ln: continue
        m = _OH_LINE.search(ln)
        if not m: continue
        oi, hi = safe_int(m.group(1)), safe_int(m.group(2))
        if oi is None or hi is None: continue
        pairs.append((oi, hi))
    return pairs

def _find_oh_by_geometry(text: str, max_dist=1.20, relax=1.25):
    """Nearest O–H by distance if NBO is missing; returns (O, H, elems, coords)."""
    elems, coords = _parse_center_coords(text)
    if not elems or not coords:
        return None, None, elems, coords
    oxy = [i for i,s in elems.items() if s.upper() == 'O']
    hyd = [i for i,s in elems.items() if s.upper() == 'H']
    if not oxy or not hyd:
        return None, None, elems, coords

    def dist(a,b):
        xa,ya,za = coords[a]; xb,yb,zb = coords[b]
        dx,dy,dz = xa-xb, ya-yb, za-zb
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    best = (None, None, 1e9)
    for o in oxy:
        for h in hyd:
            d = dist(o,h)
            if d < best[2]:
                best = (o, h, d)

    if best[0] is not None and (best[2] <= max_dist or best[2] <= relax):
        return best[0], best[1], elems, coords
    return None, None, elems, coords

# ---------------- Robust FG derivation ----------------

def derive_fg_from_geometry_robust(text: str, prefer_single_bonds: bool=True):
    """
    Steps:
      1) Try NBO OH; else geometry OH (nearest O–H)
      2) Build BD graph; augment with geometry adjacency (<=1.70 Å) if needed
      3) C1 = carbon neighbor of O
      4) C2 = a carbon neighbor of C1 (≠ O), pick one with larger degree
      5) F,G = two carbon neighbors of C2 (≠ C1), prefer single bonds
    """
    # Step 1: OH
    oh = find_oh_bonds(text)
    if oh:
        O, H = oh[0]
    else:
        O, H, elems_g, coords_g = _find_oh_by_geometry(text)
        if O is None or H is None:
            elems2, coords2 = _parse_center_coords(text)
            return None, None, None, None, elems2, coords2

    # Step 2: Graph
    bonds = _extract_bd_bonds_anywhere(text)
    g = _bond_graph_from_bd(bonds)

    elems_geo, coords_geo = _parse_center_coords(text)
    if elems_geo and coords_geo:
        idxs = list(coords_geo.keys())
        for i in idxs:
            for j in idxs:
                if j <= i: continue
                xi, yi, zi = coords_geo[i]; xj, yj, zj = coords_geo[j]
                d = math.dist((xi,yi,zi),(xj,yj,zj))
                if d <= 1.70:  # geometry adjacency
                    si = elems_geo.get(i, 'X').upper()
                    sj = elems_geo.get(j, 'X').upper()
                    g.setdefault(i, []).append((j, 1, sj))
                    g.setdefault(j, []).append((i, 1, si))

    # Step 3: C1
    neigh_O = g.get(O, [])
    cands = [(n,ordr,sym) for (n,ordr,sym) in neigh_O if sym.upper() == 'C']
    if not cands:
        return None, None, None, None, elems_geo, coords_geo
    if prefer_single_bonds:
        singles = [n for (n,ordr,_) in cands if ordr == 1]
        C1 = singles[0] if singles else cands[0][0]
    else:
        C1 = cands[0][0]

    # Step 4: C2
    neigh_C1 = [(n,ordr,sym) for (n,ordr,sym) in g.get(C1, []) if n != O]
    carb = [n for (n,ordr,sym) in neigh_C1 if sym.upper() == 'C']
    pool = carb or [n for (n,_,_) in neigh_C1]
    if not pool:
        return C1, None, None, None, elems_geo, coords_geo
    C2 = sorted(pool, key=lambda n: len(g.get(n, [])), reverse=True)[0]

    # Step 5: F & G
    fg = [(n,ordr,sym) for (n,ordr,sym) in g.get(C2, []) if n != C1]
    singles = [n for (n,ordr,sym) in fg if (ordr == 1 and sym.upper() == 'C')]
    others  = [n for (n,ordr,sym) in fg if n not in singles and sym.upper() == 'C']
    ordered = singles + others
    F = ordered[0] if len(ordered) >= 1 else (fg[0][0] if fg else None)
    G = ordered[1] if len(ordered) >= 2 else (fg[1][0] if len(fg) > 1 else None)

    return C1, C2, F, G, elems_geo, coords_geo

def derive_fg_from_geometry(text: str):
    return derive_fg_from_geometry_robust(text)

# ---------------- NBO / geometry helpers for Sterimol ----------------

def extract_nbo_section(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    m = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)-{30,}", content, re.DOTALL|re.IGNORECASE)
    if not m: return None
    return m.group(1)

def extract_last_standard_orientation(log_path):
    with open(log_path, "r", errors='ignore') as f:
        lines = f.readlines()
    geometries, block, reading = [], [], False
    for line in lines:
        if "Standard orientation" in line:
            block=[]; reading=True; continue
        if reading:
            if "-----" in line: continue
            if any(x in line for x in ["Center","Atomic","Number"]): continue
            if line.strip()=="":
                if block: geometries.append(block)
                block=[]; reading=False
            else:
                if re.match(r"^\s*\d+\s+\d+\s+\d+\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line):
                    block.append(line)
                else:
                    if block: geometries.append(block)
                    block=[]; reading=False
    if not geometries: return None
    last_geom = geometries[-1]
    atoms = []
    for line in last_geom:
        parts = line.split()
        try:
            atomic_num = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            sym = _Z2E.get(atomic_num, None)
            if sym is None: return None
            atoms.append((sym, x, y, z))
        except Exception:
            return None
    return atoms

def write_xyz(atom_list, filename):
    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\nExtracted from Gaussian log\n")
        for atom in atom_list:
            f.write(f"{atom[0]}  {atom[1]:.8f}  {atom[2]:.8f}  {atom[3]:.8f}\n")

# ---------------- Sterimol (optional) ----------------

def add_sterimol_to_df(df, log_folder):
    try:
        from morfeus import read_xyz, Sterimol
        from morfeus.utils import get_radii
    except ImportError:
        print("[sterimol] morfeus-ml not installed; skip Sterimol columns.")
        for col in ["Ar_Ster_L","Ar_Ster_B1","Ar_Ster_B5"]:
            if col not in df.columns: df[col] = None
        return df

    for col in ["Ar_Ster_L","Ar_Ster_B1","Ar_Ster_B5"]:
        if col not in df.columns: df[col] = None

    log_files = glob.glob(os.path.join(log_folder, "*.log"))
    log_map = {os.path.basename(f).replace(".log",""): f for f in log_files}

    for idx, row in df.iterrows():
        mol = str(row["Ar"])
        p = log_map.get(mol)
        print(f"\n[Sterimol] [{mol}] log: {p}")
        if not p:
            print("  [SKIP] log missing"); continue
        atoms = extract_last_standard_orientation(p)
        if not atoms:
            print("  [SKIP] atoms parse failed"); continue

        try:
            a = safe_int(row.get("Ar_a"))
            b = safe_int(row.get("Ar_b"))
            d = safe_int(row.get("Ar_d"))
            c = safe_int(row.get("Ar_c"))
            e = safe_int(row.get("Ar_e"))
            if None in (a,b,d,c,e):
                print("  [WARN] Ar indices non-numeric; skip Sterimol for this row.")
                continue

            atoms_to_keep = [a0 for i,a0 in enumerate(atoms, start=1) if i not in (a,b,d)]
            if len(atoms_to_keep) < 2:
                print("  [SKIP] atoms_to_keep < 2"); continue

            xyz_path = f"{mol}_filtered.xyz"
            write_xyz(atoms_to_keep, xyz_path)

            elements, coords = read_xyz(xyz_path)
            radii = get_radii(elements, radii_type="bondi")
            radii = [1.09 if r == 1.20 else r for r in radii]  # tweak H radius

            ster = Sterimol(elements, coords, int(c), int(e), radii=radii)
            df.at[idx,"Ar_Ster_L"]  = ster.L_value
            df.at[idx,"Ar_Ster_B1"] = ster.B_1_value
            df.at[idx,"Ar_Ster_B5"] = ster.B_5_value
            print(f"  [OK] L={ster.L_value}, B1={ster.B_1_value}, B5={ster.B_5_value}")
        except Exception as e:
            print(f"  [ERROR] Sterimol failed: {e}")
            continue
    return df

# ---------------- Excel/log helpers ----------------

def _normalize_ar_value(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s.lower() in {"nan","none",""}: return np.nan
    if re.match(r"^-?\d+\.0$", s): s = s[:-2]
    return s

def _canon_ar_cols(df: pd.DataFrame):
    ar_cols_raw = [c for c in df.columns if re.fullmatch(r"[Aa][Rr]\s*\d+", str(c).strip())]
    mapping = {}
    for c in ar_cols_raw:
        n = re.search(r"\d+", str(c)).group(0)
        mapping[c] = f"Ar{n}"
    if mapping: df.rename(columns=mapping, inplace=True)
    return [mapping.get(c, c) for c in ar_cols_raw]

def _list_log_basenames(log_folder: str):
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(log_folder, "*.log"))}

# ---------------- Modeling ----------------

def prepare_data(path, features, target):
    data = pd.read_excel(path)
    data = data.dropna(subset=features + [target])
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

def compute_loocv_metrics(X, y):
    n = X.shape[0]
    Xd = np.hstack([np.ones((n,1)), X])
    XtX = Xd.T @ Xd
    XtX += 1e-8 * np.eye(XtX.shape[0])
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ Xd.T @ y
    H = Xd @ XtX_inv @ Xd.T
    h = np.diag(H)
    yhat = Xd @ beta
    yloo = (yhat - h*y) / (1 - h)
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - yhat)**2)
    ss_loo = np.sum((y - yloo)**2)
    return {
        "r2_full": 1 - ss_res/ss_tot if ss_tot>0 else 0.0,
        "q2_loocv": 1 - ss_loo/ss_tot if ss_tot>0 else 0.0,
        "rmse": np.sqrt(np.mean((y - yloo)**2)),
        "coefficients": beta[1:].tolist(),
        "intercept": beta[0],
    }

def evaluate_combinations(data, target, feature_set):
    try:
        X = data[feature_set].astype(float).values
        y = data[target].values
        res = compute_loocv_metrics(X, y)
        res["features"] = feature_set
        return res
    except Exception as e:
        print(f"[ERROR] Combo {feature_set} failed: {e}")
        return None

def search_best_models(data, features, target, max_features, r2_threshold=0.7, q2_threshold=0.7,
                       save_csv=True, csv_path="regression_search_results.csv", verbose=True):
    all_results = []
    for k in range(1, max_features+1):
        if verbose: print(f"\n🔍 Testing {k}-feature combinations")
        for c in combinations(features, k):
            r = evaluate_combinations(data, target, list(c))
            if r and (r["r2_full"] >= r2_threshold) and (r["q2_loocv"] >= q2_threshold):
                all_results.append(r)
                if verbose:
                    print(f"✅ {list(c)} | R²={r['r2_full']:.3f} | Q²={r['q2_loocv']:.3f}")
            elif verbose:
                print(f"❌ {list(c)} | skipped")
    if not all_results:
        print("⚠️ No valid models found."); return [], None
    df_all = pd.DataFrame(all_results)
    df_all["num_features"] = df_all["features"].apply(len)
    if save_csv:
        df_all.to_csv(csv_path, index=False)
        if verbose: print(f"\n📄 Saved all {len(df_all)} results to {csv_path}")
    best_model = df_all.sort_values(by="q2_loocv", ascending=False).iloc[0].to_dict()
    if verbose:
        print(f"\n🏆 Best model: {best_model['features']} | Q²={best_model['q2_loocv']:.3f} | R²={best_model['r2_full']:.3f}")
    return df_all.to_dict(orient="records"), best_model

# ---------------- Diagnostics ----------------

def plot_best_regression(target, df, best_model, savepath='Regression_Plot.png'):
    X_columns = best_model['features']
    coefficients = np.array(best_model['coefficients'])
    intercept = best_model['intercept']
    y_actual = df[target]; X_values = df[X_columns].values
    y_pred = np.dot(X_values, coefficients) + intercept
    formula = f'{target} = ' + ' + '.join([f"{c:.2f}({f})" for c,f in zip(coefficients, X_columns)]) + f" + {intercept:.2f}"
    print("Regression Formula:", formula)
    fig, ax = plt.subplots(figsize=(8,7))
    ax.set_facecolor('w')
    ax.plot(y_actual, y_actual, color='k')
    ax.scatter(y_actual, y_pred, edgecolor='b', facecolor='b', alpha=0.7)
    ax.set_ylabel(f'Predicted {target}', fontsize=18, color='k')
    ax.set_xlabel(f'Experimental {target}', fontsize=18, color='k')
    ax.spines['bottom'].set_color('k'); ax.spines['left'].set_color('k')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.text(0.55,0.35, f'$R^2= {best_model["r2_full"]:.2f}$', fontsize=16)
    fig.text(0.55,0.30, f'rmse = {best_model["rmse"]:.2f}', fontsize=16)
    fig.text(0.55,0.25, f'$Q^2= {best_model["q2_loocv"]:.2f}$ (LOO)', fontsize=16)
    fig.text(0.55,0.20, f'{len(y_actual)} data points', fontsize=16, style='italic')
    fig.tight_layout(); plt.savefig(savepath, bbox_inches='tight'); plt.show()

def report_index_problems(df, log_folder=None):
    idx_cols = ["Ar_c","Ar_e","Ar_a","Ar_b","Ar_d","Ar_f","Ar_g"]
    def bad(row): return any((x is None) or (isinstance(x,float) and np.isnan(x)) for x in row[idx_cols])
    bad_rows = df[df.apply(bad, axis=1)]
    if len(bad_rows)==0:
        print("✅ No molecules have index columns as None/NaN; all extracted correctly!")
    else:
        print("❗Rows with None/NaN indices:\n"); print(bad_rows[["Ar"]+idx_cols])
        if log_folder is not None and "log_file" not in bad_rows.columns:
            bad_rows = bad_rows.copy()
            bad_rows["log_file"] = bad_rows["Ar"].apply(lambda ar: f"{log_folder}/{ar}.log")
            print("\nCorresponding log_file:"); print(bad_rows[["Ar","log_file"]])
        bad_rows.to_excel("problem_index_report.xlsx", index=False)
        print("\nSaved as problem_index_report.xlsx")

# ---------------- Main pipeline ----------------

def run_full_pipeline(log_folder, xlsx_path, max_features, target="ln(kobs)",
                      output_path="final_output.xlsx", plot_path='Regression_Plot.png',
                      auto_pairing=True):
    print(f"\n[STEP1] Read Excel: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    print(f"\n[STEP2] Extracting log features...")
    ar_cols = _canon_ar_cols(df)
    if "Ar" in df.columns and not ar_cols:
        ar_series = df["Ar"].dropna()
    else:
        stacks = [df[c].dropna() for c in ar_cols] if ar_cols else []
        ar_series = pd.concat(stacks, ignore_index=True).dropna() if stacks else pd.Series([], dtype=object)
    ar_series = ar_series.apply(_normalize_ar_value).dropna()
    unique_ar_df = pd.DataFrame({"Ar": ar_series.unique()})
    unique_ar_df["Ar"] = unique_ar_df["Ar"].apply(_normalize_ar_value)
    unique_ar_df["Ar_key"] = unique_ar_df["Ar"]
    unique_ar_df["log_path"] = unique_ar_df["Ar"].apply(lambda ar: os.path.join(log_folder, f"{ar}.log"))
    unique_ar_df["log_exists"] = unique_ar_df["log_path"].apply(os.path.exists)
    unique_ar_df = unique_ar_df[unique_ar_df["log_exists"]].reset_index(drop=True)
    if unique_ar_df.empty:
        print("⚠️ No valid Ar entries with matching log files. Stop."); return df, [], {}

    for index, row in unique_ar_df.iterrows():
        ar = row["Ar"]; log_file = row["log_path"]
        print(f"\n==== [{index+1}/{len(unique_ar_df)}] [{ar}] Processing log: {log_file} ====")
        try:
            with open(log_file, "r", errors="ignore") as fh: log_text = fh.read()

            # Robust FG first
            C1=C2=F=G=None; elems=coords=None
            try:
                C1,C2,F,G,elems,coords = derive_fg_from_geometry_robust(log_text)
                # small debug
                ohs = find_oh_bonds(log_text)
                bonds_dbg = _extract_bd_bonds_anywhere(log_text)
                print(f"      [DBG] OH={len(ohs)}, BD edges={len(bonds_dbg)}, FG=({C1},{C2},{F},{G})")
            except Exception as e:
                print(f"      [DBG] robust FG error: {e}")

            # Attach parsed indices
            unique_ar_df.at[index,"Ar_c"] = C1
            unique_ar_df.at[index,"Ar_e"] = C2
            unique_ar_df.at[index,"Ar_f"] = F
            unique_ar_df.at[index,"Ar_g"] = G
            # Optional NBO extras if ever needed later
            unique_ar_df.at[index,"Ar_a"] = None
            unique_ar_df.at[index,"Ar_b"] = None
            unique_ar_df.at[index,"Ar_d"] = None

        except Exception as e:
            print(f"[ERROR] Error occurred while processing Ar={ar}: {e}")
            continue

    # Coerce numeric before downstream usage
    for col in ["Ar","Ar_c","Ar_e","Ar_a","Ar_b","Ar_d","Ar_f","Ar_g"]:
        if col in unique_ar_df.columns:
            if col=="Ar": 
                unique_ar_df[col] = unique_ar_df[col].apply(_normalize_ar_value)
            else:
                unique_ar_df[col] = pd.to_numeric(unique_ar_df[col], errors="coerce")

    # Sterimol (optional; safe to skip)
    unique_ar_df = add_sterimol_to_df(unique_ar_df, log_folder)

    report_index_problems(unique_ar_df, log_folder)

    essential_cols = ["Ar_c","Ar_e","Ar_f","Ar_g"]
    before = len(unique_ar_df)
    unique_ar_df = unique_ar_df.dropna(subset=essential_cols)
    after  = len(unique_ar_df)
    print(f"🧹 Dropped {before - after} Ar rows with missing essentials ({essential_cols})")
    unique_ar_df.to_excel("unique_ar_features.xlsx", index=False)

    print(f"\n[STEP3] Merge features into main df")
    ar_cols = _canon_ar_cols(df)

    if not ar_cols and "Ar" in df.columns:
        df["Ar"] = df["Ar"].apply(_normalize_ar_value)
        df = df.merge(unique_ar_df, on="Ar", how="left")
        meta_ex = {"Ar","Ar_key","log_path","log_exists"}
        features = [c for c in unique_ar_df.columns if c not in meta_ex]
    else:
        for c in ar_cols: df[c] = df[c].apply(_normalize_ar_value)
        for c in ar_cols:
            pref = f"{c}_"; right = unique_ar_df.add_prefix(pref)
            df = df.merge(right, left_on=c, right_on=pref+"Ar", how="left")
        feature_ex_suffix = ("_Ar","_Ar_key","_log_path","_log_exists")
        features = [c for c in df.columns if re.match(r"^Ar\d+_", c) and not any(c.endswith(s) for s in feature_ex_suffix)]

    # Guarded numeric coercion (only cast columns that exist)
    cols_to_cast = []
    for base in ["Ar_f","Ar_g"]:
        if base in df.columns: cols_to_cast.append(base)
    cols_to_cast.extend([c for c in df.columns if c.endswith(("_Ar_f","_Ar_g"))])
    for col in cols_to_cast:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_excel(output_path, index=False)

    print(f"\n[STEP4] Regression")
    if not features:
        print("⚠️ No available feature columns."); return df, [], {}

    df_model = df.dropna(subset=features + [target])
    if df_model.empty:
        print("⚠️ No data for regression after dropna."); return df, [], {}

    # Simple ungrouped search (robust and fast for small max_features)
    results, best = search_best_models(df_model, features, target, max_features,
                                       r2_threshold=0.7, q2_threshold=0.7,
                                       save_csv=True, csv_path="regression_search_results.csv",
                                       verbose=True)

    if best:
        plot_best_regression(target, df_model, best, savepath='Regression_Plot.png')

    print("\n✅ Analysis complete!")
    return df, results, best
