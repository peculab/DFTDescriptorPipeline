# extractor_regr.py (cleaned)
import os, re, glob, math, random, itertools as it
import numpy as np
import pandas as pd
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# =========================
# Element helpers
# =========================
_Z2E = {1:'H',6:'C',7:'N',8:'O',9:'F',15:'P',16:'S',17:'Cl',35:'Br',53:'I'}
_COV = {'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'S':1.05,'Cl':1.02,'Br':1.20,'I':1.39,'X':0.80}

def _dist(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

# =========================
# Ultra-robust coordinate parsers (dict[int->sym], dict[int->(x,y,z)])
# =========================
def _ultra_parse_standard_or_input(text):
    # Find the last "Standard/Input orientation:" header then parse the following table.
    head_re = re.compile(r"(Standard|Input)\s+orientation\s*:\s*", re.IGNORECASE)
    hits = [m.start() for m in head_re.finditer(text)]
    if not hits:
        return {}, {}
    start = hits[-1]
    tail = text[start:]
    lines = tail.splitlines()

    # Locate a header line that contains X Y Z and some of Center/Atomic/Coordinates
    header_idx = None
    for i, ln in enumerate(lines[:300]):
        s = ln.strip()
        if ("X" in s and "Y" in s and "Z" in s) and (("Center" in s) or ("Atomic" in s) or ("Coordinates" in s)):
            header_idx = i
            break
    if header_idx is None:
        return {}, {}

    elems, coords = {}, {}
    idx_seen = set()
    for ln in lines[header_idx+1:header_idx+1+2000]:
        s = ln.strip()
        if not s:
            if coords: break
            else: continue
        if set(s) <= set("-= "):
            if coords: break
            else: continue
        parts = s.split()
        # Expect: center_idx atomicZ atomicType x y z
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            try:
                center_idx = int(parts[0]); Z = int(parts[1])
                x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            except Exception:
                continue
            if center_idx not in idx_seen:
                elems[center_idx] = _Z2E.get(Z, 'X')
                coords[center_idx] = (x,y,z)
                idx_seen.add(center_idx)
        else:
            # Stop if we reached another section header
            if s.lower().startswith(("standard", "input")) and "orientation" in s.lower():
                break
    return elems, coords

def _ultra_parse_checkpoint(text):
    m = re.search(r"Structure\s+from\s+the\s+checkpoint\s+file[\s\S]{0,2000}?Coordinates", text, re.IGNORECASE)
    if not m:
        return {}, {}
    lines = text[m.end():].splitlines()
    elems, coords = {}, {}
    idx = 1
    for ln in lines:
        s = ln.strip()
        if not s:
            if coords: break
            else: continue
        if set(s) <= set("-= "):
            if coords: break
            else: continue
        if any(k in s for k in ["Charge", "Multiplicity", "Standard", "Input"]):
            break
        parts = s.split()
        try:
            if len(parts) >= 5 and parts[0].isdigit():
                sym = parts[1]; x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            else:
                sym = parts[0]; x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            elems[idx] = sym
            coords[idx] = (x,y,z)
            idx += 1
        except Exception:
            pass
    return elems, coords

def _ultra_parse_zmatrix_or_input(text):
    m_iter = list(re.finditer(r"(Z-Matrix|Input)\s+orientation\s*:", text, re.IGNORECASE))
    if not m_iter:
        return {}, {}
    start = m_iter[-1].end()
    tail = text[start:]
    lines = tail.splitlines()

    # Find a line that looks like the table header
    header_i = None
    for i, ln in enumerate(lines[:300]):
        s = ln.strip().lower()
        if ("x" in s and "y" in s and "z" in s) and ("center" in s or "atomic" in s or "coordinates" in s):
            header_i = i
            break
    if header_i is None:
        header_i = 0

    elems, coords = {}, {}
    idx_seen = set()
    for ln in lines[header_i+1:header_i+1+2000]:
        s = ln.strip()
        if not s:
            if coords: break
            else: continue
        if set(s) <= set("-= "):
            if coords: break
            else: continue
        parts = s.split()
        if len(parts) < 4:
            continue
        try:
            x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
        except:
            continue
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            center_idx = int(parts[0]); Z = int(parts[1]); sym = _Z2E.get(Z,'X')
        else:
            center_idx = (max(idx_seen)+1) if idx_seen else 1
            sym = parts[0] if parts[0].isalpha() else 'X'
        if center_idx in idx_seen:
            continue
        elems[center_idx] = sym
        coords[center_idx] = (x,y,z)
        idx_seen.add(center_idx)
    return elems, coords

def _ultra_parse_any_center_table(text):
    pat = re.compile(r"(Center.*Atomic.*Coordinates.*X.*Y.*Z)", re.IGNORECASE)
    spans = [m.span() for m in pat.finditer(text)]
    if not spans:
        return {}, {}
    start = spans[-1][1]
    lines = text[start:].splitlines()
    elems, coords = {}, {}
    idx_seen = set()
    for ln in lines[:2000]:
        s = ln.strip()
        if not s:
            if coords: break
            else: continue
        if set(s) <= set("-= "):
            if coords: break
            else: continue
        parts = s.split()
        if len(parts) >= 4:
            try:
                x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            except:
                continue
            if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
                center_idx = int(parts[0]); Z = int(parts[1]); sym = _Z2E.get(Z,'X')
            else:
                center_idx = (max(idx_seen)+1) if idx_seen else 1; sym = parts[0] if parts[0].isalpha() else 'X'
            if center_idx in idx_seen: continue
            elems[center_idx] = sym; coords[center_idx] = (x,y,z); idx_seen.add(center_idx)
    return elems, coords

def _get_coords_robust(text):
    for fn in (_ultra_parse_standard_or_input,
               _ultra_parse_checkpoint,
               _ultra_parse_zmatrix_or_input,
               _ultra_parse_any_center_table):
        e,c = fn(text)
        if e and c: return e,c
    return {}, {}

# =========================
# Bond list from NBO "BD (n) X i - Y j"
# =========================
def _extract_bd_bonds_anywhere(text):
    pat = re.compile(r"BD\s*\(\s*(\d+)\s*\)\s*([A-Za-z]+)\s*(\d+)\s*-\s*([A-Za-z]+)\s*(\d+)", re.IGNORECASE)
    bonds = []
    for m in pat.finditer(text):
        order = int(m.group(1))
        xsym, xi = m.group(2).upper(), int(m.group(3))
        ysym, yj = m.group(4).upper(), int(m.group(5))
        bonds.append((order, xsym, xi, ysym, yj))
    return bonds

def _bond_graph_from_bd(bonds):
    g={}
    for order,xs,xi,ys,yj in bonds:
        g.setdefault(xi,[]).append((yj,order,ys))
        g.setdefault(yj,[]).append((xi,order,xs))
    return g

# =========================
# Public helpers used by your pipeline
# =========================
def extract_nbo_section(log_file_or_text):
    # accept path or raw text
    if os.path.exists(str(log_file_or_text)):
        content = open(log_file_or_text, 'r', encoding='utf-8', errors='ignore').read()
    else:
        content = str(log_file_or_text)
    m = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)-{30,}", content, re.DOTALL)
    return m.group(1) if m else content

def find_oh_bonds(nbo_text_or_full):
    text = nbo_text_or_full
    # O–H singles
    oh = re.findall(r"BD \(\s*1\s*\)\s*O\s*(\d+)\s*-\s*H\s*(\d+)", text)
    # H–O (convert to O,H)
    ho = re.findall(r"BD \(\s*1\s*\)\s*H\s*(\d+)\s*-\s*O\s*(\d+)", text)
    oh += [(o,h) for h,o in ho]
    uniq, seen = [], set()
    for a,b in oh:
        a,b = int(a), int(b)
        if (a,b) not in seen:
            uniq.append((a,b)); seen.add((a,b))
    return uniq

# =========================
# Core: derive C1,C2,F,G (+ coords)
# =========================
def derive_fg_from_geometry_robust(log_text, prefer_single_bonds=True):
    # 1) O–H from NBO (search in full text too)
    oh = find_oh_bonds(log_text)
    if not oh:
        return None, None, None, None, {}, {}
    O,H = oh[0]

    # 2) Topology from BD
    bonds = _extract_bd_bonds_anywhere(log_text)
    g = _bond_graph_from_bd(bonds)

    # C1 = carbon neighbor of O
    neigh_o = g.get(O, [])
    c1_cands = [(n,ordr,sym) for (n,ordr,sym) in neigh_o if sym=='C']
    if not c1_cands:
        return None, None, None, None, {}, {}
    if prefer_single_bonds:
        singles = [n for (n,ordr,_) in c1_cands if ordr==1]
        C1 = singles[0] if singles else c1_cands[0][0]
    else:
        C1 = c1_cands[0][0]

    # C2 = carbon neighbor of C1 (≠ O)
    neigh_c1 = g.get(C1, [])
    c2_cands = [(n,ordr,sym) for (n,ordr,sym) in neigh_c1 if n != O and sym=='C']
    if not c2_cands:
        c2_cands = [(n,ordr,sym) for (n,ordr,sym) in neigh_c1 if n != O]
    if not c2_cands:
        return C1, None, None, None, {}, {}
    pool = [n for (n,ordr,sym) in c2_cands if (not prefer_single_bonds) or ordr==1]
    if not pool: pool = [n for (n,_,_) in c2_cands]
    C2 = sorted(pool, key=lambda n: len(g.get(n, [])), reverse=True)[0]

    # F,G = neighbors of C2 (≠ C1), prefer single-bond carbons
    neigh_c2 = g.get(C2, [])
    fg = [(n,ordr,sym) for (n,ordr,sym) in neigh_c2 if n != C1]
    single_carbons = [n for (n,ordr,sym) in fg if ordr==1 and sym=='C']
    others = [n for (n,ordr,sym) in fg if n not in single_carbons]
    ordered = single_carbons + others
    F = ordered[0] if len(ordered)>=1 else None
    G = ordered[1] if len(ordered)>=2 else None

    elems, coords = _get_coords_robust(log_text)
    return C1, C2, F, G, elems, coords

# =========================
# Regression utilities (unchanged structure)
# =========================
def _normalize_ar_value(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}: return np.nan
    if re.match(r"^-?\d+\.0$", s): s = s[:-2]
    return s

def _canon_ar_cols(df: pd.DataFrame):
    ar_cols_raw = [c for c in df.columns if re.fullmatch(r"[Aa][Rr]\s*\d+", str(c).strip())]
    mapping = {}
    for c in ar_cols_raw:
        n = re.search(r"\d+", str(c)).group(0)
        mapping[c] = f"Ar{n}"
    if mapping:
        df.rename(columns=mapping, inplace=True)
    return [mapping.get(c, c) for c in ar_cols_raw]

def _list_log_basenames(log_folder: str):
    return {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(log_folder, "*.log"))}

# ---- Minimal feature extractors you already rely on ----
def extract_homo_lumo(log_file):
    content = open(log_file,'r',encoding='utf-8',errors='ignore').read()
    matches = re.findall(r"Population.*?SCF [Dd]ensity.*?(\s+Alpha.*?)\n\s*Condensed", content, re.DOTALL)
    if not matches: return None, None
    scf = matches[-1]
    energies_alpha = [re.findall(r"([-+]?\d*\.\d+|\d+)", s_part) for s_part in scf.split("Alpha virt.", 1)]
    if len(energies_alpha)!=2: return None, None
    occ, unocc = [list(map(float, e)) for e in energies_alpha]
    return (max(occ) if occ else None, min(unocc) if unocc else None)

def extract_dipole_moment(log_file):
    content = open(log_file,'r',encoding='utf-8',errors='ignore').read()
    m = re.findall(r"Dipole moment \(field-independent basis, Debye\):.*?(X=.*?Tot=.*?)\n", content, re.DOTALL)
    if not m: return None
    tot = re.search(r"Tot=\s*([-+]?\d*\.\d+|\d+)", m[-1])
    return float(tot.group(1)) if tot else None

def extract_polarizability(log_file):
    content = open(log_file,'r',encoding='utf-8',errors='ignore').read()
    m = re.findall(r"Exact polarizability:\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", content)
    if not m: return None
    a = m[-1]; vals = [float(a[i]) for i in [0,2,5]]
    return sum(vals)/len(vals)

def extract_nbo_values(log_file, c1, c2, a):
    content = open(log_file,'r',encoding='utf-8',errors='ignore').read()
    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)-{30,}", content, re.DOTALL)
    if not match: return None
    nbo = match.group(1)
    pat = {
        "C1-O": rf"BD \(   1\) C\s+{c1}\s+-\s+O\s+{a}\s+([\d\.]+)\s+([-\d\.]+)",
        "C1-C2": rf"BD \(   1\) C\s+{c2}\s+-\s+C\s+{c1}\s+([\d\.]+)\s+([-\d\.]+)"
    }
    occ_C1O=eng_C1O=occ_C1C2=eng_C1C2=None,None,None,None
    for key, p in pat.items():
        m = re.search(p, nbo)
        if m:
            if key=="C1-O":
                occ_C1O, eng_C1O = float(m.group(1)), float(m.group(2))
            else:
                occ_C1C2, eng_C1C2 = float(m.group(1)), float(m.group(2))
    return occ_C1O, eng_C1O, occ_C1C2, eng_C1C2

def extract_coordinates(log_file, c1, c2):
    coords = {}
    inside = False
    with open(log_file,'r',errors='ignore') as f:
        for line in f:
            if "Standard orientation" in line:
                inside = True; continue
            if inside:
                if "----" in line: continue
                parts = line.split()
                if len(parts)==6 and parts[0].isdigit() and parts[1].isdigit():
                    center = int(parts[0]); Z=int(parts[1])
                    x,y,z = map(float, parts[3:])
                    if Z==6: coords[center]=(x,y,z)
    if c1 in coords and c2 in coords:
        p1,p2 = coords[c1], coords[c2]
        d = _dist(p1,p2)
        return p1,p2,d
    return None, None, None

def extract_nbo_charges(log_file, c1, c2, a):
    lines = open(log_file,'r',encoding='utf-8',errors='ignore').readlines()
    summary_index=None
    for i in range(len(lines)-1,-1,-1):
        if "Summary of Natural Population Analysis" in lines[i]:
            summary_index=i; break
    if summary_index is None: return None,None,None,None
    charges = {}
    for line in lines[summary_index:]:
        m = re.match(r'\s*(\w+)\s+(\d+)\s+([-\d\.]+)', line)
        if m:
            atom, num, charge = m.groups()
            charges[f"{atom}{num}"] = float(charge)
    Ar_NBO_C1 = charges.get(f"C{c1}")
    Ar_NBO_C2 = charges.get(f"C{c2}")
    Ar_NBO_O1 = charges.get(f"O{a-1}")
    Ar_NBO_O2 = charges.get(f"O{a}")
    return Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2

# =========================
# Regression / model search and plot (same interface as你原本)
# =========================
def compute_loocv_metrics(X, y):
    n = X.shape[0]
    Xd = np.hstack([np.ones((n,1)), X])
    XtX = Xd.T @ Xd + 1e-8*np.eye(Xd.shape[1])
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ Xd.T @ y
    H = Xd @ XtX_inv @ Xd.T
    h = np.diag(H)
    y_pred = Xd @ beta
    y_loo = (y_pred - h*y) / (1 - h)
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    ss_res_loo = np.sum((y - y_loo)**2)
    return {
        "r2_full": 1 - ss_res/ss_tot if ss_tot>0 else 0.0,
        "q2_loocv": 1 - ss_res_loo/ss_tot if ss_tot>0 else 0.0,
        "rmse": np.sqrt(np.mean((y - y_loo)**2)),
        "coefficients": beta[1:].tolist(),
        "intercept": beta[0]
    }

def evaluate_combinations(data, target, feature_set):
    try:
        X = data[feature_set].astype(float).values
        y = data[target].values
        out = compute_loocv_metrics(X, y)
        out["features"] = feature_set
        return out
    except Exception as e:
        print(f"[ERROR] {feature_set} failed: {e}")
        return None

def search_best_models(data, features, target, max_features, r2_threshold=0.7,
                       save_csv=True, csv_path="regression_search_results.csv", verbose=True):
    all_results = []
    for k in range(1, max_features+1):
        if verbose: print(f"\n🔍 Testing {k}-feature combinations")
        for c in combinations(features, k):
            result = evaluate_combinations(data, target, list(c))
            if result and result["r2_full"]>=r2_threshold:
                all_results.append(result)
                if verbose: print(f"✅ {list(c)} | R²={result['r2_full']:.3f} | Q²={result['q2_loocv']:.3f}")
            elif verbose:
                print(f"❌ {list(c)} | skipped")
    if not all_results:
        print("⚠️ No valid models found."); return [], None
    df_all = pd.DataFrame(all_results); df_all["num_features"]=df_all["features"].apply(len)
    if save_csv: df_all.to_csv(csv_path, index=False); 
    best_model = df_all.sort_values(by="q2_loocv", ascending=False).iloc[0].to_dict()
    if verbose: print(f"\n🏆 Best: {best_model['features']} | Q²={best_model['q2_loocv']:.3f} | R²={best_model['r2_full']:.3f}")
    return df_all.to_dict(orient="records"), best_model

def plot_best_regression(target, df, best_model, savepath='Regression_Plot.png'):
    X_columns = best_model['features']
    coefficients = np.array(best_model['coefficients'])
    intercept = best_model['intercept']
    y_actual = df[target]; X_values = df[X_columns].values
    y_pred = np.dot(X_values, coefficients) + intercept
    formula = f'{target} = {" + ".join([f"{c:.2f}({f})" for c, f in zip(coefficients, X_columns)])} + {intercept:.2f}'
    print("Regression Formula:", formula)
    fig, ax = plt.subplots(figsize=(8,7))
    ax.set_facecolor('w'); ax.plot(y_actual, y_actual, color='k')
    ax.scatter(y_actual, y_pred, edgecolor='b', facecolor='b', alpha=0.7)
    ax.set_ylabel(f'Predicted {target}', fontsize=18, color='k')
    ax.set_xlabel(f'Experimental {target}', fontsize=18, color='k')
    ax.spines['bottom'].set_color('k'); ax.spines['left'].set_color('k')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.text(0.55, 0.35, f'$R^2= {best_model["r2_full"]:.2f}$', fontsize=16)
    fig.text(0.55, 0.30, f'rmse = {best_model["rmse"]:.2f}', fontsize=16)
    fig.text(0.55, 0.25, f'$Q^2= {best_model["q2_loocv"]:.2f}$ (LOO)', fontsize=16)
    fig.text(0.55, 0.20, f'{len(y_actual)} data points', fontsize=16, style='italic')
    fig.tight_layout(); plt.savefig(savepath, bbox_inches='tight'); plt.show()

# =========================
# F/G Automatic completion and flip repair
# =========================
def fill_missing_ring_FG(df, resolver, store_mode='index', flip_on_missing=True, verbose=True):
    import numpy as np, pandas as pd

    need = [c for c in df.columns if c.endswith(("_Ar_f","_Ar_g"))]
    if not all(c in df.columns for c in need):
        for c in need:
            if c not in df.columns: df[c] = np.nan

    def _safe_int(x):
        try: return int(x)
        except: return None

    def _ang(coords, H,O,C1,F):
        import math
        if coords and all(i in coords for i in [H,O,C1,F]) and F:
            def v(a,b): return (b[0]-a[0], b[1]-a[1], b[2]-a[2])
            def dot(u,v): return u[0]*v[0]+u[1]*v[1]+u[2]*v[2]
            def cross(u,v): return (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
            b1=v(p2:=coords[O],p1:=coords[H]); b2=v(p2,coords[C1]); b3=v(coords[F],coords[C1])
            bn=(dot(b2,b2))**0.5
            if bn==0: return np.nan
            b2u=(b2[0]/bn,b2[1]/bn,b2[2]/bn)
            v1=cross(b1,b2u); v2=cross(b3,b2u)
            x=dot(v1,v2); y=dot(cross(v1,b2u),v2)
            return math.degrees(math.atan2(y,x))
        return np.nan

    updated = 0
    for i,row in df.iterrows():
        comp = str(row['Compound'])
        path = resolver(comp)
        try:
            text = open(path,'r',encoding='utf-8',errors='ignore').read()
        except Exception:
            if verbose: print(f"[skip] {comp}: log not found"); continue

        try:
            oh = find_oh_bonds(text)
            if not oh: continue
            O,H = oh[0]
            C1,C2,F,G,elems,coords = derive_fg_from_geometry_robust(text)
            if not elems or not coords: continue

            # 判斷缺值 → 補齊
            if pd.isna(row['Ar1_Ar_f']) or pd.isna(row['Ar1_Ar_g']):
                df.at[i,'Ar1_Ar_f'] = float(F) if F else np.nan
                df.at[i,'Ar1_Ar_g'] = float(G) if G else np.nan

            # 如果 Ar2 缺值 → 先用 azo 另一側補
            both = derive_fg_for_both_rings_azo(text)
            right, left = both.get('right',{}), both.get('left',{})
            if pd.isna(row['Ar2_Ar_f']) and right.get('F'):
                df.at[i,'Ar2_Ar_f'] = float(right['F'])
            if pd.isna(row['Ar2_Ar_g']) and right.get('G'):
                df.at[i,'Ar2_Ar_g'] = float(right['G'])

            # 若還缺 → 翻面修復
            if flip_on_missing and (pd.isna(row['Ar1_Ar_f']) or pd.isna(row['Ar2_Ar_f'])):
                if left.get('F') and pd.isna(row['Ar1_Ar_f']):
                    df.at[i,'Ar1_Ar_f'] = float(left['F'])
                if left.get('G') and pd.isna(row['Ar1_Ar_g']):
                    df.at[i,'Ar1_Ar_g'] = float(left['G'])
                if right.get('F') and pd.isna(row['Ar2_Ar_f']):
                    df.at[i,'Ar2_Ar_f'] = float(right['F'])
                if right.get('G') and pd.isna(row['Ar2_Ar_g']):
                    df.at[i,'Ar2_Ar_g'] = float(right['G'])

            updated += 1
        except Exception as e:
            if verbose: print(f"[fail] {comp}: {e}")
            continue

    if verbose: print(f"[fill_missing_ring_FG] updated rows: {updated}")
    return df

def run_full_pipeline(log_folder, xlsx_path, max_features=3, target='ln(kobs)'):
    
    import os
    import pandas as pd
    import numpy as np

    df = pd.read_excel(xlsx_path)
    print(f"[run_full_pipeline] Loaded {len(df)} samples from: {os.path.basename(xlsx_path)}")

    try:
        # resolver: 'A13' -> '<log_folder>/13.log'
        def resolver(compound):
            num = int(str(compound).lstrip("A"))
            return os.path.join(log_folder, f"{num}.log")

        if 'fill_missing_ring_FG' in globals():
            _need_cols = [c for c in df.columns if c.endswith(("_Ar_f","_Ar_g"))]
            _missing_before = int(df[_need_cols].isna().any(axis=1).sum()) if _need_cols else -1

            df = fill_missing_ring_FG(df, resolver, store_mode='index', verbose=False)

            _need_cols = [c for c in df.columns if c.endswith(("_Ar_f","_Ar_g"))]
            _missing_after = int(df[_need_cols].isna().any(axis=1).sum()) if _need_cols else -1
            print(f"[run_full_pipeline] AutoFill f/g missing: { _missing_before } → { _missing_after }")
        else:
            print("[run_full_pipeline] fill_missing_ring_FG not found; skip auto-fill.")
    except Exception as e:
        print(f"[run_full_pipeline] AutoFill error (skipped): {e}")

    numeric_cols = [c for c in df.columns if (c not in ['Compound', target]) and (df[c].dtype != 'O')]
    features = numeric_cols
    print(f"[run_full_pipeline] Feature count: {len(features)}")

    if 'search_best_models' not in globals():
        raise RuntimeError("search_best_models() not found in this module.")

    all_results, best_model = search_best_models(
        data=df,
        features=features,
        target=target,
        max_features=max_features,
        r2_threshold=0.0,
        save_csv=True,
        csv_path="regression_search_results_full.csv",
        verbose=True
    )

    if 'plot_best_regression' in globals():
        plot_best_regression(target, df, best_model, savepath="Regression_Plot_full.png")
    else:
        print("[run_full_pipeline] plot_best_regression() not found; skip plotting.")

    return df, all_results, best_model
