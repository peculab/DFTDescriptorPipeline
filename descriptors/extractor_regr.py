# extractor_regr.py

# ====== [Auto-install morfeus-ml if missing, and force restart Colab/Jupyter] ======
import sys
import subprocess

def ensure_morfeus():
    try:
        import morfeus  # noqa: F401
    except ImportError:
        print("\n[Auto-installing morfeus-ml... If you see Successfully installed below, please RESTART and rerun this script/Colab cell!]\n")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "morfeus-ml"])
        print("\n[Auto-installed morfeus-ml. Please RESTART and rerun your script/Notebook!]\n")
        import os; os._exit(0)  # Force exit for user to restart

ensure_morfeus()
# ====== [END] ======

import os
import re
import glob
import math
import numpy as np
import pandas as pd
import random
import itertools as it

from itertools import combinations
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# morfeus (optional import again for IDEs; ensure_morfeus already handled install)
try:
    from morfeus import read_xyz, Sterimol
    from morfeus.utils import get_radii
except Exception:
    pass

# ==== F/G fallback from geometry (paste near the top) ====
import math

COV_RAD = {"H":0.31,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"S":1.05,"Cl":1.02,"Br":1.20,"I":1.39}

def _dist(p, q):
    return math.sqrt((p[0]-q[0])**2+(p[1]-q[1])**2+(p[2]-q[2])**2)

def _bond_cutoff(el1, el2, scale=1.20):
    r1 = COV_RAD.get(el1, 0.77); r2 = COV_RAD.get(el2, 0.77)
    return scale*(r1+r2)

def _build_connectivity(elements, coords):
    n=len(elements); g=[set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if _dist(coords[i],coords[j]) <= _bond_cutoff(elements[i],elements[j]):
                g[i].add(j); g[j].add(i)
    return g

def _last_standard_orientation(text):
    blocks = text.split("Standard orientation:")
    if len(blocks)<2: return None, None
    tail = blocks[-1]; lines = tail.splitlines()
    start=None
    for k,ln in enumerate(lines):
        if "---------------------------------------------------------------------" in ln:
            start=k+2; break
    if start is None: return None, None
    elems, xyz=[], []
    Z2E={1:"H",6:"C",7:"N",8:"O",9:"F",16:"S",17:"Cl",35:"Br",53:"I"}
    for ln in lines[start:]:
        if "---------------------------------------------------------------------" in ln: break
        parts=ln.split()
        if len(parts)<6: continue
        atno=int(parts[1]); x,y,z = float(parts[3]), float(parts[4]), float(parts[5])
        elems.append(Z2E.get(atno,"C")); xyz.append((x,y,z))
    return elems, xyz

def _shortest_co(elements, coords, g):
    cand=[]
    for c in range(len(elements)):
        if elements[c]!="C": continue
        for o in g[c]:
            if elements[o]!="O": continue
            cand.append((_dist(coords[c],coords[o]), c, o))
    cand.sort()
    if not cand: return None, None
    _,ci,oi = cand[0]
    return ci, oi

def _find_ring6(g, start, avoid=None):
    avoid = avoid or set()
    stack=[(start,[start])]
    while stack:
        node, path = stack.pop()
        if len(path)>6: continue
        for nb in g[node]:
            if nb in avoid: continue
            if len(path)>=2 and nb==path[-2]: continue
            if nb==path[0] and 3<len(path)<=6:
                if len(path)==6: return set(path)
                else: continue
            if nb not in path: stack.append((nb, path+[nb]))
    return None

def _fg_on_ring(g, ring, c2):
    nbrs=[v for v in g[c2] if v in ring]
    if len(nbrs)>=2: return nbrs[0], nbrs[1]
    return None, None

def derive_fg_from_geometry(log_text):
    """Return (c1, c2, f, g) by geometry fallback; any not-found is None."""
    elems, xyz = _last_standard_orientation(log_text)
    if not elems: return None, None, None, None
    g = _build_connectivity(elems, xyz)
    c1, _ = _shortest_co(elems, xyz, g)
    if c1 is None: return None, None, None, None
    # pick a carbon neighbor of C1 that lies on a 6-ring → C2
    c2 = None; ring=None
    for nb in [v for v in g[c1] if elems[v]=="C"]:
        ring6 = _find_ring6(g, nb, avoid={c1})
        if ring6: c2=nb; ring=ring6; break
    if c2 is None: return c1, None, None, None
    f, g2 = _fg_on_ring(g, ring, c2)
    return c1, c2, f, g2
# ==== end F/G fallback utilities ====

# ============ Helpers (normalization, column detection, logs) ============

def _normalize_ar_value(x):
    """Normalize Ar key values for consistent merging: strip, cast to str, convert '101.0'→'101'."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.lower() in {"nan", "none", ""}:
        return np.nan
    # Convert '123.0' to '123'
    if re.match(r"^-?\d+\.0$", s):
        s = s[:-2]
    return s

def _canon_ar_cols(df: pd.DataFrame):
    """
    Detect columns like Ar1, AR2, 'Ar 3' (case/space tolerant), and rename them to canonical 'ArN' names.
    Returns the canonical list (e.g., ['Ar1','Ar2',...]).
    """
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

# ============ 1. Parameter Extraction (log + xlsx) ============

def extract_homo_lumo(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    matches = re.findall(r"Population.*?SCF [Dd]ensity.*?(\s+Alpha.*?)\n\s*Condensed", content, re.DOTALL)
    if not matches:
        return None, None
    scf_section = matches[-1]
    energies_alpha = [re.findall(r"([-+]?\d*\.\d+|\d+)", s_part) for s_part in scf_section.split("Alpha virt.", 1)]
    if len(energies_alpha) == 2:
        occupied_energies_alpha, unoccupied_energies_alpha = [list(map(float, e)) for e in energies_alpha]
        homo_alpha = max(occupied_energies_alpha) if occupied_energies_alpha else None
        lumo_alpha = min(unoccupied_energies_alpha) if unoccupied_energies_alpha else None
        return homo_alpha, lumo_alpha
    return None, None

def extract_dipole_moment(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    matches = re.findall(r"Dipole moment \(field-independent basis, Debye\):.*?(X=.*?Tot=.*?)\n", content, re.DOTALL)
    if not matches:
        return None
    last_dipole_section = matches[-1]
    tot_match = re.search(r"Tot=\s*([-+]?\d*\.\d+|\d+)", last_dipole_section)
    if tot_match:
        return float(tot_match.group(1))
    return None

def extract_polarizability(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    matches = re.findall(r"Exact polarizability:\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", content)
    if not matches:
        return None
    last_polarizability = matches[-1]
    values = [float(last_polarizability[i]) for i in [0, 2, 5]]
    avg_polarizability = sum(values) / len(values)
    return avg_polarizability

def extract_nbo_section(log_file):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # 抓到所有 Summary，回傳最後一段（最完整）
    blocks = re.findall(r"Natural Bond Orbitals \(Summary\):(.*?)(?:-{5,}\n)", content, re.DOTALL)
    if not blocks:
        return None
    return blocks[-1]

def find_oh_bonds(nbo_section):
    pairs = re.findall(r"BD \(\s*1\s*\)\s*(?:O\s*(\d+)\s*-\s*H\s*(\d+)|H\s*(\d+)\s*-\s*O\s*(\d+))", nbo_section)
    out = []
    for a,b,c,d in pairs:
        if a and b: out.append((int(a), int(b)))   # O a - H b
        elif c and d: out.append((int(d), int(c))) # H c - O d  → 轉成 (O d, H c)
    return out

def find_c1_c2(nbo_section, oh_bond_atoms):
    last_found = (None, None, None, None, None, None, None)
    for a, b in oh_bond_atoms:
        c_candidates = re.findall(
            rf"BD \(\s*1\s*\)\s*(?:C\s*(\d+)\s*-\s*O\s*{a}|O\s*{a}\s*-\s*C\s*(\d+))",
            nbo_section
        )
        c_ids = [int(x) for tup in c_candidates for x in tup if x]
        for c in c_ids:
            o_d_candidates = re.findall(
                rf"BD \(\s*[12]\s*\)\s*(?:C\s*{c}\s*-\s*O\s*(\d+)|O\s*(\d+)\s*-\s*C\s*{c})",
                nbo_section
            )
            d_ids = [int(x) for tup in o_d_candidates for x in tup if x]
            d_ids = [d for d in d_ids if d != a]          # ← 排除 OH 的氧
            for d in d_ids:
                e_candidates = re.findall(
                    rf"BD \(\s*1\s*\)\s*(?:C\s*(\d+)\s*-\s*C\s*{c}|C\s*{c}\s*-\s*C\s*(\d+))",
                    nbo_section
                )
                e_ids = [int(x) for tup in e_candidates for x in tup if x]
                for e in e_ids:
                    bond_types = re.findall(
                        rf"BD \(\s*(1|2)\s*\)\s*(\w+)\s*(\d+)\s*-\s*(\w+)\s*(\d+)",
                        nbo_section
                    )
                    bond_pairs = {}
                    e_neighbors = []
                    for bond_type, atom1, num1, atom2, num2 in bond_types:
                        num1, num2 = int(num1), int(num2)
                        if num1 == e or num2 == e:
                            other = num2 if num1 == e else num1
                            e_neighbors.append((bond_type, other))
                            bp = frozenset((num1, num2))
                            bond_pairs.setdefault(bp, set()).add(bond_type)

                    single_count = sum("1" in types for types in bond_pairs.values())
                    double_count = sum("2" in types for types in bond_pairs.values())
                    last_found = (c, e, a, b, d, None, None)

                    if single_count >= 2 and double_count >= 1:
                        # —— 只改這段挑 F/G 的邏輯 ——
                        singles = []
                        doubles = []
                        for t, n in e_neighbors:
                            if n == c:    # ← 排除側鏈碳 c
                                continue
                            if t == "1" and n not in singles:
                                singles.append(n)
                            elif t == "2" and n not in doubles:
                                doubles.append(n)

                        f = g = None
                        # 優先「單+雙」
                        if singles and doubles:
                            f = singles[0]
                            g = doubles[0] if doubles[0] != f else (doubles[1] if len(doubles) > 1 else None)
                        # 其次「單+單」
                        if g is None and len(singles) >= 2:
                            f, g = singles[0], singles[1]
                        # 再者「雙+雙」
                        if g is None and len(doubles) >= 2:
                            f, g = doubles[0], doubles[1]
                        # 最保底：從所有鄰居（已排除 c）湊兩個不同
                        if g is None:
                            pool = []
                            for _, n in e_neighbors:
                                if n != c and n not in pool:
                                    pool.append(n)
                            if pool:
                                f = pool[0]
                                g = pool[1] if len(pool) > 1 else None

                        if g == f:
                            g = next((n for n in singles + doubles if n not in (None, f, c)), None)

                        print(f"Found C1: {c}, C2: {e}, A: {a}, B: {b}, D: {d}, F: {f}, G: {g}")
                        return c, e, a, b, d, f, g

    if last_found[0] is not None:
        print(f"[WARN] No C1-C2 pairs with the required bonding pattern found, returning last found values: {last_found}")
        return last_found
    return None, None, None, None, None, None, None

def extract_nbo_values(log_file, c1, c2, a):
    nbo_section = extract_nbo_section(log_file)
    if not nbo_section:
        return None
    # 允許雙向：C1–O(a) 與 O(a)–C1；C1–C2 與 C2–C1
    pat_c1_o  = rf"BD \(\s*1\s*\)\s*(?:C\s*{c1}\s*-\s*O\s*{a}|O\s*{a}\s*-\s*C\s*{c1})\s+([\d\.]+)\s+([-\d\.]+)"
    pat_c1_c2 = rf"BD \(\s*1\s*\)\s*(?:C\s*{c1}\s*-\s*C\s*{c2}|C\s*{c2}\s*-\s*C\s*{c1})\s+([\d\.]+)\s+([-\d\.]+)"

    m1 = re.search(pat_c1_o, nbo_section)
    m2 = re.search(pat_c1_c2, nbo_section)

    occ_c1_o = ene_c1_o = occ_c1_c2 = ene_c1_c2 = None
    if m1:
        occ_c1_o  = float(m1.group(1)); ene_c1_o  = float(m1.group(2))
    if m2:
        occ_c1_c2 = float(m2.group(1)); ene_c1_c2 = float(m2.group(2))
    return occ_c1_o, ene_c1_o, occ_c1_c2, ene_c1_c2

def extract_coordinates(log_file, c1, c2):
    coordinates = {}
    inside_standard_orientation = False
    with open(log_file, 'r', errors='ignore') as file:
        for line in file:
            if "Standard orientation" in line:
                inside_standard_orientation = True
                continue
            if inside_standard_orientation:
                if "----" in line:
                    continue
                parts = line.split()
                if len(parts) == 6 and parts[0].isdigit() and parts[1].isdigit():
                    center_number = int(parts[0])
                    atomic_number = int(parts[1])
                    x, y, z = map(float, parts[3:])
                    if atomic_number == 6:
                        coordinates[center_number] = (x, y, z)
    if c1 in coordinates and c2 in coordinates:
        x1, y1, z1 = coordinates[c1]
        x2, y2, z2 = coordinates[c2]
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        return (x1, y1, z1), (x2, y2, z2), distance
    else:
        return None, None, None

def extract_nbo_charges(log_file, c1, c2, a):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.readlines()
    summary_index = None
    for i in range(len(content) - 1, -1, -1):
        if "Summary of Natural Population Analysis" in content[i]:
            summary_index = i
            break
    if summary_index is None:
        raise ValueError(f"Summary of Natural Population Analysis block NOT found in {log_file}")
    charges = {}
    for line in content[summary_index:]:
        match = re.match(r'\s*(\w+)\s+(\d+)\s+([-\d\.]+)', line)
        if match:
            atom, num, charge = match.groups()
            charges[f"{atom}{num}"] = float(charge)
    Ar_NBO_C1 = charges.get(f"C{c1}", None)
    Ar_NBO_C2 = charges.get(f"C{c2}", None)
    Ar_NBO_O1 = charges.get(f"O{a-1}", None)
    Ar_NBO_O2 = charges.get(f"O{a}", None)
    return Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2

def parse_floats(line):
    return [float(x) for x in re.findall(r'-?\d+\.\d+', line)]

def extract_frequencies(log_file, atom_c, atom_d):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.readlines()
    vib_start = None
    for i in range(len(content)):
        if "Frequencies --" in content[i]:
            vib_start = i
            break
    if vib_start is None:
        raise ValueError("Frequencies block NOT found")
    matched_frequencies = []
    i = vib_start
    while i < len(content):
        if "Frequencies --" in content[i]:
            try:
                freq_line = parse_floats(content[i])
                red_mass_line = parse_floats(content[i + 1])
                frc_consts_line = parse_floats(content[i + 2])
                ir_inten_line = parse_floats(content[i + 3])
                atom_displacements = []
                j = i + 5
                while j < len(content) and content[j].strip() and "Frequencies --" not in content[j]:
                    parts = content[j].split()
                    if len(parts) >= 11:
                        disp1 = list(map(float, parts[2:5]))
                        disp2 = list(map(float, parts[5:8]))
                        disp3 = list(map(float, parts[8:11]))
                        atom_displacements.append([disp1, disp2, disp3])
                    j += 1
                for mode_index, freq in enumerate(freq_line):
                    if 1800 <= freq <= 1900:
                        try:
                            v1 = atom_displacements[atom_c - 1][mode_index]
                            v2 = atom_displacements[atom_d - 1][mode_index]
                            disp_vec = [(a - b) for a, b in zip(v1, v2)]
                            disp_mag = sum(x**2 for x in disp_vec) ** 0.5
                            matched_frequencies.append((freq, ir_inten_line[mode_index], disp_mag))
                        except IndexError:
                            continue
                i = j
            except Exception:
                i += 1
        else:
            i += 1
    if not matched_frequencies:
        raise ValueError("No vibration modes found for atom_c and atom_d in 1800–1900 cm⁻¹ range")
    matched_frequencies.sort(key=lambda x: x[2], reverse=True)
    best_freq, best_ir, best_disp = matched_frequencies[0]
    return best_ir, best_freq

# ============ 2. Sterimol parameters (require morfeus) ============
atomic_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}

def extract_last_standard_orientation(log_path):
    with open(log_path, "r", errors='ignore') as f:
        lines = f.readlines()
    geometries = []
    block = []
    reading = False
    for line in lines:
        if "Standard orientation" in line:
            block = []
            reading = True
            continue
        if reading:
            if "-----" in line:
                continue
            if any(x in line for x in ["Center", "Atomic", "Number"]):
                continue
            if line.strip() == "":
                if block:
                    geometries.append(block)
                    block = []
                reading = False
            else:
                if re.match(r"^\s*\d+\s+\d+\s+\d+\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?\s+[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", line):
                    block.append(line)
                else:
                    if block:
                        geometries.append(block)
                        block = []
                    reading = False
    if not geometries:
        return None
    last_geom = geometries[-1]
    atoms = []
    for line in last_geom:
        parts = line.split()
        try:
            atomic_num = int(parts[1])
            x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
            symbol = atomic_symbols.get(atomic_num, None)
            if symbol is None:
                return None
            atoms.append((symbol, x, y, z))
        except Exception:
            return None
    return atoms

def write_xyz(atom_list, filename):
    with open(filename, "w") as f:
        f.write(f"{len(atom_list)}\n")
        f.write("Extracted from Gaussian log\n")
        for atom in atom_list:
            f.write(f"{atom[0]}  {atom[1]:.8f}  {atom[2]:.8f}  {atom[3]:.8f}\n")

def add_sterimol_to_df(df, log_folder):
    try:
        from morfeus import read_xyz, Sterimol
        from morfeus.utils import get_radii
    except ImportError:
        raise ImportError("morfeus library required for sterimol (please install via pip install morfeus-ml and restart the kernel).")
    df["Ar_Ster_L"] = None
    df["Ar_Ster_B1"] = None
    df["Ar_Ster_B5"] = None
    log_files = glob.glob(os.path.join(log_folder, "*.log"))
    log_map = {os.path.basename(f).replace(".log", ""): f for f in log_files}
    for idx, row in df.iterrows():
        mol_name = str(row["Ar"])
        log_path = log_map.get(mol_name)
        print(f"\n[Sterimol] [{mol_name}] log: {log_path}")
        if not log_path:
            print("  [SKIP] Log file not found")
            continue
        atoms = extract_last_standard_orientation(log_path)
        if not atoms:
            print("  [SKIP] Failed to extract atoms")
            continue
        try:
            if any(pd.isna(x) for x in [row.get("Ar_a"), row.get("Ar_b"), row.get("Ar_d"), row.get("Ar_c"), row.get("Ar_e")]):
                print(f"  [WARN] Some atom index columns are NaN: Ar_a={row.get('Ar_a')}, Ar_b={row.get('Ar_b')}, Ar_d={row.get('Ar_d')}, Ar_c={row.get('Ar_c')}, Ar_e={row.get('Ar_e')}; sterimol set to None for this molecule.")
                df.at[idx, "Ar_Ster_L"] = None
                df.at[idx, "Ar_Ster_B1"] = None
                df.at[idx, "Ar_Ster_B5"] = None
                continue
            exclude_atoms = [int(row["Ar_a"]), int(row["Ar_b"]), int(row["Ar_d"])]
            atoms_to_keep = [a for i, a in enumerate(atoms) if (i + 1) not in exclude_atoms]
            if len(atoms_to_keep) < 2:
                print("  [SKIP] atoms_to_keep < 2")
                continue
            xyz_path = f"{mol_name}_filtered.xyz"
            write_xyz(atoms_to_keep, xyz_path)
            atom1 = int(row["Ar_c"])
            atom2 = int(row["Ar_e"])
            elements, coords = read_xyz(xyz_path)
            radii = get_radii(elements, radii_type="bondi")
            radii = [1.09 if r == 1.20 else r for r in radii]
            sterimol = Sterimol(elements, coords, atom1, atom2, radii=radii)
            df.at[idx, "Ar_Ster_L"] = sterimol.L_value
            df.at[idx, "Ar_Ster_B1"] = sterimol.B_1_value
            df.at[idx, "Ar_Ster_B5"] = sterimol.B_5_value
            print(f"  [OK] Sterimol: L={sterimol.L_value}, B1={sterimol.B_1_value}, B5={sterimol.B_5_value}")
        except Exception as e:
            print(f"  [ERROR] Sterimol calculation failed: {e}")
            df.at[idx, "Ar_Ster_L"] = None
            df.at[idx, "Ar_Ster_B1"] = None
            df.at[idx, "Ar_Ster_B5"] = None
            continue
    return df

# ============ 3. Regr/Learning ==============

def prepare_data(path, features, target):
    data = pd.read_excel(path)
    data = data.dropna(subset=features + [target])
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

def compute_loocv_metrics(X, y):
    n = X.shape[0]
    X_design = np.hstack([np.ones((n, 1)), X])
    XtX = X_design.T @ X_design
    # tiny ridge for stability
    XtX += 1e-8 * np.eye(XtX.shape[0])
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X_design.T @ y
    H = X_design @ XtX_inv @ X_design.T
    h = np.diag(H)
    y_pred = X_design @ beta
    # LOO formula
    y_loo = (y_pred - h * y) / (1 - h)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_res_loocv = np.sum((y - y_loo)**2)
    ss_res_full = np.sum((y - y_pred)**2)
    return {
        "r2_full": 1 - ss_res_full / ss_total if ss_total > 0 else 0.0,
        "q2_loocv": 1 - ss_res_loocv / ss_total if ss_total > 0 else 0.0,
        "rmse": np.sqrt(np.mean((y - y_loo)**2)),
        "coefficients": beta[1:].tolist(),
        "intercept": beta[0]
    }

def evaluate_combinations(data, target, feature_set):
    """Return metrics for a given feature set; do not filter here (let search_* handle thresholds)."""
    try:
        X = data[feature_set].astype(float).values
        y = data[target].values
        result = compute_loocv_metrics(X, y)
        result["features"] = feature_set
        return result
    except Exception as e:
        print(f"[ERROR] Combo {feature_set} failed: {e}")
        return None

# ---- Combination helpers ----

def _integer_compositions_with_bounds(total, mins, maxs):
    m = len(mins)
    if sum(mins) > total or sum(maxs) < total:
        return
    def dfs(i, remain, path):
        if i == m - 1:
            x = remain
            if mins[i] <= x <= maxs[i]:
                yield tuple(path + [x])
            return
        low = max(mins[i], remain - sum(maxs[i+1:]))
        high = min(maxs[i], remain - sum(mins[i+1:]))
        for v in range(low, high + 1):
            yield from dfs(i + 1, remain - v, path + [v])
    yield from dfs(0, total, [])

def _balanced_bounds(num_groups, k):
    base = k // num_groups
    up = (k + num_groups - 1) // num_groups
    mins = [base] * num_groups
    maxs = [up] * num_groups
    return mins, maxs

def search_best_models_general(
    data,
    target,
    groups,                          # e.g. {"Ar1":[...], "Ar2":[...], "Ar3":[...]}
    max_features,
    r2_threshold=0.7,
    balance="bounds",                # default: each group must contribute at least 1
    group_bounds=None,               # if None -> each group (1, len(group))
    save_csv=True,
    csv_path="regression_search_results.csv",
    verbose=True,
    max_combinations_per_k=20000,
    random_seed=42,
):
    random.seed(random_seed)
    group_names = list(groups.keys())
    group_lists = [groups[g] for g in group_names]
    m = len(group_names)
    if m == 0:
        print("⚠️ No groups detected.")
        return [], None

    if group_bounds is None:
        group_bounds = {g: (1, len(groups[g])) for g in group_names}

    all_results = []

    for k in range(1, max_features + 1):
        if verbose:
            print(f"\n🔍 Testing {k}-feature combinations across {m} groups")

        if balance == "bounds" and k < m:
            if verbose:
                print(f"⏭️  skip k={k} (need ≥ {m} to give each group ≥1 feature)")
            continue

        if balance == "equal":
            mins, maxs = _balanced_bounds(m, k)
        else:
            mins, maxs = [], []
            for gname, glist in zip(group_names, group_lists):
                lo, hi = group_bounds.get(gname, (0, len(glist)))
                mins.append(max(0, lo))
                maxs.append(min(len(glist), hi))

        allocations = list(_integer_compositions_with_bounds(k, mins, maxs))
        if not allocations:
            if verbose:
                print(f"⚠️ No feasible allocations for k={k}.")
            continue

        combos_seen = 0
        for alloc in allocations:
            per_group_choices = []
            infeasible = False
            for glist, take in zip(group_lists, alloc):
                if take == 0:
                    per_group_choices.append([()])
                else:
                    if take > len(glist):
                        infeasible = True
                        break
                    per_group_choices.append(list(it.combinations(glist, take)))
            if infeasible:
                continue

            for tpl in it.product(*per_group_choices):
                combo = []
                for part in tpl:
                    combo.extend(list(part))
                if len(combo) != k:
                    continue

                combos_seen += 1
                if (max_combinations_per_k is not None) and (combos_seen > max_combinations_per_k):
                    if random.random() < 0.98:
                        continue

                result = evaluate_combinations(data, target, combo)
                if result and result.get("r2_full", -1) >= r2_threshold:
                    all_results.append(result)
                    if verbose:
                        print(f"✅ {combo} | R²={result['r2_full']:.3f} | Q²={result['q2_loocv']:.3f}")
                elif verbose:
                    print(f"❌ {combo} | skipped")

    if not all_results:
        print("⚠️ No valid models found.")
        return [], None

    df_all = pd.DataFrame(all_results)
    df_all["num_features"] = df_all["features"].apply(len)

    if save_csv:
        df_all.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n📄 Saved all {len(df_all)} results to {csv_path}")

    best_model = df_all.sort_values(by="q2_loocv", ascending=False).iloc[0].to_dict()
    if verbose:
        print(f"\n🏆 Best model: {best_model['features']} | Q²={best_model['q2_loocv']:.3f} | R²={best_model['r2_full']:.3f}")

    return df_all.to_dict(orient="records"), best_model

def search_best_models(data, features, target, max_features, r2_threshold=0.7,
                       save_csv=True, csv_path="regression_search_results.csv", verbose=True):
    """Legacy ungrouped exhaustive search (kept as fallback)."""
    all_results = []
    for k in range(1, max_features + 1):
        if verbose:
            print(f"\n🔍 Testing {k}-feature combinations")
        combs = list(combinations(features, k))
        for c in combs:
            result = evaluate_combinations(data, target, list(c))
            if result and result["r2_full"] >= r2_threshold:
                all_results.append(result)
                if verbose:
                    print(f"✅ {list(c)} | R² = {result['r2_full']:.3f} | Q² = {result['q2_loocv']:.3f}")
            elif verbose:
                print(f"❌ {list(c)} | skipped")
    if not all_results:
        print("⚠️ No valid models found.")
        return [], None
    df_all = pd.DataFrame(all_results)
    df_all["num_features"] = df_all["features"].apply(len)
    if save_csv:
        df_all.to_csv(csv_path, index=False)
        if verbose:
            print(f"\n📄 Saved all {len(df_all)} results to {csv_path}")
    best_model = df_all.sort_values(by="q2_loocv", ascending=False).iloc[0].to_dict()
    if verbose:
        print(f"\n🏆 Best model: {best_model['features']} | Q² = {best_model['q2_loocv']:.3f} | R² = {best_model['r2_full']:.3f}")
    return df_all.to_dict(orient="records"), best_model

# ============ 4. Regression Plot ============
def plot_best_regression(target, df, best_model, savepath='Regression_Plot.png'):
    X_columns = best_model['features']
    coefficients = np.array(best_model['coefficients'])
    intercept = best_model['intercept']
    
    y_actual = df[target]
    X_values = df[X_columns].values
    y_pred = np.dot(X_values, coefficients) + intercept

    formula = f'{target} = {" + ".join([f"{c:.2f}({f})" for c, f in zip(coefficients, X_columns)])} + {intercept:.2f}'

    print("Regression Formula:", formula)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('w')
    ax.plot(y_actual, y_actual, color='k')
    ax.scatter(y_actual, y_pred, edgecolor='b', facecolor='b', alpha=0.7)
    ax.set_ylabel(f'Predicted {target}', fontsize=18, color='k')
    ax.set_xlabel(f'Experimental {target}', fontsize=18, color='k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    fig.text(0.55, 0.35, f'$R^2= {best_model["r2_full"]:.2f}$', fontsize=16)
    fig.text(0.55, 0.30, f'rmse = {best_model["rmse"]:.2f}', fontsize=16)
    fig.text(0.55, 0.25, f'$Q^2= {best_model["q2_loocv"]:.2f}$ (LOO)', fontsize=16)
    fig.text(0.55, 0.20, f'{len(y_actual)} data points', fontsize=16, style='italic')

    fig.tight_layout()
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def report_index_problems(df, log_folder=None):
    """
    Report all molecules whose index columns are None/NaN, and save as Excel.
    """
    index_cols = ["Ar_c", "Ar_e", "Ar_a", "Ar_b", "Ar_d", "Ar_f", "Ar_g"]
    def is_any_nan_or_none(row):
        return any((x is None) or (isinstance(x, float) and np.isnan(x)) for x in row[index_cols])
    problem_rows = df[df.apply(is_any_nan_or_none, axis=1)]
    if len(problem_rows) == 0:
        print("✅ No molecules have index columns as None/NaN; all extracted correctly!")
    else:
        print("❗The following molecules have atom index as None/NaN during extraction:\n")
        print(problem_rows[["Ar"] + index_cols])
        if log_folder is not None and "log_file" not in problem_rows.columns:
            problem_rows = problem_rows.copy()
            problem_rows["log_file"] = problem_rows["Ar"].apply(lambda ar: f"{log_folder}/{ar}.log")
            print("\nCorresponding log_file:")
            print(problem_rows[["Ar", "log_file"]])
        problem_rows.to_excel("problem_index_report.xlsx", index=False)
        print("\nSaved as problem_index_report.xlsx for manual checking!")

# ============ 5. Main Pipeline =============

def run_full_pipeline(log_folder, xlsx_path, max_features, target="ln(kobs)",
                      output_path="final_output.xlsx", plot_path='Regression_Plot.png',
                      auto_pairing=True):
    print(f"\n[STEP1] Read Excel: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    print(f"\n[STEP2] Extracting log features for each unique Ar...")

    ar_cols = _canon_ar_cols(df)  # may rename to canonical Ar1, Ar2, ...
    if "Ar" in df.columns and not ar_cols:
        ar_series = df["Ar"].dropna()
    else:
        stacks = []
        for c in ar_cols:
            stacks.append(df[c].dropna())
        ar_series = pd.concat(stacks, ignore_index=True).dropna() if stacks else pd.Series([], dtype=object)

    # Normalize keys
    ar_series = ar_series.apply(_normalize_ar_value).dropna()
    unique_ar_df = pd.DataFrame({"Ar": ar_series.unique()})
    unique_ar_df["Ar"] = unique_ar_df["Ar"].apply(_normalize_ar_value)
    unique_ar_df["Ar_key"] = unique_ar_df["Ar"]
    unique_ar_df["log_path"] = unique_ar_df["Ar"].apply(lambda ar: os.path.join(log_folder, f"{ar}.log"))
    unique_ar_df["log_exists"] = unique_ar_df["log_path"].apply(os.path.exists)
    unique_ar_df = unique_ar_df[unique_ar_df["log_exists"]].reset_index(drop=True)

    if unique_ar_df.empty:
        print("⚠️ No valid Ar entries with matching log files. Stop.")
        return df, [], {}

    for index, row in unique_ar_df.iterrows():
        ar = row["Ar"]
        log_file = row["log_path"]
        print(f"\n==== [{index+1}/{len(unique_ar_df)}] [{ar}] Processing log: {log_file} ====")
        try:
            avg_polar = extract_polarizability(log_file)
            homo, lumo = extract_homo_lumo(log_file)
            dipole_moment = extract_dipole_moment(log_file)
            nbo_content = extract_nbo_section(log_file)
            with open(log_file, "r", errors="ignore") as fh:
                log_text = fh.read() 

            Ar_c = Ar_e = Ar_a = None
            Ar_NBO_C2 = Ar_NBO_O1 = Ar_NBO_O2 = Ar_v_C_O = Ar_I_C_O = L_C1_C2 = None
            Ar_b = Ar_d = Ar_f = Ar_g = None

            if nbo_content:
                oh_atoms = find_oh_bonds(nbo_content)
                c1, c2, a, b, d, f, g = find_c1_c2(nbo_content, oh_atoms)
                Ar_c, Ar_e, Ar_a, Ar_b, Ar_d, Ar_f, Ar_g = c1, c2, a, b, d, f, g

                if (Ar_f is None) or (Ar_g is None):
                    c1_geo, c2_geo, f_geo, g_geo = derive_fg_from_geometry(log_text)
                    if (Ar_c is None) and (c1_geo is not None): Ar_c = c1_geo
                    if (Ar_e is None) and (c2_geo is not None): Ar_e = c2_geo
                    if (Ar_f is None) and (f_geo  is not None): Ar_f = f_geo
                    if (Ar_g is None) and (g_geo  is not None): Ar_g = g_geo

                if c1 and c2 and a:
                    try:
                        occupancy_C1_O, energy_C1_O, occupancy_C1_C2, energy_C1_C2 = extract_nbo_values(log_file, c1, c2, a)
                    except Exception:
                        pass
                    try:
                        Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2 = extract_nbo_charges(log_file, c1, c2, a)
                    except Exception:
                        pass
                    try:
                        Ar_I_C_O, Ar_v_C_O = extract_frequencies(log_file, Ar_c, Ar_d)
                    except Exception:
                        pass
                    try:
                        coord_C1, coord_C2, L_C1_C2 = extract_coordinates(log_file, c1, c2)
                    except Exception:
                        pass
            else:
                c1_geo, c2_geo, f_geo, g_geo = derive_fg_from_geometry(log_text)
                Ar_c, Ar_e, Ar_f, Ar_g = c1_geo, c2_geo, f_geo, g_geo

            unique_ar_df.at[index, "Ar_NBO_C2"] = Ar_NBO_C2
            unique_ar_df.at[index, "Ar_NBO_=O"] = Ar_NBO_O1
            unique_ar_df.at[index, "Ar_NBO_-O"] = Ar_NBO_O2
            unique_ar_df.at[index, "Ar_v_C=O"] = Ar_v_C_O
            unique_ar_df.at[index, "Ar_I_C=O"] = Ar_I_C_O
            unique_ar_df.at[index, "Ar_dp"] = dipole_moment
            unique_ar_df.at[index, "Ar_polar"] = avg_polar
            unique_ar_df.at[index, "Ar_LUMO"] = lumo
            unique_ar_df.at[index, "Ar_HOMO"] = homo
            unique_ar_df.at[index, "L_C1_C2"] = L_C1_C2
            unique_ar_df.at[index, "Ar_c"] = Ar_c
            unique_ar_df.at[index, "Ar_e"] = Ar_e
            unique_ar_df.at[index, "Ar_a"] = Ar_a
            unique_ar_df.at[index, "Ar_b"] = Ar_b
            unique_ar_df.at[index, "Ar_d"] = Ar_d
            unique_ar_df.at[index, "Ar_f"] = Ar_f
            unique_ar_df.at[index, "Ar_g"] = Ar_g
        except Exception as e:
            print(f"[ERROR] Error occurred while processing Ar={ar}: {e}")
            continue

    unique_ar_df = add_sterimol_to_df(unique_ar_df, log_folder)
    report_index_problems(unique_ar_df, log_folder)

    essential_cols = [
        "Ar_NBO_C2", "Ar_NBO_=O", "Ar_NBO_-O", "Ar_v_C=O", "Ar_I_C=O", "Ar_dp",
        "Ar_polar", "Ar_LUMO", "Ar_HOMO", "L_C1_C2",
        "Ar_Ster_L", "Ar_Ster_B1", "Ar_Ster_B5"
    ]
    before_drop = len(unique_ar_df)
    unique_ar_df = unique_ar_df.dropna(subset=essential_cols)
    after_drop = len(unique_ar_df)
    print(f"🧹 Dropped {before_drop - after_drop} Ar rows with missing essential features")

    unique_ar_df.to_excel("unique_ar_features.xlsx", index=False)

    print(f"\n[STEP3] Merging features into main dataframe")

    ar_cols = _canon_ar_cols(df)  # e.g., ['Ar1','Ar2',...]
    logs_in_folder = _list_log_basenames(log_folder)

    if not ar_cols and "Ar" in df.columns:
        df["Ar"] = df["Ar"].apply(_normalize_ar_value)
        df = df.merge(unique_ar_df, on="Ar", how="left")
        meta_exclude = {"Ar", "Ar_key", "log_path", "log_exists"}
        features = [c for c in unique_ar_df.columns if c not in meta_exclude]
    else:
        for c in ar_cols:
            df[c] = df[c].apply(_normalize_ar_value)
        for c in ar_cols:
            pref = f"{c}_"
            right = unique_ar_df.add_prefix(pref)
            df = df.merge(right, left_on=c, right_on=pref + "Ar_key", how="left")
            for m in (pref + "Ar", pref + "Ar_key", pref + "log_path", pref + "log_exists"):
                if m in df.columns:
                    df.drop(columns=m, inplace=True)
        feature_exclude_suffixes = ("_Ar", "_Ar_key", "_log_path", "_log_exists")
        features = [c for c in df.columns
                    if re.match(r"^Ar\d+_", c) and not any(c.endswith(suf) for suf in feature_exclude_suffixes)]

    print(f"🔎 Detected Ar columns in sheet: {ar_cols if ar_cols else ['Ar']}")
    total_unique = unique_ar_df.shape[0]
    print(f"🗂️ Unique Ar with matching log files: {total_unique}")
    print(f"🧩 Total feature columns found: {len(features)}")

    if len(features) == 0:
        missing = set()
        if ar_cols:
            seen = set()
            for c in ar_cols:
                seen.update({str(_normalize_ar_value(x)) for x in df[c].dropna().unique().tolist()})
            missing = seen - logs_in_folder
        elif "Ar" in df.columns:
            seen = set(map(str, df["Ar"].dropna().unique()))
            missing = seen - logs_in_folder
        if missing:
            print(f"⚠️ No feature columns produced. Examples of missing logs (first 5): {sorted(list(missing))[:5]}")

    df.to_excel(output_path, index=False)

    print(f"\n[STEP4] Performing regression modeling")

    if not features:
        print("⚠️ No available feature columns, terminating process.")
        return df, [], {}

    df_model = df.dropna(subset=features + [target])
    if df_model.empty:
        print("⚠️ No data available for regression (data exhausted after dropping missing values).")
        return df, [], {}

    # Automatically create groups based on the descriptor prefix (e.g., Ar1_... becomes group Ar1).
    groups = {}
    for col in features:
        g = col.split("_", 1)[0]
        groups.setdefault(g, []).append(col)

    # Show feature count per group.
    for g, cols in groups.items():
        print(f"   • {g}: {len(cols)} features")

    # Fallback: If no groups are detected (theoretically unlikely, but for safety).
    if not groups:
        print("⚠️ No groups detected. Falling back to ungrouped exhaustive search.")
        results, best_model = search_best_models(
            data=df_model,
            features=features,
            target=target,
            max_features=max_features,
            r2_threshold=0.7,
            save_csv=True,
            csv_path="regression_search_results.csv",
            verbose=True
        )
        if best_model:
            plot_best_regression(target, df_model, best_model, plot_path)
        else:
            print("⚠️ No valid model found, skipping plot.")
        print("\n✅ Analysis complete!")
        return df, results, best_model

    # Formal: Group-Constrained Regression (Balance across groups, e.g., 1-3 features per group, bounds are user-customizable).
    group_bounds = {g: (1, min(3, len(cols))) for g, cols in groups.items()}
    results, best_model = search_best_models_general(
        data=df_model,
        target=target,
        groups=groups,
        max_features=max_features,
        r2_threshold=0.7,
        balance="bounds",
        group_bounds=group_bounds,
        save_csv=True,
        csv_path="regression_search_results.csv",
        verbose=True,
        max_combinations_per_k=20000
    )

    if best_model:
        plot_best_regression(target, df_model, best_model, plot_path)
    else:
        print("⚠️ No valid model found, skipping plot.")

    print(f"\n✅ Analysis complete!")
    return df, results, best_model





