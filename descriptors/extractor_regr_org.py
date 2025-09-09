# extractor_regr.py

# ====== [Auto-install morfeus-ml if missing, and force restart Colab/Jupyter] ======
import sys
import subprocess

def ensure_morfeus():
    try:
        import morfeus
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

from itertools import combinations
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# If using morfeus, need to install it
try:
    from morfeus import read_xyz, Sterimol
    from morfeus.utils import get_radii
except ImportError:
    pass

# ============ 1. Parameter Extraction (log + xlsx) ============

def extract_homo_lumo(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
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
    with open(log_file, 'r', encoding='utf-8') as f:
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
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    matches = re.findall(r"Exact polarizability:\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)\s+([-+]?\d*\.\d+|\d+)", content)
    if not matches:
        return None
    last_polarizability = matches[-1]
    values = [float(last_polarizability[i]) for i in [0, 2, 5]]
    avg_polarizability = sum(values) / len(values)
    return avg_polarizability

def extract_nbo_section(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)(-+\n)", content, re.DOTALL)
    if not match:
        return None
    return match.group(1)

def find_oh_bonds(nbo_section):
    oh_bonds = re.findall(r"BD \(\s*1\s*\)\s*O\s*(\d+)\s*-\s*H\s*(\d+)", nbo_section)
    return [(int(a), int(b)) for a, b in oh_bonds]

def find_c1_c2(nbo_section, oh_bond_atoms):
    last_found = (None, None, None, None, None, None, None)

    for a, b in oh_bond_atoms:
        c_candidates = re.findall(rf"BD \(\s*1\s*\)\s*C\s*(\d+)\s*-\s*O\s*{a}", nbo_section)

        for c in c_candidates:
            c = int(c)
            o_d_candidates = re.findall(rf"BD \(\s*[12]\s*\)\s*C\s*{c}\s*-\s*O\s*(\d+)", nbo_section)

            for d in o_d_candidates:
                d = int(d)
                e_candidates = re.findall(rf"BD \(\s*1\s*\)\s*C\s*(\d+)\s*-\s*C\s*{c}", nbo_section)

                for e in e_candidates:
                    e = int(e)

                    # Search for bonds connected to e
                    bond_types = re.findall(rf"BD \(\s*(1|2)\s*\)\s*(\w+)\s*(\d+)\s*-\s*(\w+)\s*(\d+)", nbo_section)

                    bond_pairs = {}
                    e_neighbors = []

                    for bond_type, atom1, num1, atom2, num2 in bond_types:
                        num1, num2 = int(num1), int(num2)

                        # Only record bonds related to e
                        if num1 == e or num2 == e:
                            other = num2 if num1 == e else num1
                            e_neighbors.append((bond_type, other))

                            bond_pair = frozenset([num1, num2])
                            if bond_pair not in bond_pairs:
                                bond_pairs[bond_pair] = set()
                            bond_pairs[bond_pair].add(bond_type)

                    # Count single and double bonds
                    single_count = sum("1" in types for types in bond_pairs.values())
                    double_count = sum("2" in types for types in bond_pairs.values())

                    # Always keep the last found result even if it does not match conditions
                    last_found = (c, e, a, b, d, None, None)

                    if single_count >= 2 and double_count >= 1:
                        # Further find f and g
                        f, g = None, None
                        single_neighbors = [n for t, n in e_neighbors if t == "1"]
                        double_neighbors = [n for t, n in e_neighbors if t == "2"]

                        for neighbor in single_neighbors:
                            if f is None:
                                f = neighbor
                            elif g is None and neighbor != f:
                                g = neighbor

                        for neighbor in double_neighbors:
                            if g is None or neighbor == f:
                                g = neighbor

                        print(f"Found C1: {c}, C2: {e}, A: {a}, B: {b}, D: {d}, F: {f}, G: {g}")
                        return c, e, a, b, d, f, g

    # If nothing matched, return last found result
    if last_found[0] is not None:
        print(f"[WARN] No C1-C2 pairs with the required bonding pattern found, returning last found values: {last_found}")
        return last_found

    # No results at all
    return None, None, None, None, None, None, None

def extract_nbo_values(log_file, c1, c2, a):
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
    match = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)-{30,}", content, re.DOTALL)
    if not match:
        return None
    nbo_section = match.group(1)
    bond_patterns = {
        "C1-O": rf"BD \(   1\) C\s+{c1}\s+-\s+O\s+{a}\s+([\d\.]+)\s+([-\d\.]+)",
        "C1-C2": rf"BD \(   1\) C\s+{c2}\s+-\s+C\s+{c1}\s+([\d\.]+)\s+([-\d\.]+)"
    }
    occupancy_C1_O = occupancy_C1_C2 = None
    energy_C1_O = energy_C1_C2 = None
    for key, pattern in bond_patterns.items():
        match = re.search(pattern, nbo_section)
        if match:
            if key == "C1-O":
                occupancy_C1_O = float(match.group(1))
                energy_C1_O = float(match.group(2))
            elif key == "C1-C2":
                occupancy_C1_C2 = float(match.group(1))
                energy_C1_C2 = float(match.group(2))
    return occupancy_C1_O, energy_C1_O, occupancy_C1_C2, energy_C1_C2

def extract_coordinates(log_file, c1, c2):
    coordinates = {}
    inside_standard_orientation = False
    with open(log_file, 'r') as file:
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
    with open(log_file, 'r', encoding='utf-8') as f:
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
    with open(log_file, 'r', encoding='utf-8') as f:
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
            except Exception as e:
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
    with open(log_path, "r") as f:
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
        except Exception as e:
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

        # Allow some index to be None, but explicitly report and set to None
        try:
            if any(pd.isna(x) for x in [row["Ar_a"], row["Ar_b"], row["Ar_d"], row["Ar_c"], row["Ar_e"]]):
                print(f"  [WARN] Some atom index columns are NaN: Ar_a={row['Ar_a']}, Ar_b={row['Ar_b']}, Ar_d={row['Ar_d']}, Ar_c={row['Ar_c']}, Ar_e={row['Ar_e']}; sterimol set to None for this molecule.")
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
    XtX_inv = np.linalg.inv(X_design.T @ X_design)
    beta = XtX_inv @ X_design.T @ y
    H = X_design @ XtX_inv @ X_design.T
    h = np.diag(H)
    y_pred = X_design @ beta
    y_loo = (y_pred - h * y) / (1 - h)
    ss_total = np.sum((y - np.mean(y))**2)
    ss_res_loocv = np.sum((y - y_loo)**2)
    ss_res_full = np.sum((y - y_pred)**2)
    return {
        "r2_full": 1 - ss_res_full / ss_total,
        "q2_loocv": 1 - ss_res_loocv / ss_total,
        "rmse": np.sqrt(np.mean((y - y_loo)**2)),
        "coefficients": beta[1:].tolist(),
        "intercept": beta[0]
    }

def evaluate_combinations(data, target, feature_set):
    try:
        X = data[feature_set].astype(float).values  # ✅ 強制轉為 float 避免 @ 矩陣乘法出錯
        y = data[target].values
        result = compute_loocv_metrics(X, y)
        result["features"] = feature_set
        return result if result["r2_full"] > 0.7 else None
    except Exception as e:
        print(f"[ERROR] Combo {feature_set} failed: {e}")
        return None

def search_best_models(data, features, target, max_features=5, r2_threshold=0.7,
                                    save_csv=True, csv_path="regression_search_results.csv", verbose=True):

    ar1_features = [f for f in features if f.startswith("Ar1_")]
    ar2_features = [f for f in features if f.startswith("Ar2_")]
    enforce_balanced = bool(ar1_features) and bool(ar2_features)

    all_results = []

    for k in range(1, max_features + 1):
        if verbose:
            print(f"\n🔍 Testing {k}-feature combinations")

        if enforce_balanced and k >= 2:
            # 強制 Ar1 + Ar2 平衡
            for ar1_k in range(1, k):
                ar2_k = k - ar1_k
                ar1_combs = list(combinations(ar1_features, ar1_k))
                ar2_combs = list(combinations(ar2_features, ar2_k))
                for a1 in ar1_combs:
                    for a2 in ar2_combs:
                        comb = list(a1) + list(a2)
                        result = evaluate_combinations(data, target, comb)
                        if result and result["r2_full"] >= r2_threshold:
                            all_results.append(result)
                            if verbose:
                                print(f"✅ {comb} | R² = {result['r2_full']:.3f} | Q² = {result['q2_loocv']:.3f}")
                        elif verbose:
                            print(f"❌ {comb} | skipped")
        else:
            # 一般全面組合
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

    # 🔹 建立包含所有資料的 DataFrame，同時加入作為判斷依據的 Compound, Ar1, Ar2 欄位
    # 確保這些欄位存在於原始的 df 之中
    plot_df = pd.DataFrame({
        'Compound': df['Compound'],
        'Ar1': df['Ar1'],
        'Ar2': df['Ar2'],
        'y_actual': y_actual, 
        'y_pred': y_pred
    })

    # 🔹 根據 Compound, Ar1, Ar2 這三個欄位來去除重複的點
    plot_df = plot_df.drop_duplicates(subset=['Compound', 'Ar1', 'Ar2'])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_facecolor('w')
    ax.plot(plot_df['y_actual'], plot_df['y_actual'], color='k')
    ax.scatter(plot_df['y_actual'], plot_df['y_pred'], edgecolor='b', facecolor='b', alpha=0.7)
    ax.set_ylabel(f'Predicted {target}', fontsize=18, color='k')
    ax.set_xlabel(f'Experimental {target}', fontsize=18, color='k')
    ax.spines['bottom'].set_color('k')
    ax.spines['left'].set_color('k')
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig.text(0.13, 0.95, f'{target} = {" + ".join([f"{c:.2f}({f})" for c, f in zip(coefficients, X_columns)])} + {intercept:.2f}', fontsize=10)
    fig.text(0.55, 0.35, f'$R^2= {best_model["r2_full"]:.2f}$', fontsize=16)
    fig.text(0.55, 0.30, f'rmse = {best_model["rmse"]:.2f}', fontsize=16)
    fig.text(0.55, 0.25, f'$Q^2= {best_model["q2_loocv"]:.2f}$ (LOO)', fontsize=16)
    fig.text(0.55, 0.20, f'{len(plot_df)} unique data points', fontsize=16, style='italic')
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
        # If no log_file column, try to fill it automatically
        if log_folder is not None and "log_file" not in problem_rows.columns:
            problem_rows = problem_rows.copy()
            problem_rows["log_file"] = problem_rows["Ar"].apply(lambda ar: f"{log_folder}/{ar}.log")
            print("\nCorresponding log_file:")
            print(problem_rows[["Ar", "log_file"]])
        # Export
        problem_rows.to_excel("problem_index_report.xlsx", index=False)
        print("\nSaved as problem_index_report.xlsx for manual checking!")



# ============ 5. Main Pipeline =============

def run_full_pipeline(log_folder, xlsx_path, target="ln(kobs)",
                      output_path="final_output.xlsx", plot_path='Regression_Plot.png',
                      auto_pairing=True):
    print(f"\n[STEP1] Read Excel: {xlsx_path}")
    df = pd.read_excel(xlsx_path)

    # ========== STEP 0: 建立唯一 Ar 特徵表 (unique_ar_df) ==========
    print(f"\n[STEP2] Extracting log features for each unique Ar...")
    unique_ar_df = df[['Ar']].drop_duplicates().copy()
    unique_ar_df["log_path"] = unique_ar_df["Ar"].apply(lambda ar: os.path.join(log_folder, f"{ar}.log"))
    unique_ar_df["log_exists"] = unique_ar_df["log_path"].apply(os.path.exists)
    unique_ar_df = unique_ar_df[unique_ar_df["log_exists"]].reset_index(drop=True)

    # 萃取 log features for each unique Ar
    for index, row in unique_ar_df.iterrows():
        ar = row["Ar"]
        log_file = row["log_path"]
        print(f"\n==== [{index+1}/{len(unique_ar_df)}] [{ar}] Processing log: {log_file} ====")

        try:
            avg_polar = extract_polarizability(log_file)
            homo, lumo = extract_homo_lumo(log_file)
            dipole_moment = extract_dipole_moment(log_file)
            nbo_content = extract_nbo_section(log_file)

            Ar_c = Ar_e = Ar_a = None
            Ar_NBO_C2 = Ar_NBO_O1 = Ar_NBO_O2 = Ar_v_C_O = Ar_I_C_O = L_C1_C2 = None
            Ar_b = Ar_d = Ar_f = Ar_g = None

            if nbo_content:
                oh_atoms = find_oh_bonds(nbo_content)
                c1, c2, a, b, d, f, g = find_c1_c2(nbo_content, oh_atoms)
                Ar_c, Ar_e, Ar_a, Ar_b, Ar_d, Ar_f, Ar_g = c1, c2, a, b, d, f, g
                if c1 and c2 and a:
                    try:
                        occupancy_C1_O, energy_C1_O, occupancy_C1_C2, energy_C1_C2 = extract_nbo_values(log_file, c1, c2, a)
                    except:
                        pass
                    try:
                        Ar_NBO_C1, Ar_NBO_C2, Ar_NBO_O1, Ar_NBO_O2 = extract_nbo_charges(log_file, c1, c2, a)
                    except:
                        pass
                    try:
                        Ar_I_C_O, Ar_v_C_O = extract_frequencies(log_file, Ar_c, Ar_d)
                    except:
                        pass
                    try:
                        coord_C1, coord_C2, L_C1_C2 = extract_coordinates(log_file, c1, c2)
                    except:
                        pass

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

    # 加入 Sterimol
    unique_ar_df = add_sterimol_to_df(unique_ar_df, log_folder)
    report_index_problems(unique_ar_df, log_folder)

    # ✅ 新增：過濾掉任何關鍵特徵為 NaN 的 Ar
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

    # ========== STEP 3: 合併特徵 ==========
    print(f"\n[STEP3] Merging features into main dataframe")

    if auto_pairing:
        df = df.merge(unique_ar_df.add_prefix("Ar1_"), left_on="Ar1", right_on="Ar1_Ar", how="left")
        df = df.merge(unique_ar_df.add_prefix("Ar2_"), left_on="Ar2", right_on="Ar2_Ar", how="left")
        df = df.drop(columns=["Ar1_Ar", "Ar2_Ar"])
        features = [col for col in df.columns if col.startswith("Ar1_") or col.startswith("Ar2_")]
    else:
        # 1找出會重疊的欄位（除了 key）
        overlap_cols = [col for col in unique_ar_df.columns if col in df.columns and col != "Ar"]
        # 2️先把 df 中這些重疊欄位刪掉
        df = df.drop(columns=overlap_cols)
        # 3️再合併，只保留 unique_ar_df 的同名欄位
        df = df.merge(unique_ar_df, on="Ar", how="left")
        features = essential_cols

    df.to_excel(output_path, index=False)

    # ========== STEP 4: 建模 ==========
    print(f"\n[STEP4] Performing regression modeling")

    df = df.dropna(subset=features + [target])
    if df.empty:
        print("⚠️ 無可用資料進行回歸")
        return df, [], {}

    results, best_model = search_best_models(
        data=df,
        features=features,
        target=target,
        max_features=5,
        r2_threshold=0.7,
        save_csv=True,
        csv_path="regression_search_results.csv",
        verbose=True
    )

    if best_model:
        plot_best_regression(target, df, best_model, plot_path)
    else:
        print("⚠️ 沒有符合條件的模型，跳過繪圖")

    print(f"\n✅ Analysis complete!")

    return df, results, best_model





