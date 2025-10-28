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

# ==== BEGIN: geometry fallback parsers ====
def _z_to_sym(Z):
    # 常用元素表（可自行擴充）
    table = {1:'H',6:'C',7:'N',8:'O',9:'F',15:'P',16:'S',17:'Cl',35:'Br',53:'I'}
    return table.get(Z, 'X')

def _parse_standard_or_input(text):
    """
    讀取最後一個 'Standard orientation:' 或 'Input orientation:' 區塊。
    回傳: (elements: dict[idx->symbol], coords: dict[idx->(x,y,z)])
    """
    import re
    pattern = (
        r"(Standard|Input)\s+orientation:\s*?\n\s*-+\n"
        r"\s*Center\s+Atomic\s+Atomic\s+Coordinates\s+\(Angstroms\)\s*\n"
        r"\s*Number\s+Number\s+Type\s+X\s+Y\s+Z\s*\n\s*-+\n"
        r"([\s\S]+?)\n\s*-+\n"
    )
    elems, coords = {}, {}
    blocks = list(re.finditer(pattern, text, re.IGNORECASE))
    if not blocks:
        return elems, coords
    block = blocks[-1].group(2)
    for line in block.strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            try:
                idx = int(parts[0])
                Z = int(parts[1])
                x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                elems[idx] = _z_to_sym(Z)
                coords[idx] = (x, y, z)
            except Exception:
                pass
    return elems, coords

def _parse_checkpoint_structure(text):
    """
    讀取 'Structure from the checkpoint file' 的座標。
    回傳: (elements: dict[idx->symbol], coords: dict[idx->(x,y,z)])
    """
    import re
    m = re.search(r"Structure\s+from\s+the\s+checkpoint\s+file[\s\S]{0,2000}?Coordinates[\s\S]+?\n", text, re.IGNORECASE)
    if not m:
        return {}, {}
    elems, coords = {}, {}
    idx = 1
    for ln in text[m.end():].splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.match(r"^-{3,}|^={3,}", s) or re.match(r"^(Charge|Multiplicity|Standard|Input)\b", s):
            break
        parts = s.split()
        try:
            # 支援 "n Sym x y z" 或 "Sym x y z"
            if len(parts) >= 5 and parts[0].isdigit():
                sym = parts[1]; x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            else:
                sym = parts[0]; x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            elems[idx] = sym
            coords[idx] = (x, y, z)
            idx += 1
        except Exception:
            pass
    return elems, coords

def _last_standard_orientation(text):
    # 先試 Standard/Input，失敗再試 checkpoint
    elems, coords = _parse_standard_or_input(text)
    if elems and coords:
        return elems, coords
    return _parse_checkpoint_structure(text)
# ==== END: geometry fallback parsers ====

# ==== BEGIN: ultra-robust coordinate parsers ====
def _ultra_parse_standard_or_input(text):
    """
    超寬鬆：擷取最後一個 'Standard orientation:' 或 'Input orientation:' 表格，
    放寬對抬頭與橫線的要求；回傳 (elements: dict[idx->sym], coords: dict[idx->(x,y,z)])
    """
    import re
    # 找所有可能的表頭
    head_re = re.compile(r"(Standard|Input)\s+orientation\s*:\s*", re.IGNORECASE)
    hits = [m.start() for m in head_re.finditer(text)]
    if not hits:
        return {}, {}
    start = hits[-1]
    tail = text[start:]
    # 從抬頭後開始，找到 "Center ... X Y Z" 的那一行，再往下收集數據行
    lines = tail.splitlines()
    header_idx = None
    for i, ln in enumerate(lines[:200]):  # 限制在抬頭後的前幾百行內找表頭
        s = ln.strip()
        if ("Center" in s and "Atomic" in s and "Coordinates" in s and "X" in s and "Y" in s and "Z" in s):
            header_idx = i
            break
    if header_idx is None:
        return {}, {}
    # 從 header_idx 往下找數據行，直到遇到明顯結束（全空、下一個抬頭、或長分隔線）
    import math
    elems, coords = {}, {}
    Z2E = {1:'H',6:'C',7:'N',8:'O',9:'F',15:'P',16:'S',17:'Cl',35:'Br',53:'I'}
    idx_seen = set()
    for ln in lines[header_idx+1:header_idx+1+1000]:
        s = ln.strip()
        if not s:
            if len(coords) >= 1:
                break
            else:
                continue
        if (s.lower().startswith("standard") or s.lower().startswith("input")) and "orientation" in s.lower():
            break
        if set(s) <= set("-= "):  # 分隔線
            if len(coords) >= 1:
                break
            else:
                continue
        parts = s.split()
        # 常見格式: center_idx  atomicZ  atomicType  x  y  z
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            try:
                center_idx = int(parts[0])
                Z = int(parts[1])
                x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                if center_idx not in idx_seen:
                    elems[center_idx] = Z2E.get(Z,'X')
                    coords[center_idx] = (x,y,z)
                    idx_seen.add(center_idx)
            except:
                pass
        # 退而求其次：有些表沒有 center_idx/atomicZ（很少見），這裡不處理
    return elems, coords

def _ultra_parse_checkpoint(text):
    import re
    m = re.search(r"Structure\s+from\s+the\s+checkpoint\s+file[\s\S]{0,2000}?Coordinates", text, re.IGNORECASE)
    if not m:
        return {}, {}
    # 從 m.end() 往下抓到下一個抬頭/分隔
    lines = text[m.end():].splitlines()
    elems, coords = {}, {}
    idx = 1
    for ln in lines:
        s = ln.strip()
        if not s:
            if len(coords) >= 1:
                break
            else:
                continue
        if set(s) <= set("-= "):  # 分隔線
            if len(coords) >= 1:
                break
            else:
                continue
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
        except:
            pass
    return elems, coords

def _ultra_parse_any_center_table(text):
    """
    萬一前兩種都抓不到，這裡嘗試找任意包含 Center/Atomic/X/Y/Z 的表格（最後一個），
    放寬到只要偵測到數值列就收。
    """
    import re
    # 找出所有包含關鍵字的表頭位置
    pat = re.compile(r"(Center.*Atomic.*Coordinates.*X.*Y.*Z)", re.IGNORECASE)
    spans = [m.span() for m in pat.finditer(text)]
    if not spans:
        return {}, {}
    start = spans[-1][1]
    lines = text[start:].splitlines()
    elems, coords = {}, {}
    Z2E = {1:'H',6:'C',7:'N',8:'O',9:'F',15:'P',16:'S',17:'Cl',35:'Br',53:'I'}
    idx_seen = set()
    for ln in lines[:1200]:
        s = ln.strip()
        if not s:
            if len(coords) >= 1:
                break
            else:
                continue
        if set(s) <= set("-= "):
            if len(coords) >= 1:
                break
            else:
                continue
        parts = s.split()
        # 儘可能容忍不同欄位數；只要末三個是 x y z 就收，前兩個盡可能當 center_idx 與 Z
        if len(parts) >= 4:
            try:
                x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            except:
                continue
            center_idx = None; Z = None; sym = None
            # 嘗試讀 center_idx 與 Z
            if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
                center_idx = int(parts[0]); Z = int(parts[1]); sym = Z2E.get(Z,'X')
            else:
                # 猜測 symbol 在最前面
                if parts[0].isalpha():
                    sym = parts[0]; center_idx = (max(idx_seen) + 1) if idx_seen else 1
                else:
                    center_idx = (max(idx_seen) + 1) if idx_seen else 1; sym = 'X'
            elems[center_idx] = sym
            coords[center_idx] = (x,y,z)
            idx_seen.add(center_idx)
    return elems, coords

def _get_coords_robust(text):
    """
    嘗試三種解析器，回傳 (elems, coords)；若都失敗，回傳兩個空 dict。
    """
    e,c = _ultra_parse_standard_or_input(text)
    if e and c:
        return e,c
    e,c = _ultra_parse_checkpoint(text)
    if e and c:
        return e,c
    e,c = _ultra_parse_any_center_table(text)
    return e,c
# ==== END: ultra-robust coordinate parsers ====

def _extract_bd_bonds_anywhere(text):
    import re
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

def derive_fg_from_geometry_robust(log_text, prefer_single_bonds=True):
    """
    先用 BD 鍵結在 NBO 索引空間找 C1/C2/F/G（不依賴座標）；
    若能再取得座標（索引一致），即可用來算二面角。
    回傳: (C1, C2, F, G, elems, coords)
    """
    # 1) O–H
    oh = find_oh_bonds(log_text)  # 我們前面已改成回傳全文；或此處直接用全文
    if not oh:
        return None, None, None, None, {}, {}
    O,H = oh[0]

    # 2) 用 BD 鍵結建圖（NBO 索引空間）
    bonds = _extract_bd_bonds_anywhere(log_text)
    g = _bond_graph_from_bd(bonds)

    # C1：O 的鄰碳
    neigh_o = g.get(O, [])
    c1_cands = [(n,ordr,sym) for (n,ordr,sym) in neigh_o if sym=='C']
    if not c1_cands:
        return None, None, None, None, {}, {}
    if prefer_single_bonds:
        singles = [n for (n,ordr,_) in c1_cands if ordr==1]
        C1 = singles[0] if singles else c1_cands[0][0]
    else:
        C1 = c1_cands[0][0]

    # C2：C1 的鄰碳（排除 O）
    neigh_c1 = g.get(C1, [])
    c2_cands = [(n,ordr,sym) for (n,ordr,sym) in neigh_c1 if sym=='C' and n != O]
    if not c2_cands:
        c2_cands = [(n,ordr,sym) for (n,ordr,sym) in neigh_c1 if n != O]
    if not c2_cands:
        return C1, None, None, None, {}, {}

    if prefer_single_bonds:
        singles = [n for (n,ordr,_) in c2_cands if ordr==1]
        pool = singles if singles else [n for (n,_,_) in c2_cands]
    else:
        pool = [n for (n,_,_) in c2_cands]

    C2 = sorted(pool, key=lambda n: len(g.get(n, [])), reverse=True)[0]

    # F/G：C2 的兩個鄰居（排除 C1）
    neigh_c2 = g.get(C2, [])
    fg = [(n,ordr,sym) for (n,ordr,sym) in neigh_c2 if n != C1]
    single_carbons = [n for (n,ordr,sym) in fg if ordr==1 and sym=='C']
    others = [n for (n,ordr,sym) in fg if n not in single_carbons]
    ordered = single_carbons + [n for (n,_,_) in others]
    F = ordered[0] if len(ordered)>=1 else None
    G = ordered[1] if len(ordered)>=2 else None

    # 3) 取座標（若成功，後面你算二面角就能對上）
    elems, coords = _get_coords_robust(log_text)

    return C1, C2, F, G, elems, coords

# ==== BEGIN: connectivity (若你專案內已有 _build_connectivity / COV_RAD，可用現有的) ====
# 簡化用共價半徑（Å）
_COV = {'H':0.31,'C':0.76,'N':0.71,'O':0.66,'F':0.57,'P':1.07,'S':1.05,'Cl':1.02,'Br':1.20,'I':1.39,'X':0.80}

def _dist(a,b):
    import math
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2+(a[2]-b[2])**2)

def _build_connectivity(elems, coords, scale=1.25):
    """
    以半徑和 * scale 為閾值建鄰接圖。
    回傳: graph: dict[idx -> list[(nbr_idx, 'approx_single', sym) ...]]
    """
    idxs = sorted(coords.keys())
    g = {i: [] for i in idxs}
    for i in idxs:
        for j in idxs:
            if j <= i: 
                continue
            si, sj = elems.get(i, 'X'), elems.get(j, 'X')
            ri = _COV.get(si, 0.80); rj = _COV.get(sj, 0.80)
            cutoff = scale * (ri + rj)
            if _dist(coords[i], coords[j]) <= cutoff:
                # 近似當成單鍵
                g[i].append((j, 1, sj))
                g[j].append((i, 1, si))
    return g
# ==== END: connectivity ====


# ==== BEGIN: derive F/G ====
def derive_fg_from_geometry(log_text, prefer_single_bonds=True):
    """
    從 log 文字推回 C1/C2/F/G：
    - 找 O–H / H–O → O（A），再找接 O 的碳 = C1
    - 從 C1 找鄰碳 = C2
    - 從 C2 的鄰居（排除 C1）中，優先挑單鍵碳兩個作 F、G
    回傳: (C1, C2, F, G)（找不到則為 None）
    """
    import re

    # 先抓 O–H
    nbo = log_text  # 直接用全文即可；若想沿用你的 extract_nbo_section 亦可
    oh = find_oh_bonds(nbo)
    if not oh:
        return None, None, None, None
    O, H = oh[0]  # O,H

    # 讀座標：Standard/Input → checkpoint
    elems, coords = _last_standard_orientation(log_text)
    if not coords:
        return None, None, None, None

    # 建鄰接
    graph = _build_connectivity(elems, coords)

    # C1：O 的鄰碳
    neigh_o = graph.get(O, [])
    c1_candidates = [n for (n,ordr,sym) in neigh_o if sym == 'C']
    if not c1_candidates:
        return None, None, None, None
    if prefer_single_bonds:
        singles = [n for (n,ordr,sym) in neigh_o if sym=='C' and ordr==1]
        C1 = singles[0] if singles else c1_candidates[0]
    else:
        C1 = c1_candidates[0]

    # C2：C1 的鄰碳（排除 O）
    neigh_c1 = graph.get(C1, [])
    c2_candidates = [n for (n,ordr,sym) in neigh_c1 if sym=='C' and n != O]
    if not c2_candidates:
        # 退而求其次：任何非 O 鄰居
        c2_candidates = [n for (n,ordr,sym) in neigh_c1 if n != O]
    if not c2_candidates:
        return C1, None, None, None

    if prefer_single_bonds:
        singles = [n for (n,ordr,sym) in neigh_c1 if n in c2_candidates and ordr==1]
        pool = singles if singles else c2_candidates
    else:
        pool = c2_candidates

    # 以 degree 作 tie-break（偏向環上的 C）
    C2 = sorted(pool, key=lambda n: len(graph.get(n, [])), reverse=True)[0]

    # F/G：C2 的兩個鄰居（排除 C1）
    neigh_c2 = graph.get(C2, [])
    fg = [(n,ordr,sym) for (n,ordr,sym) in neigh_c2 if n != C1]

    # 優先單鍵碳 → 其餘
    single_carbons = [n for (n,ordr,sym) in fg if ordr==1 and sym=='C']
    others = [n for (n,ordr,sym) in fg if n not in single_carbons]
    ordered = single_carbons + [n for (n,_,_) in others]

    F = ordered[0] if len(ordered) >= 1 else None
    G = ordered[1] if len(ordered) >= 2 else None

    return C1, C2, F, G
# ==== END: derive F/G ====

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


# ==== BEGIN: geometry parsers (extended) ====
def _parse_standard_or_input(text):
    """Return (elems, xyz) from the last 'Standard orientation:' or 'Input orientation:' block."""
    import re
    blocks = re.split(r'(Standard|Input)\s+orientation:', text, flags=re.IGNORECASE)
    if len(blocks) < 3:
        return None, None
    tail = blocks[-1]
    lines = tail.splitlines()
    start = None
    for k, ln in enumerate(lines):
        if re.match(r'^-+', ln.strip()):
            start = k + 2
            break
    if start is None:
        return None, None
    elems, xyz = [], []
    Z2E = {1:'H',6:'C',7:'N',8:'O',9:'F',16:'S',17:'Cl',35:'Br',53:'I'}
    for ln in lines[start:]:
        if re.match(r'^-+', ln.strip()):
            break
        parts = ln.strip().split()
        if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
            try:
                Z = int(parts[1]); x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                elems.append(Z2E.get(Z,'X')); xyz.append((x,y,z))
            except Exception:
                continue
    return (elems, xyz) if elems else (None, None)

def _parse_checkpoint_structure(text):
    """Return (elems, xyz) from 'Structure from the checkpoint file' coordinates block (best-effort)."""
    import re
    m = re.search(r"Structure\s+from\s+the\s+checkpoint\s+file[\s\S]{0,2000}?Coordinates[\s\S]+?\n", text, flags=re.IGNORECASE)
    if not m:
        return None, None
    lines = text[m.end():].splitlines()
    elems, xyz = [], []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if re.match(r'^-+|^={3,}', s) or re.match(r"^(Charge|Multiplicity|Standard|Input)\b", s):
            break
        parts = s.split()
        try:
            if len(parts) >= 5 and parts[0].isdigit():
                sym = parts[1]; x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            else:
                sym = parts[0]; x,y,z = float(parts[-3]), float(parts[-2]), float(parts[-1])
            elems.append(sym); xyz.append((x,y,z))
        except Exception:
            continue
    return (elems, xyz) if elems else (None, None)

def _last_standard_orientation(text):
    # Try Standard/Input orientation first; if absent, fall back to checkpoint structure
    elems, xyz = _parse_standard_or_input(text)
    if elems:
        return elems, xyz
    return _parse_checkpoint_structure(text)
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


# --- REPLACE THIS FUNCTION IN extractor_regr.py ---
def extract_nbo_section(log_file):
    import re
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    m = re.search(r"Natural Bond Orbitals \(Summary\):(.*?)(-+\n)", content, re.DOTALL)
    if m:
        return m.group(1)
    # Fallback: 沒有 Summary 就回傳全文，讓 BD(...) 仍可被找到
    return content

# --- REPLACE THIS FUNCTION IN extractor_regr.py ---
def find_oh_bonds(nbo_section):
    import re
    # O–H
    oh = re.findall(r"BD \(\s*1\s*\)\s*O\s*(\d+)\s*-\s*H\s*(\d+)", nbo_section)
    # H–O（換成 O,H）
    ho = re.findall(r"BD \(\s*1\s*\)\s*H\s*(\d+)\s*-\s*O\s*(\d+)", nbo_section)
    oh += [(o, h) for h, o in ho]
    # 去重 + 轉 int
    uniq, seen = [], set()
    for a,b in oh:
        a, b = int(a), int(b)
        if (a,b) not in seen:
            uniq.append((a,b)); seen.add((a,b))
    return uniq

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
                    bond_types = re.findall(rf"BD \(\s*(1|2)\s*\)\s*(\w+)\s*(\d+)\s*-\s*(\w+)\s*(\d+)", nbo_section)
                    bond_pairs = {}
                    e_neighbors = []
                    for bond_type, atom1, num1, atom2, num2 in bond_types:
                        num1, num2 = int(num1), int(num2)
                        if num1 == e or num2 == e:
                            other = num2 if num1 == e else num1
                            e_neighbors.append((bond_type, other))
                            bond_pair = frozenset([num1, num2])
                            if bond_pair not in bond_pairs:
                                bond_pairs[bond_pair] = set()
                            bond_pairs[bond_pair].add(bond_type)
                    single_count = sum("1" in types for types in bond_pairs.values())
                    double_count = sum("2" in types for types in bond_pairs.values())
                    last_found = (c, e, a, b, d, None, None)
                    if single_count >= 2 and double_count >= 1:
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
    if last_found[0] is not None:
        print(f"[WARN] No C1-C2 pairs with the required bonding pattern found, returning last found values: {last_found}")
        return last_found
    return None, None, None, None, None, None, None

def extract_nbo_values(log_file, c1, c2, a):
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
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
                    c1_geo, c2_geo, f_geo, g_geo = derive_fg_from_geometry_robust(log_text)
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
                c1_geo, c2_geo, f_geo, g_geo = derive_fg_from_geometry_robust(log_text)
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


