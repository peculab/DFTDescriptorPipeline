import os
import sys
import math
import time
import csv
import re
from glob import glob
from itertools import combinations

def parse_last_orientation(path):
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    starts = [
        i for i, line in enumerate(lines)
        if "orientation:" in line and ("Standard" in line or "Input" in line)
    ]
    if not starts:
        raise ValueError("No orientation found")

    start = starts[-1]

    i = start + 1
    n = len(lines)
    while i < n and not lines[i].strip().startswith("-----"):
        i += 1
    if i >= n:
        raise ValueError("No table separator after orientation header")

    i += 4

    data = []
    while i < n and not lines[i].strip().startswith("-----"):
        parts = lines[i].split()
        if len(parts) >= 6:
            try:
                center = int(parts[0])
                atnum  = int(parts[1])
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])
                data.append((center, atnum, x, y, z))
            except ValueError:
                pass
        i += 1

    if not data:
        raise ValueError("Orientation table parsed but no atoms found")

    return data


COVALENT_RADII = {
    1: 0.31,  # H
    6: 0.76,  # C
    7: 0.71,  # N
    8: 0.66,  # O
}

def build_adjacency(data):
    coords = {i: (x, y, z) for i, at, x, y, z in data}
    atoms = {i: at for i, at, x, y, z in data}
    adj = {i: set() for i in coords}

    for i, j in combinations(coords.keys(), 2):
        xi, yi, zi = coords[i]
        xj, yj, zj = coords[j]
        d = math.dist((xi, yi, zi), (xj, yj, zj))
        ri = COVALENT_RADII.get(atoms[i], 0.7)
        rj = COVALENT_RADII.get(atoms[j], 0.7)
        if d < 1.25 * (ri + rj):
            adj[i].add(j)
            adj[j].add(i)
    return atoms, adj

# ======================================================
# find "a b c d e f g"
# ======================================================
def identify_labels(data, category):
    atoms, adj = build_adjacency(data)
    coords = {i: (x, y, z) for i, at, x, y, z in data}
    
    carboxyl_candidates = []  # (c_index, o_neighbors, e_score)
    for i, at in atoms.items():
        if at != 6:
            continue
        oxy = [j for j in adj[i] if atoms[j] == 8]
        if len(oxy) == 2:
            e_neighbors = [j for j in adj[i] if atoms[j] not in (1, 8)]
            e_score = min(e_neighbors) if e_neighbors else 9999
            carboxyl_candidates.append((i, oxy, e_score))

    if not carboxyl_candidates:
        raise ValueError("Cannot find carboxyl carbon (c)")

    if category == "indigo_aryl_alkyl":
        c, o_neighbors, _ = sorted(
            carboxyl_candidates, key=lambda t: (t[2], t[0])
        )[0]
    else:
        c, o_neighbors, _ = sorted(
            carboxyl_candidates, key=lambda t: t[0]
        )[0]

    a = b = d = None
    for o in o_neighbors:
        h_neigh = [h for h in adj[o] if atoms[h] == 1]
        if h_neigh:
            if a is None:
                a = o
                b = h_neigh[0]
        else:
            if d is None:
                d = o
    
    if not (a and b and d):
        hydrogens = [i for i, at in atoms.items() if at == 1]
        if len(o_neighbors) != 2 or not hydrogens:
            raise ValueError("Cannot assign a/b/d")

        best_dist = None
        best_o = None
        best_h = None
        for o in o_neighbors:
            x1, y1, z1 = coords[o]
            for h in hydrogens:
                x2, y2, z2 = coords[h]
                d_oh = math.dist((x1, y1, z1), (x2, y2, z2))
                if best_dist is None or d_oh < best_dist:
                    best_dist = d_oh
                    best_o = o
                    best_h = h

        if best_o is None:
            raise ValueError("Cannot assign a/b/d (no O–H found)")

        if best_dist > 2.2:
            raise ValueError(
                f"Cannot assign a/b/d (nearest O–H distance {best_dist:.2f} Å too large)"
            )

        a = best_o
        b = best_h
        d = o_neighbors[0] if o_neighbors[0] != a else o_neighbors[1]
    
    e_candidates = [j for j in adj[c] if atoms[j] not in (1, 8)]
    if not e_candidates:
        raise ValueError("Cannot find e")
    e = min(e_candidates)
    
    fg_candidates = [j for j in adj[e] if j != c and atoms[j] != 1]
    if len(fg_candidates) < 2:
        raise ValueError("Cannot find f,g")

    fg_candidates = sorted(fg_candidates)
    f, g = fg_candidates[0], fg_candidates[1]

    return dict(a=a, b=b, c=c, d=d, e=e, f=f, g=g)

def run_one_directory(log_dir, category):
    files = sorted(glob(os.path.join(log_dir, "*.log")))
    writer = csv.writer(sys.stdout)
    writer.writerow(
        ["category", "logfile", "a", "b", "c", "d", "e", "f", "g", "ok", "error", "elapsed"]
    )

    for path in files:
        logname = os.path.basename(path)
        start = time.time()
        ok = 0
        a = b = c = d = e = f = g = ""
        err = ""

        try:
            data = parse_last_orientation(path)
            labels = identify_labels(data, category)

            a, b, c, d, e, f, g = (
                labels["a"],
                labels["b"],
                labels["c"],
                labels["d"],
                labels["e"],
                labels["f"],
                labels["g"],
            )
            ok = 1
        except Exception as ex:
            err = str(ex)

        elapsed = round(time.time() - start, 3)
        writer.writerow(
            [category, logname, a, b, c, d, e, f, g, ok, err, elapsed]
        )

# ======================================================
# main
# ======================================================
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_abcefg_from_logs.py <log_dir> <category>")
        sys.exit(1)

    log_dir = sys.argv[1]
    category = sys.argv[2]
    run_one_directory(log_dir, category)