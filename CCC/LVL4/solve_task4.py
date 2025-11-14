from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# --- Helper: number words & temperature parsing (reuses Level 1 logic) ---

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}


def parse_number_word(text: str) -> float | None:
    text = text.lower().strip()
    if text in NUMBER_WORDS:
        return float(NUMBER_WORDS[text])
    parts = text.split("-")
    if len(parts) == 2:
        tens = NUMBER_WORDS.get(parts[0], 0)
        ones = NUMBER_WORDS.get(parts[1], 0)
        if tens > 0 and ones > 0:
            return float(tens + ones)
    return None


def parse_temperature(temp_str: str) -> float | None:
    """Parse a temperature value, handling word numbers and °F bug."""
    try:
        return float(temp_str)
    except ValueError:
        num = parse_number_word(temp_str)
        if num is None:
            return None
        return num


def load_level1_temps(path: Path) -> Dict[int, float]:
    """Load BOP -> temperature (°C) from Level 1 file.

    Temperatures that are obviously in Fahrenheit (> 60) are converted
    to Celsius.
    """
    bop_temp: Dict[int, float] = {}
    with path.open() as f:
        r = csv.reader(f)
        next(r)  # header
        for bop, t, h in r:
            tval = parse_temperature(t)
            if tval is None:
                continue
            if tval > 60:  # very likely Fahrenheit
                tval = (tval - 32.0) * 5.0 / 9.0
            bop_temp[int(bop)] = tval
    return bop_temp


# --- Load Level 4 flocks ---


def load_level4_flocks(path: Path) -> Tuple[Dict[int, List[List[int]]], Dict[int, str | None]]:
    """Load flocks and labels from level_4.in.

    Returns:
      flocks: dict[fid] -> list of BOP sequences
      labels: dict[fid] -> species name or None if "missing".
    """
    flocks: Dict[int, List[List[int]]] = defaultdict(list)
    species_per_flock: Dict[int, set[str]] = defaultdict(set)

    with path.open() as f:
        r = csv.reader(f)
        next(r)  # header
        for flock_id_str, path_str, species in r:
            fid = int(flock_id_str)
            species = species.strip()
            if path_str.strip():
                seq = list(map(int, path_str.split()))
                flocks[fid].append(seq)
            species_per_flock[fid].add(species)

    labels: Dict[int, str | None] = {}
    for fid, sset in species_per_flock.items():
        real = [s for s in sset if s.lower() != "missing"]
        labels[fid] = real[0] if real else None
    return flocks, labels


# --- Flock‑level feature engineering ---


def compute_flock_features(
    flocks: Dict[int, List[List[int]]],
    bop_temp: Dict[int, float],
) -> Dict[int, List[float]]:
    """Compute a feature vector per flock.

    Features (per flock):
      0: birds           – number of paths in flock
      1: avg_len         – average path length
      2: pal_frac        – fraction of palindromic paths
      3: outback_frac    – fraction of "out-and-back" palindromes
      4: start_end_frac  – fraction with same start & end BOP
      5: prefix_norm     – common prefix length / avg_len
      6: distinct_paths  – number of distinct full paths
      7: uniq_bops       – number of distinct BOPs in flock
      8: avg_temp        – mean temperature over flock BOPs
      9: min_temp        – min temperature over flock BOPs
     10: max_temp        – max temperature over flock BOPs
    """
    feats: Dict[int, List[float]] = {}

    for fid, paths in flocks.items():
        birds = len(paths)
        lens = [len(p) for p in paths]
        avg_len = sum(lens) / birds

        pal = 0
        outback = 0
        start_end = 0
        for p in paths:
            if p[0] == p[-1]:
                start_end += 1
            if p == p[::-1]:
                pal += 1
                half = (len(p) + 1) // 2
                if len(set(p[:half])) == half:
                    outback += 1

        pal_frac = pal / birds
        outback_frac = outback / birds
        start_end_frac = start_end / birds

        # common prefix length across all paths
        prefix_len = 0
        if len(paths) > 1:
            first = paths[0]
            prefix_len = len(first)
            for p in paths[1:]:
                i = 0
                while i < min(len(first), len(p)) and first[i] == p[i]:
                    i += 1
                prefix_len = min(prefix_len, i)
        prefix_norm = prefix_len / avg_len

        # distinct full paths
        distinct_paths = len({tuple(p) for p in paths})

        # BOP / temperature features
        bop_counts = Counter(b for p in paths for b in p)
        uniq_bops = float(len(bop_counts))
        temps = [bop_temp.get(b) for b in bop_counts]
        temps = [t for t in temps if t is not None]
        if temps:
            avg_temp = sum(temps) / len(temps)
            min_temp = min(temps)
            max_temp = max(temps)
        else:
            avg_temp = 0.0
            min_temp = 0.0
            max_temp = 0.0

        feats[fid] = [
            float(birds),
            float(avg_len),
            float(pal_frac),
            float(outback_frac),
            float(start_end_frac),
            float(prefix_norm),
            float(distinct_paths),
            uniq_bops,
            float(avg_temp),
            float(min_temp),
            float(max_temp),
        ]

    return feats


# --- Standardization, distance & metrics ---


def standardize(X: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    if not X:
        return X, [0.0] * 11, [1.0] * 11
    n_feat = len(X[0])
    means: List[float] = []
    stds: List[float] = []
    for j in range(n_feat):
        col = [row[j] for row in X]
        m = sum(col) / len(col)
        v = sum((x - m) ** 2 for x in col) / len(col)
        s = math.sqrt(v) if v > 1e-12 else 1.0
        means.append(m)
        stds.append(s)
    X_std: List[List[float]] = []
    for row in X:
        X_std.append([(row[j] - means[j]) / stds[j] for j in range(n_feat)])
    return X_std, means, stds


def dist2(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


# --- kNN & evaluation ---


def knn_predict(
    X_train: List[List[float]],
    y_train: List[str],
    x: List[float],
    k: int,
) -> str:
    dists: List[Tuple[float, str]] = []
    for xi, yi in zip(X_train, y_train):
        d = dist2(xi, x)
        dists.append((d, yi))
    dists.sort(key=lambda t: t[0])
    k_neigh = dists[:k]
    votes: Dict[str, float] = defaultdict(float)
    for d, yi in k_neigh:
        w = 1.0 / (1e-6 + math.sqrt(max(d, 0.0)))
        votes[yi] += w
    return max(votes.items(), key=lambda t: t[1])[0]


def macro_f1(y_true: List[str], y_pred: List[str]) -> float:
    labels = sorted(set(y_true))
    f1s: List[float] = []
    for lab in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp == lab)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lab and yp == lab)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lab and yp != lab)
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s) if f1s else 0.0


def tune_k_loocv(X: List[List[float]], y: List[str]) -> int:
    """Select k by leave‑one‑out CV maximizing macro‑F1.

    This uses flock‑level features only, which is fast enough for the
    number of flocks in level_4.in.
    """
    n = len(X)
    if n <= 5:
        return 1

    candidate_k = [1, 3, 5, 7, 9, 11, 15]
    best_k = candidate_k[0]
    best_f1 = -1.0

    for k in candidate_k:
        y_pred: List[str] = []
        for i in range(n):
            X_train = X[:i] + X[i + 1 :]
            y_train = y[:i] + y[i + 1 :]
            x_test = X[i]
            # For LOOCV we reuse the same standardization (global),
            # which is fine for our purposes.
            y_pred.append(knn_predict(X_train, y_train, x_test, k=k))
        f1 = macro_f1(y, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
    return best_k


# --- Main pipeline ---


def main() -> None:
    base = Path(__file__).resolve().parent

    # Load temps & flocks
    bop_temp = load_level1_temps(base / "all_data_from_level_1.in")
    flocks, labels = load_level4_flocks(base / "level_4.in")

    # Compute flock‑level features
    feats_by_fid = compute_flock_features(flocks, bop_temp)

    # Split into labelled and unknown flocks
    labelled_ids: List[int] = []
    X_labelled_raw: List[List[float]] = []
    y_labelled: List[str] = []
    unknown_ids: List[int] = []

    for fid, feat in feats_by_fid.items():
        species = labels.get(fid)
        if species is None:
            unknown_ids.append(fid)
        else:
            labelled_ids.append(fid)
            X_labelled_raw.append(feat)
            y_labelled.append(species)

    # Standardize using all labelled flocks
    X_labelled, means, stds = standardize(X_labelled_raw)

    # Tune k via LOOCV on labelled flocks
    k = tune_k_loocv(X_labelled, y_labelled)

    # Predict for unknown flocks
    flock_predictions: Dict[int, str] = {}
    for fid in unknown_ids:
        feat = feats_by_fid[fid]
        x_std = [(feat[j] - means[j]) / stds[j] for j in range(len(feat))]
        pred = knn_predict(X_labelled, y_labelled, x_std, k=k)
        flock_predictions[fid] = pred

    # Write submission file
    out_path = base / "level_4.out"
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Flock ID", "Species"])
        for fid in sorted(unknown_ids):
            w.writerow([fid, flock_predictions[fid]])

    print(f"Labelled flocks: {len(labelled_ids)}, tuned k={k}")
    print("Predicted species for missing flocks:")
    for fid in sorted(unknown_ids):
        print(f"  Flock {fid}: {flock_predictions[fid]}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
