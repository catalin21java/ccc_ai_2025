

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

LEVEL5_FILE = "all_data_from_level_5.in"
OUTPUT_FILE = "level_6.out"
SOURCE_DAY = 730
TARGET_DAY = 791


def compute_top50_for_day(base: Path, day: int) -> List[int]:
    """Return the top 50 BOP ids for a given Level‑5 day by Arrivals."""
    path = base / LEVEL5_FILE
    rows: List[Tuple[float, int]] = []  # (Arrivals, BOP)

    with path.open() as f:
        r = csv.reader(f)
        next(r)  # header
        for d_s, bop_s, arr_s, *_ in r:
            if int(d_s) != day:
                continue
            bop = int(bop_s)
            arr = float(arr_s)
            rows.append((arr, bop))

    # Sort by Arrivals desc, then BOP asc
    rows.sort(key=lambda t: (-t[0], t[1]))
    top50 = [bop for _, bop in rows[:50]]
    return top50


def main() -> None:
    base = Path(__file__).resolve().parent
    top50 = compute_top50_for_day(base, SOURCE_DAY)

    out_path = base / OUTPUT_FILE
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Day", "Top 50 Arrivals BOPs"])
        w.writerow([TARGET_DAY, " ".join(str(b) for b in top50)])

    print(f"Computed top 50 BOPs for Level‑5 day {SOURCE_DAY} and wrote as day {TARGET_DAY} to {out_path}")


if __name__ == "__main__":
    main()
