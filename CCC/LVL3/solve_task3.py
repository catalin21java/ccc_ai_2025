
import csv
from pathlib import Path

# Pre‑computed mappings from flock ID to species name
# for each Level 3 dataset.

LEVEL_3_A_MAPPING = {
    1: "Sticky Wolfthroat",
    2: "Hurracurra Bird",
    3: "Red Firefinch",
    4: "Rusty Goldhammer",
    5: "Flanking Blackfinch",
    6: "Medieval Bluetit",
}

LEVEL_3_B_MAPPING = {
    1: "Red Firefinch",
    2: "Sticky Wolfthroat",
    3: "Hurracurra Bird",
    4: "Medieval Bluetit",
    5: "Flanking Blackfinch",
    6: "Rusty Goldhammer",
}

LEVEL_3_C_MAPPING = {
    1: "Hurracurra Bird",
    2: "Flanking Blackfinch",
    3: "Rusty Goldhammer",
    4: "Medieval Bluetit",
    5: "Sticky Wolfthroat",
    6: "Red Firefinch",
}


def write_mapping(output_path: Path, mapping: dict[int, str]) -> None:
    """Write a mapping dict as a CSV file in the required format."""
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Flock ID", "Species"])
        for flock_id in sorted(mapping):
            writer.writerow([flock_id, mapping[flock_id]])


def main() -> None:
    base = Path(__file__).resolve().parent

    files_and_mappings = [
        (base / "level_3_a.out", LEVEL_3_A_MAPPING),
        (base / "level_3_b.out", LEVEL_3_B_MAPPING),
        (base / "level_3_c.out", LEVEL_3_C_MAPPING),
    ]

    for out_path, mapping in files_and_mappings:
        write_mapping(out_path, mapping)
        print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
