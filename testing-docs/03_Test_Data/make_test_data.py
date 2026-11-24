"""Create the system-test input files TD-01 to TD-14 (STP 8.6, System_Test_Cases.xlsx).

Run from the repository root with the project virtual environment:

    .venv\\Scripts\\python.exe testing-docs/03_Test_Data/make_test_data.py
    .venv\\Scripts\\python.exe testing-docs/03_Test_Data/make_test_data.py --oversize

Files are written next to this script only. The bundled samples in data/samples/ are read,
never changed. TD-14 (about 101 MB) is created only with --oversize.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # testing-docs/03_Test_Data -> repository root
BANKING = ROOT / "data" / "samples" / "banking.csv"
SEED = 42


def write(name, frame):
    frame.to_csv(HERE / name, index=False)


def resample_target(frame, target, counts, rng):
    """Rows drawn (with replacement) so that ``target`` has exactly ``counts`` per class."""
    parts = []
    for label, count in counts.items():
        pool = frame[frame[target] == label]
        parts.append(
            pool.sample(n=count, replace=count > len(pool), random_state=rng.integers(1e9))
        )
    out = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out["CustomerId"] = np.arange(15_000_001, 15_000_001 + len(out))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oversize", action="store_true", help="also create TD-14 (~101 MB)")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    banking = pd.read_csv(BANKING)

    (HERE / "TD-01_notes.txt").write_text("This is a plain text file, not a dataset.\n")
    (HERE / "TD-02_lookalike.csv.exe").write_text(banking.head(60).to_csv(index=False))
    banking.to_excel(HERE / "TD-03_banking.xlsx", index=False)
    write("TD-04_rows_49.csv", banking.head(49))
    write("TD-05_rows_50.csv", banking.head(50))
    write("TD-06_one_column.csv", banking[["Age"]].head(60))
    (HERE / "TD-07_empty.csv").write_bytes(b"")
    (HERE / "TD-08_corrupt.xlsx").write_text("this,is,not\nan,excel,file\n")
    write("TD-09_imbalanced_4to1.csv", resample_target(banking, "Exited", {0: 160, 1: 40}, rng))
    write("TD-10_imbalanced_3to1.csv", resample_target(banking, "Exited", {0: 150, 1: 50}, rng))

    single = banking.copy()
    single["Exited"] = single["Exited"].astype(str)
    single.loc[0, "Exited"] = "Unknown"
    write("TD-11_single_row_class.csv", single)

    large = banking.sample(n=40_000, replace=True, random_state=SEED).reset_index(drop=True)
    large["CustomerId"] = np.arange(20_000_001, 20_040_001)
    for column, scale in (("Age", 2.0), ("Balance", 2500.0), ("CreditScore", 15.0)):
        noise = rng.normal(0, scale, len(large))
        large[column] = (large[column] + noise).round(0 if column != "Balance" else 2)
    write("TD-12_large_40000.csv", large)

    write("TD-13_=SUM(1,1).csv", banking)

    if args.oversize:
        path = HERE / "TD-14_oversize_101MB.csv"
        block = banking.to_csv(index=False).encode()
        header, body = block.split(b"\n", 1)
        with path.open("wb") as handle:
            handle.write(header + b"\n")
            while handle.tell() < 101 * 1024 * 1024:
                handle.write(body)
    print("Test data written to", HERE)


if __name__ == "__main__":
    main()
