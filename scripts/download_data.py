"""Download the Palmer Penguins demo dataset into data/penguins.csv.

Source: https://github.com/allisonhorst/palmerpenguins (CC0).
Run: uv run python scripts/download_data.py [--force]
"""

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/"
    "inst/extdata/penguins.csv"
)
EXPECTED_HEADER = (
    "species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year"
)
DEST = Path(__file__).resolve().parents[1] / "data" / "penguins.csv"


def main(force: bool = False) -> Path:
    if DEST.exists() and not force:
        print(f"{DEST} already exists (use --force to re-download)")
        return DEST
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=30) as resp:
        text = resp.read().decode("utf-8")
    if text.splitlines()[0].strip() != EXPECTED_HEADER:
        raise SystemExit("Unexpected file contents; refusing to save.")
    DEST.write_text(text, encoding="utf-8", newline="\n")
    print(f"Saved {len(text.splitlines()) - 1} rows to {DEST}")
    return DEST


if __name__ == "__main__":
    main(force="--force" in sys.argv)
