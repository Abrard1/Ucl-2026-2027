from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import load_matches, load_domestic_matches, canonical_team
FIXTURES = ROOT / "data" / "fixtures.csv"

def main():
    fixtures = pd.read_csv(FIXTURES)
    ucl = load_matches()
    domestic = load_domestic_matches()
    known = set(ucl["home_team"].map(canonical_team)) | set(ucl["away_team"].map(canonical_team))
    known |= set(domestic["home_team"].map(canonical_team)) | set(domestic["away_team"].map(canonical_team))
    missing = []
    for _, row in fixtures.iterrows():
        for side in ("home_team", "away_team"):
            original = str(row[side])
            key = canonical_team(original)
            if key not in known:
                missing.append((original, key))
    print("Fixtures:", len(fixtures))
    print("Unique normalised club identities in data:", len(known))
    if missing:
        print("Fixture teams with no historical match record after normalisation:")
        for name, key in missing:
            print(f"  {name!r} -> {key!r}")
    else:
        print("All fixture teams have at least one record in UCL/domestic data.")
    print("Current UCL results imported:", len(ucl[ucl["season"].astype(str) == "2026-27"]))

if __name__ == "__main__":
    main()
