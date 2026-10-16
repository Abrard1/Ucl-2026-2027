"""Quick data-quality audit for the UCL and domestic match CSV files."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def audit(path: Path, kind: str) -> None:
    print(f"\n=== {kind}: {path.relative_to(ROOT)} ===")
    if not path.exists():
        print("Not found yet.")
        return
    df = pd.read_csv(path)
    print(f"Rows: {len(df):,}")
    print("Columns:", ", ".join(df.columns))
    if "date" in df.columns:
        parsed = pd.to_datetime(df["date"], errors="coerce")
        print(f"Date range: {parsed.min()} to {parsed.max()}")
        print(f"Missing/unparseable dates: {int(parsed.isna().sum()):,}")
    if "season" in df.columns:
        seasons = sorted(df["season"].dropna().astype(str).unique())
        print(f"Season labels: {len(seasons)}")
        print("First seasons:", ", ".join(seasons[:5]))
        print("Last seasons:", ", ".join(seasons[-5:]))
    required = ["home_team", "away_team", "home_goals", "away_goals"]
    if all(col in df.columns for col in required):
        goals_h = pd.to_numeric(df["home_goals"], errors="coerce")
        goals_a = pd.to_numeric(df["away_goals"], errors="coerce")
        invalid_scores = int((goals_h.isna() | goals_a.isna() | (goals_h < 0) | (goals_a < 0)).sum())
        print(f"Invalid/missing scores: {invalid_scores:,}")
        keys = [c for c in ["date", "league", "season", "home_team", "away_team", "home_goals", "away_goals"] if c in df.columns]
        print(f"Exact duplicate rows by match keys: {int(df.duplicated(subset=keys).sum()):,}")
        print(f"Unique home teams: {df['home_team'].nunique():,}; unique away teams: {df['away_team'].nunique():,}")


if __name__ == "__main__":
    audit(ROOT / "data" / "matches.csv", "Champions League")
    audit(ROOT / "data" / "domestic_matches.csv", "Domestic leagues")
