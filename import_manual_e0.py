#!/usr/bin/env python3
"""Import a manually downloaded Football-Data.co.uk 2026/27 EPL CSV safely.

Run from the project root:
    python3 scripts/import_manual_e0.py

Expected source: data/raw_downloads/E0.csv
Destination:     data/domestic_matches.csv
A timestamped backup is made before modifying the destination.
"""
from pathlib import Path
from datetime import date, datetime
import shutil
import sys
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / "data" / "raw_downloads" / "E0.csv"
DEST = ROOT / "data" / "domestic_matches.csv"

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
OPTIONAL_MAP = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",
    "B365H": "odds_b365_home",
    "B365D": "odds_b365_draw",
    "B365A": "odds_b365_away",
    "AvgH": "odds_avg_home",
    "AvgD": "odds_avg_draw",
    "AvgA": "odds_avg_away",
    "PSH": "odds_pinnacle_home",
    "PSD": "odds_pinnacle_draw",
    "PSA": "odds_pinnacle_away",
}


def main():
    if not SOURCE.is_file():
        print(f"ERROR: Can't find {SOURCE}")
        print("Check that raw_downloads is a FOLDER inside data and E0.csv is inside it.")
        print("Your folder should contain: data/raw_downloads/E0.csv")
        sys.exit(1)
    if not DEST.is_file():
        print(f"ERROR: Can't find existing domestic dataset: {DEST}")
        print("This script will not create a new domestic dataset from scratch.")
        sys.exit(1)

    try:
        raw = pd.read_csv(SOURCE, encoding="latin1")
    except Exception as exc:
        print(f"ERROR reading downloaded CSV: {exc}")
        sys.exit(1)

    # Normalise BOM/spacing/case in source column names while retaining a lookup.
    orig_columns = list(raw.columns)
    lookup = {str(c).replace("\ufeff", "").strip().upper(): c for c in orig_columns}
    missing = [c for c in REQUIRED if c.upper() not in lookup]
    if missing:
        print("ERROR: This doesn't look like a standard Football-Data.co.uk E0.csv.")
        print("Missing required columns:", ", ".join(missing))
        print("Columns found:", orig_columns)
        print("Open the file and confirm it is a CSV dataset, not an HTML error page.")
        sys.exit(1)

    def source_col(name):
        return raw[lookup[name.upper()]] if name.upper() in lookup else pd.Series(pd.NA, index=raw.index)

    matches = pd.DataFrame()
    matches["date"] = pd.to_datetime(source_col("Date"), dayfirst=True, errors="coerce")
    matches["season"] = "2026-27"
    matches["league"] = "English Premier League 2026/27"
    matches["home_team"] = source_col("HomeTeam").astype("string").str.strip()
    matches["away_team"] = source_col("AwayTeam").astype("string").str.strip()
    matches["home_goals"] = pd.to_numeric(source_col("FTHG"), errors="coerce")
    matches["away_goals"] = pd.to_numeric(source_col("FTAG"), errors="coerce")

    for source_name, output_name in OPTIONAL_MAP.items():
        if source_name.upper() in lookup:
            matches[output_name] = pd.to_numeric(source_col(source_name), errors="coerce")

    # Filter to completed 2026/27 season fixtures only; ignore future blank fixture rows.
    today = pd.Timestamp(date.today())
    season_start = pd.Timestamp("2026-07-01")
    matches = matches[
        matches["date"].notna()
        & matches["home_goals"].notna()
        & matches["away_goals"].notna()
        & matches["home_team"].notna()
        & matches["away_team"].notna()
        & (matches["home_team"].str.len() > 0)
        & (matches["away_team"].str.len() > 0)
        & (matches["date"] >= season_start)
        & (matches["date"] <= today)
        & (matches["home_goals"] >= 0)
        & (matches["away_goals"] >= 0)
    ].copy()

    if matches.empty:
        print("No completed 2026/27 matches were found in data/raw_downloads/E0.csv.")
        print("The existing domestic dataset was not changed.")
        print("Source columns:", orig_columns)
        print("The file may be older than expected or may not be the 2026/27 EPL CSV.")
        sys.exit(2)

    matches["date"] = matches["date"].dt.strftime("%Y-%m-%d")
    matches["home_goals"] = matches["home_goals"].astype(int)
    matches["away_goals"] = matches["away_goals"].astype(int)
    matches["round"] = pd.NA
    matches["source_file"] = "football-data.co.uk:E0.csv"

    existing = pd.read_csv(DEST, encoding="utf-8-sig")
    required_existing = {"date", "league", "home_team", "away_team", "home_goals", "away_goals"}
    if not required_existing.issubset(existing.columns):
        print("ERROR: Existing domestic_matches.csv has unexpected columns:", list(existing.columns))
        print("No changes made.")
        sys.exit(1)

    # Compare normalised identity keys so trivial case/spacing differences don't duplicate rows.
    team_aliases = {
        "man city": "manchester city",
        "man united": "manchester united",
        "nott'm forest": "nottingham forest",
        "nottm forest": "nottingham forest",
        "spurs": "tottenham hotspur",
        "brighton": "brighton & hove albion",
        "wolves": "wolverhampton wanderers",
        "west ham": "west ham united",
        "newcastle": "newcastle united",
        "bournemouth": "afc bournemouth",
        "leeds": "leeds united",
    }

    def norm(series, field=""):
        values = series.fillna("").astype(str).str.strip().str.casefold().str.replace(r"\s+", " ", regex=True)
        if field == "league":
            # OpenFootball competition labels include a trailing season, while this manual
            # importer labels the file with its season as well. Date keeps seasons distinct.
            values = values.str.replace(r"\s+20\d{2}/\d{2,4}$", "", regex=True)
        if field in {"home_team", "away_team"}:
            values = values.replace(team_aliases)
        return values

    key_cols = ["date", "league", "home_team", "away_team"]
    existing_keys = set(zip(*(norm(existing[c], c) for c in key_cols)))
    candidate_keys = list(zip(*(norm(matches[c], c) for c in key_cols)))
    is_new = [k not in existing_keys for k in candidate_keys]
    matches = matches.loc[is_new].copy()

    if matches.empty:
        print("No new matches to append: all completed EPL records from this file already exist.")
        print(f"Source file rows: {len(raw)}; completed 2026/27 rows considered: {len(candidate_keys)}")
        print("Existing domestic dataset was not changed.")
        return

    # Preserve the full destination schema; append optional stats/odds as new columns.
    combined_columns = list(existing.columns) + [c for c in matches.columns if c not in existing.columns]
    existing = existing.reindex(columns=combined_columns)
    matches = matches.reindex(columns=combined_columns)
    combined = pd.concat([existing, matches], ignore_index=True)

    # Timestamped backup before writing. Never overwrite the original without a backup.
    backup = DEST.with_name(f"domestic_matches.csv.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(DEST, backup)
    temp_dest = DEST.with_suffix(".csv.tmp")
    combined.to_csv(temp_dest, index=False)
    temp_dest.replace(DEST)

    print(f"Source rows read: {len(raw):,}")
    print(f"Completed EPL matches in 2026/27 file through {today.date()}: {len(candidate_keys):,}")
    print(f"New matches appended: {len(matches):,}")
    print(f"Domestic dataset rows: {len(existing):,} -> {len(combined):,}")
    if matches["date"].notna().any():
        print(f"Dates appended: {matches['date'].min()} to {matches['date'].max()}")
    new_stats = [c for c in OPTIONAL_MAP.values() if c in matches.columns and matches[c].notna().any()]
    print("Statistics/odds columns imported:", ", ".join(new_stats) if new_stats else "none available in source")
    print(f"Backup created: {backup.relative_to(ROOT)}")
    print(f"Saved: {DEST.relative_to(ROOT)}")
    print("Next: run python3 scripts/audit_matches.py to verify the updated dataset.")


if __name__ == "__main__":
    main()
