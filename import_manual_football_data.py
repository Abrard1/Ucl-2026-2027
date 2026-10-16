#!/usr/bin/env python3
"""Import manually downloaded Football-Data.co.uk 2026/27 league CSVs.

Put files such as E0.csv, SP1.csv, D1.csv in data/raw_downloads/ and run
from the project root:
    python3 scripts/import_manual_football_data.py

The script merges statistics/odds into matching existing domestic match rows
by date and normalised teams, appends new completed 2026/27 matches, avoids
exact fixture duplicates, and makes a timestamped backup before writing.
Python 3.9+ compatible.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import shutil
import sys
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw_downloads"
DEST = ROOT / "data" / "domestic_matches.csv"
SEASON_START = pd.Timestamp("2026-07-01")

# Football-Data.co.uk file codes for leagues most relevant to UCL clubs.
LEAGUES = {
    "E0": "English Premier League",
    "SC0": "Scottish Premiership",
    "D1": "German Bundesliga",
    "I1": "Italian Serie A",
    "SP1": "Primera División de España",
    "F1": "French Ligue 1",
    "N1": "Dutch Eredivisie",
    "B1": "Belgian First Division A",
    "P1": "Portuguese Primeira Liga",
    "T1": "Turkish Süper Lig",
    "G1": "Greek Super League",
}

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
STAT_MAP = {
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellow_cards", "AY": "away_yellow_cards",
    "HR": "home_red_cards", "AR": "away_red_cards",
    "B365H": "odds_b365_home", "B365D": "odds_b365_draw", "B365A": "odds_b365_away",
    "BWH": "odds_bwin_home", "BWD": "odds_bwin_draw", "BWA": "odds_bwin_away",
    "PSH": "odds_pinnacle_home", "PSD": "odds_pinnacle_draw", "PSA": "odds_pinnacle_away",
    "AvgH": "odds_avg_home", "AvgD": "odds_avg_draw", "AvgA": "odds_avg_away",
    "MaxH": "odds_max_home", "MaxD": "odds_max_draw", "MaxA": "odds_max_away",
    "BbAvH": "odds_betbrain_avg_home", "BbAvD": "odds_betbrain_avg_draw", "BbAvA": "odds_betbrain_avg_away",
    "BbMxH": "odds_betbrain_max_home", "BbMxD": "odds_betbrain_max_draw", "BbMxA": "odds_betbrain_max_away",
    "AHh": "asian_handicap_home_line", "B365>2.5": "odds_b365_over25",
    "B365<2.5": "odds_b365_under25", "P>2.5": "odds_pinnacle_over25", "P<2.5": "odds_pinnacle_under25",
}

TEAM_ALIASES = {
    "ath bilbao": "athletic club", "athletic bilbao": "athletic club",
    "athletic club bilbao": "athletic club",
    "manchester city": "manchester city", "man city": "manchester city",
    "manchester united": "manchester united", "man utd": "manchester united", "man united": "manchester united",
    "brighton": "brighton and hove albion", "brighton and hove albion": "brighton and hove albion",
    "newcastle": "newcastle united", "newcastle united": "newcastle united",
    "west ham": "west ham united", "west ham united": "west ham united",
    "wolves": "wolverhampton wanderers", "wolverhampton wanderers": "wolverhampton wanderers",
    "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur", "tottenham hotspur": "tottenham hotspur",
    "nottm forest": "nottingham forest", "notts forest": "nottingham forest", "nottingham forest": "nottingham forest",
    "bayern munchen": "bayern munich", "bayern munich": "bayern munich",
    "psv eindhoven": "psv", "psv": "psv",
    "real betis seville": "real betis", "real betis": "real betis",
    "atletico de madrid": "atletico madrid", "atletico madrid": "atletico madrid", "ath madrid": "atletico madrid",
    "inter milan": "inter", "inter milano": "inter", "inter": "inter",
    "slavia praha": "slavia prague", "slavia prague": "slavia prague",
    "bodo glimt": "bodo glimt", "bodoe glimt": "bodo glimt",
    "lens": "lens", "rc lens": "lens", "racing club de lens": "lens",
}


def canonical_team(value: object) -> str:
    text = str(value or "").strip()
    # Remove OpenFootball suffixes such as (ENG)/(ESP).
    import re
    text = re.sub(r"\s*\([A-Za-z]{3}\)\s*$", "", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    text = re.sub(r"\b(fc|cf|afc|sc|ssc|ac|as|fk|sk)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return TEAM_ALIASES.get(text, text)


def source_column(raw: pd.DataFrame, lookup: dict[str, object], name: str) -> pd.Series:
    original = lookup.get(name.upper())
    if original is None:
        return pd.Series(pd.NA, index=raw.index)
    return raw[original]


def identity_key(row: pd.Series) -> tuple[str, str, str]:
    return (
        str(row["date"]),
        canonical_team(row["home_team"]),
        canonical_team(row["away_team"]),
    )


def main() -> None:
    if not DEST.is_file():
        raise SystemExit(f"Missing existing dataset: {DEST}; no changes made.")
    if not RAW_DIR.is_dir():
        raise SystemExit(f"Missing downloads folder: {RAW_DIR}; create it and put CSVs inside.")

    paths = [p for p in sorted(RAW_DIR.glob("*.csv")) if p.stem.upper() in LEAGUES]
    if not paths:
        raise SystemExit("No recognised league files found. Expected E0.csv, SP1.csv, D1.csv, I1.csv, F1.csv, N1.csv, B1.csv, P1.csv, T1.csv, G1.csv, or SC0.csv.")

    frames: list[pd.DataFrame] = []
    today = pd.Timestamp(date.today())
    files_read = 0
    source_rows = 0
    for path in paths:
        code = path.stem.upper()
        try:
            raw = pd.read_csv(path, encoding="latin1", low_memory=False)
        except Exception as exc:
            print(f"SKIP {path.name}: couldn't read CSV ({exc})")
            continue
        lookup = {str(c).replace("\ufeff", "").strip().upper(): c for c in raw.columns}
        missing = [c for c in REQUIRED if c.upper() not in lookup]
        if missing:
            print(f"SKIP {path.name}: missing required columns {missing}. Is this the actual CSV, not an HTML page?")
            continue

        out = pd.DataFrame(index=raw.index)
        out["date"] = pd.to_datetime(source_column(raw, lookup, "Date"), dayfirst=True, errors="coerce")
        out["season"] = "2026-27"
        out["league"] = LEAGUES[code] + " 2026/27"
        out["home_team"] = source_column(raw, lookup, "HomeTeam").astype("string").str.strip()
        out["away_team"] = source_column(raw, lookup, "AwayTeam").astype("string").str.strip()
        out["home_goals"] = pd.to_numeric(source_column(raw, lookup, "FTHG"), errors="coerce")
        out["away_goals"] = pd.to_numeric(source_column(raw, lookup, "FTAG"), errors="coerce")

        imported_stats: list[str] = []
        for src, dst in STAT_MAP.items():
            if src.upper() in lookup:
                out[dst] = pd.to_numeric(source_column(raw, lookup, src), errors="coerce")
                if out[dst].notna().any():
                    imported_stats.append(dst)

        out = out[
            out["date"].notna()
            & out["home_team"].notna() & out["away_team"].notna()
            & (out["home_team"].astype(str).str.len() > 0)
            & (out["away_team"].astype(str).str.len() > 0)
            & out["home_goals"].notna() & out["away_goals"].notna()
            & (out["home_goals"] >= 0) & (out["away_goals"] >= 0)
            & (out["date"] >= SEASON_START) & (out["date"] <= today)
        ].copy()
        if out.empty:
            print(f"{path.name}: no completed 2026/27 matches through {today.date()}.")
            continue
        out["home_goals"] = out["home_goals"].astype(int)
        out["away_goals"] = out["away_goals"].astype(int)
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out["round"] = pd.NA
        out["source_file"] = "football-data.co.uk:" + path.name
        out["_file_code"] = code
        frames.append(out)
        files_read += 1
        source_rows += len(out)
        print(f"{path.name}: {len(out)} completed rows; extra stats/odds: {len(imported_stats)} columns")

    if not frames:
        raise SystemExit("No usable completed match rows found. Existing dataset was not changed.")

    incoming = pd.concat(frames, ignore_index=True, sort=False)
    existing = pd.read_csv(DEST, encoding="utf-8-sig", low_memory=False)
    required_existing = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    if not required_existing.issubset(existing.columns):
        raise SystemExit("Existing domestic CSV has unexpected columns; no changes made.")

    # Standardise dates/keys only for matching; preserve all original values.
    existing["_date_key"] = pd.to_datetime(existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    existing["_home_key"] = existing["home_team"].map(canonical_team)
    existing["_away_key"] = existing["away_team"].map(canonical_team)
    keys_to_indices: dict[tuple[str, str, str], list[int]] = {}
    for idx, row in existing.iterrows():
        if pd.notna(row["_date_key"]):
            key = (str(row["_date_key"]), str(row["_home_key"]), str(row["_away_key"]))
            keys_to_indices.setdefault(key, []).append(idx)

    appended: list[pd.Series] = []
    merged_count = 0
    duplicate_incoming = 0
    score_conflicts: list[str] = []
    seen_incoming: set[tuple[str, str, str]] = set()
    source_columns = [c for c in incoming.columns if not c.startswith("_")]
    # Ensure destination can accept the stats/odds columns before merging.
    for col in source_columns:
        if col not in existing.columns:
            existing[col] = pd.NA

    stat_cols = list(STAT_MAP.values())
    for _, row in incoming.iterrows():
        key = identity_key(row)
        if key in seen_incoming:
            duplicate_incoming += 1
            continue
        seen_incoming.add(key)
        candidates = keys_to_indices.get(key, [])
        if candidates:
            # Prefer the candidate with matching score; otherwise use first and report the conflict.
            idx = next((i for i in candidates if int(existing.at[i, "home_goals"]) == int(row["home_goals"]) and int(existing.at[i, "away_goals"]) == int(row["away_goals"])), candidates[0])
            old_hg, old_ag = existing.at[idx, "home_goals"], existing.at[idx, "away_goals"]
            try:
                conflict = int(old_hg) != int(row["home_goals"]) or int(old_ag) != int(row["away_goals"])
            except (TypeError, ValueError):
                conflict = True
            if conflict:
                score_conflicts.append(f"{key[0]} {row['home_team']} vs {row['away_team']}: existing {old_hg}-{old_ag}, Football-Data {row['home_goals']}-{row['away_goals']}")
                continue
            changed = False
            for col in stat_cols:
                if col in row.index and pd.notna(row[col]):
                    current = existing.at[idx, col]
                    if pd.isna(current) or str(current).strip() == "":
                        existing.at[idx, col] = row[col]
                        changed = True
                    # If a previous import already has a value, keep it unless missing.
            old_source = existing.at[idx, "source_file"] if "source_file" in existing.columns else pd.NA
            source_tag = str(row.get("source_file", ""))
            if "source_file" in existing.columns and source_tag and pd.notna(old_source) and source_tag not in str(old_source):
                # Provenance field is informational; append only once.
                existing.at[idx, "source_file"] = str(old_source) + ";" + source_tag
            if changed:
                merged_count += 1
        else:
            newrow = {col: row.get(col, pd.NA) for col in source_columns}
            appended.append(pd.Series(newrow))
            # Incoming rows are deduplicated by seen_incoming before further processing.

    existing = existing.drop(columns=["_date_key", "_home_key", "_away_key"], errors="ignore")
    if appended:
        addition = pd.DataFrame(appended)
        # Reindex preserves original columns and places any added columns at the end.
        all_cols = list(existing.columns) + [c for c in addition.columns if c not in existing.columns]
        existing = existing.reindex(columns=all_cols)
        addition = addition.reindex(columns=all_cols)
        combined = pd.concat([existing, addition], ignore_index=True, sort=False)
    else:
        combined = existing.copy()

    if merged_count == 0 and not appended:
        print("No new rows or missing statistics to merge. Existing dataset unchanged.")
        if score_conflicts:
            print("Score conflicts found:")
            for item in score_conflicts[:20]:
                print("  WARNING:", item)
        return

    backup = DEST.with_name(f"domestic_matches.csv.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(DEST, backup)
    temp = DEST.with_suffix(".csv.tmp")
    combined.to_csv(temp, index=False)
    temp.replace(DEST)

    print("\nSUMMARY")
    print(f"CSV files read: {files_read}")
    print(f"Completed source rows considered: {source_rows:,}")
    print(f"Existing match rows enriched with missing statistics/odds: {merged_count:,}")
    print(f"New completed matches appended: {len(appended):,}")
    print(f"Duplicate rows across downloaded files skipped: {duplicate_incoming:,}")
    print(f"Domestic dataset rows: {len(existing):,} -> {len(combined):,}")
    print(f"Backup created: {backup.relative_to(ROOT)}")
    print(f"Saved: {DEST.relative_to(ROOT)}")
    if score_conflicts:
        print(f"WARNING: {len(score_conflicts)} score conflicts; existing rows were not overwritten. First 20:")
        for item in score_conflicts[:20]:
            print("  WARNING:", item)
    populated = [c for c in stat_cols if c in combined.columns and pd.to_numeric(combined[c], errors="coerce").notna().any()]
    print("Available stats/odds columns:", ", ".join(populated) if populated else "none")
    print("Next: run python3 scripts/audit_matches.py and check the printed warnings.")


if __name__ == "__main__":
    main()
