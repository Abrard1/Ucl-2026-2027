"""Import only UEFA Champions League main-competition results from OpenFootball.

The upstream repository contains several European competitions and uses separate
files for each competition. This importer deliberately reads ONLY each season's
cl.txt file, never every *.txt file. Qualifiers (clq.txt) are kept out of the
main UCL training/evaluation dataset to avoid mixing qualification matches with
the league/knockout phase.

Run from the project root:
    python3 scripts/import_openfootball.py
"""
from __future__ import annotations

import re
import shutil
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "champions-league"
OUT_FILE = ROOT / "data" / "matches.csv"
BACKUP_FILE = ROOT / "data" / "matches_mixed_backup.csv"
MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1
)}
DAY_RE = re.compile(
    r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:/|\s+)(\d{1,2})\b",
    re.I,
)
MATCH_RE = re.compile(
    r"^\s*(?:(?P<time>\d{1,2}:\d{2})\s+)?(?P<home>.+?)\s+v\s+(?P<away>.+?)\s+(?P<hg>\d{1,2})\s*[-–]\s*(?P<ag>\d{1,2})(?=\s|$|\()",
    re.I,
)
SEASON_RE = re.compile(r"(\d{4})-(\d{2})")
EXPECTED_RE = re.compile(r"^\s*#\s*Matches\s+(\d+)\s*$", re.I | re.M)


def season_info(path: Path) -> tuple[int, int, str] | None:
    # The season is the folder immediately containing cl.txt, e.g. 2024-25.
    m = SEASON_RE.search(path.parent.name)
    if not m:
        m = SEASON_RE.search(str(path.parent))
    if not m:
        return None
    start = int(m.group(1))
    end = (start // 100) * 100 + int(m.group(2))
    if end < start:
        end += 100
    return start, end, f"{start}-{str(end)[-2:]}"


def parse_file(path: Path) -> tuple[list[dict], int | None]:
    info = season_info(path)
    if info is None:
        print(f"WARNING: can't identify season folder for {path}")
        return [], None
    start_year, end_year, season = info
    text = path.read_text(encoding="utf-8", errors="replace")
    expected_match = EXPECTED_RE.search(text)
    expected = int(expected_match.group(1)) if expected_match else None
    current_date = None
    rows: list[dict] = []

    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        dm = DAY_RE.match(line)
        if dm:
            month = MONTHS[dm.group(1).title()]
            day = int(dm.group(2))
            year = start_year if month >= 7 else end_year
            try:
                current_date = date(year, month, day).isoformat()
            except ValueError:
                current_date = None
            continue
        mm = MATCH_RE.match(line)
        if not mm or current_date is None:
            continue
        home = re.sub(r"\s+", " ", mm.group("home")).strip(" \t")
        away = re.sub(r"\s+", " ", mm.group("away")).strip(" \t")
        if not home or not away:
            continue
        rows.append({
            "date": current_date,
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_goals": int(mm.group("hg")),
            "away_goals": int(mm.group("ag")),
            "competition": "UEFA Champions League",
            "source_file": str(path.relative_to(RAW_DIR)),
        })
    return rows, expected


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(
            "OpenFootball data not found. Run from the project root:\n"
            "git clone https://github.com/openfootball/champions-league.git data/raw/champions-league"
        )

    # CRITICAL: exact basename filter; do not use rglob('*.txt'). The repository
    # also contains Europa League / Conference League files and qualification files.
    files = sorted(path for path in RAW_DIR.rglob("cl.txt") if path.name.lower() == "cl.txt")
    if not files:
        raise SystemExit(f"No season cl.txt files found under {RAW_DIR}")

    rows: list[dict] = []
    expected_by_season: dict[str, int | None] = {}
    parsed_by_season: Counter = Counter()
    mismatches: list[str] = []
    for path in files:
        season = season_info(path)[2] if season_info(path) else path.parent.name
        parsed, expected = parse_file(path)
        rows.extend(parsed)
        parsed_by_season[season] += len(parsed)
        expected_by_season[season] = expected
        if expected is not None and len(parsed) != expected:
            mismatches.append(f"{season}: source header says {expected}, parser found {len(parsed)}")

    if not rows:
        raise SystemExit("No completed Champions League matches parsed. Check repository files and parser format.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["date", "home_goals", "away_goals", "home_team", "away_team"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df = df.drop_duplicates(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df = df.sort_values(["date", "home_team"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists() and not BACKUP_FILE.exists():
        shutil.copy2(OUT_FILE, BACKUP_FILE)
        print(f"Backed up previous file to: {BACKUP_FILE.relative_to(ROOT)}")
    df.to_csv(OUT_FILE, index=False)

    print(f"Read {len(files)} cl.txt files (main Champions League competition only).")
    print(f"Imported {len(df):,} completed UCL matches.")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("Matches by season:")
    print(df.groupby("season").size().to_string())
    if mismatches:
        print("\nWARNING: parsed count differs from the source header for these seasons:")
        for item in mismatches:
            print(" -", item)
        print("Review those seasons before relying on the backtest.")
    print(f"Saved clean main-competition dataset to: {OUT_FILE.relative_to(ROOT)}")
    print("Qualifiers (clq.txt), Europa League, and Conference League are not included in this file.")


if __name__ == "__main__":
    main()
