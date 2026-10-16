"""Convert local OpenFootball Champions League .txt seasons to data/matches.csv.

Setup:
  git clone https://github.com/openfootball/champions-league.git data/raw/champions-league
  python scripts/import_openfootball.py
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "champions-league"
OUT_FILE = ROOT / "data" / "matches.csv"
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
DAY_RE = re.compile(r"^\s*(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)(?:/|\s+)(\d{1,2})\b", re.I)
MATCH_RE = re.compile(
    r"^\s*(?:(?P<time>\d{1,2}:\d{2})\s+)?(?P<home>.+?)\s+v\s+(?P<away>.+?)\s+(?P<hg>\d{1,2})\s*[-–]\s*(?P<ag>\d{1,2})(?=\s|$|\()",
    re.I,
)
SEASON_RE = re.compile(r"(\d{4})-(\d{2})")


def season_years(file_path: Path) -> tuple[int, int, str] | None:
    match = SEASON_RE.search(str(file_path.parent))
    if not match:
        match = SEASON_RE.search(str(file_path))
    if not match:
        return None
    start = int(match.group(1))
    end_short = int(match.group(2))
    end = (start // 100) * 100 + end_short
    if end < start:
        end += 100
    return start, end, f"{start}-{str(end)[-2:]}"


def parse_file(path: Path) -> list[dict]:
    years = season_years(path)
    if not years:
        return []
    start_year, end_year, season = years
    current_date = None
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
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
        if not mm:
            continue
        home = re.sub(r"\s+", " ", mm.group("home")).strip(" \t")
        away = re.sub(r"\s+", " ", mm.group("away")).strip(" \t")
        # Reject malformed rows and likely non-match lines.
        if not home or not away or home.lower() in {"v", "vs"} or away.lower() in {"v", "vs"}:
            continue
        rows.append({
            "date": current_date or "",
            "season": season,
            "home_team": home,
            "away_team": away,
            "home_goals": int(mm.group("hg")),
            "away_goals": int(mm.group("ag")),
        })
    return rows


def main() -> None:
    if not RAW_DIR.exists():
        raise SystemExit(
            "OpenFootball data not found. Run this from the project root:\n"
            "git clone https://github.com/openfootball/champions-league.git data/raw/champions-league"
        )
    files = sorted(RAW_DIR.rglob("*.txt"))
    rows = []
    for path in files:
        rows.extend(parse_file(path))
    if not rows:
        raise SystemExit("No completed matches were parsed. Check that the repository contains season .txt files.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["season", "date", "home_team", "away_team", "home_goals", "away_goals"])
    df = df.sort_values(["season", "date", "home_team"], na_position="first")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"Imported {len(df):,} completed matches from {len(files)} text files.")
    print(f"Saved: {OUT_FILE.relative_to(ROOT)}")
    print("Review data/matches.csv after importing; data formats can vary across seasons.")


if __name__ == "__main__":
    main()
