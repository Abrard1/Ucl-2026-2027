"""Import first-division domestic football results from openfootball/football.json.

Expected repository location:
    data/raw/football.json/

Output:
    data/domestic_matches.csv

Only completed matches with a full-time score are imported. Files are selected
by the common first-division naming pattern <country-code>.1.json to avoid cup
competitions. Coverage varies by country and season; inspect the output summary.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "data" / "raw" / "football.json"
OUT_FILE = ROOT / "data" / "domestic_matches.csv"
FIRST_DIVISION_RE = re.compile(r"^[a-z]{2,4}\.1\.json$", re.IGNORECASE)
SEASON_RE = re.compile(r"\b(\d{4}(?:-\d{2})?)\b")


def season_from_path(path: Path, doc: dict[str, Any]) -> str:
    """Get a season label from the parent folder, then the competition title."""
    parent = path.parent.name.strip()
    if re.fullmatch(r"\d{4}(?:-\d{2})?", parent):
        return parent
    name = str(doc.get("name", ""))
    match = SEASON_RE.search(name)
    return match.group(1) if match else parent


def full_time_score(match: dict[str, Any]) -> tuple[int, int] | None:
    score = match.get("score")
    if not isinstance(score, dict):
        return None
    ft = score.get("ft")
    if not isinstance(ft, (list, tuple)) or len(ft) != 2:
        return None
    try:
        home_goals = int(ft[0])
        away_goals = int(ft[1])
    except (TypeError, ValueError):
        return None
    if home_goals < 0 or away_goals < 0:
        return None
    return home_goals, away_goals


def parse_json_file(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Skipping unreadable JSON {path}: {exc}")
        return []

    if not isinstance(doc, dict) or not isinstance(doc.get("matches"), list):
        return []

    season = season_from_path(path, doc)
    league_name = str(doc.get("name") or path.stem)
    rows: list[dict[str, Any]] = []
    for match in doc["matches"]:
        if not isinstance(match, dict):
            continue
        score = full_time_score(match)
        if score is None:
            continue

        match_date = match.get("date")
        home = match.get("team1")
        away = match.get("team2")
        if not match_date or not home or not away:
            continue
        try:
            clean_date = pd.to_datetime(match_date, errors="raise").date().isoformat()
        except (TypeError, ValueError):
            continue

        rows.append({
            "date": clean_date,
            "season": season,
            "league": league_name,
            "home_team": str(home).strip(),
            "away_team": str(away).strip(),
            "home_goals": score[0],
            "away_goals": score[1],
            "round": str(match.get("round", "")),
            "source_file": str(path.relative_to(RAW_ROOT)),
        })
    return rows


def main() -> None:
    if not RAW_ROOT.exists():
        raise SystemExit(
            "Domestic data repository not found. From the project root, run:\n"
            "git clone https://github.com/openfootball/football.json.git data/raw/football.json"
        )

    files = sorted(
        path for path in RAW_ROOT.rglob("*.json")
        if FIRST_DIVISION_RE.fullmatch(path.name)
    )
    if not files:
        raise SystemExit(
            f"No first-division JSON files found under {RAW_ROOT}. "
            "Check the clone and its season folders."
        )

    rows: list[dict[str, Any]] = []
    for path in files:
        rows.extend(parse_json_file(path))

    if not rows:
        raise SystemExit("Found JSON files, but no completed matches with full-time scores were parsed.")

    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(
        subset=["date", "league", "home_team", "away_team", "home_goals", "away_goals"]
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Scanned {len(files):,} first-division JSON files.")
    print(f"Imported {len(df):,} completed matches ({before - len(df):,} duplicate rows removed).")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Competitions: {df['league'].nunique():,}; seasons: {df['season'].nunique():,}")
    print("Most common competitions:")
    print(df["league"].value_counts().head(15).to_string())
    print(f"Saved: {OUT_FILE.relative_to(ROOT)}")
    print("Note: this is source coverage, not a guarantee every league/season is complete.")


if __name__ == "__main__":
    main()
