#!/usr/bin/env python3
"""Download completed domestic match xG from Understat into data/understat_matches.csv.

Understat does not offer an official API. This script uses its public JSON endpoints
conservatively: one request at a time, a delay between requests, retries, and local
caching. Use for personal/research purposes; check the source site's terms before
redistributing or using the data commercially.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "understat"
OUT_FILE = DATA_DIR / "understat_matches.csv"
BASE_URL = "https://understat.com/"
LEAGUES = {
    "EPL": "English Premier League",
    "La_Liga": "Spanish La Liga",
    "Bundesliga": "German Bundesliga",
    "Serie_A": "Italian Serie A",
    "Ligue_1": "French Ligue 1",
}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Referer": BASE_URL,
    "Accept": "application/json, text/javascript, */*; q=0.01",
}


def as_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
            return None
        number = float(value)
        return number if number >= 0 else None
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def fetch_json(session: requests.Session, url: str, attempts: int = 3) -> dict:
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            response = session.get(url, headers=HEADERS, timeout=(15, 45))
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object from Understat")
            return data
        except Exception as exc:  # Keep the exact error visible to the user.
            last_error = exc
            if attempt < attempts:
                pause = 2 * attempt
                print(f"  retry {attempt}/{attempts - 1} in {pause}s: {exc}")
                time.sleep(pause)
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def parse_season(data: dict, league_slug: str, league_name: str, year: int) -> list[dict]:
    matches = data.get("dates") or data.get("datesData") or []
    rows: list[dict] = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        # Future/unplayed fixtures are not training examples.
        if "isResult" in match and not is_true(match.get("isResult")):
            continue
        home = match.get("h") or {}
        away = match.get("a") or {}
        goals = match.get("goals") or {}
        xg = match.get("xG") or {}
        home_team = home.get("title")
        away_team = away.get("title")
        date_value = match.get("datetime") or match.get("date")
        try:
            date = pd.to_datetime(date_value, errors="raise").date().isoformat()
        except Exception:
            continue
        hg = as_int(goals.get("h"))
        ag = as_int(goals.get("a"))
        hxg = as_float(xg.get("h"))
        axg = as_float(xg.get("a"))
        if not home_team or not away_team or hg is None or ag is None or hxg is None or axg is None:
            continue
        rows.append({
            "date": date,
            "season": season_label(year),
            "league": league_name,
            "league_slug": league_slug,
            "home_team": str(home_team).strip(),
            "away_team": str(away_team).strip(),
            "home_goals": hg,
            "away_goals": ag,
            "home_xg": hxg,
            "away_xg": axg,
            "understat_match_id": match.get("id"),
            "source_url": f"https://understat.com/league/{league_slug}/{year}",
            "retrieved_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2014, help="First season start year (default: 2014)")
    parser.add_argument("--end-year", type=int, default=datetime.now().year, help="Last season start year, inclusive (default: current year)")
    parser.add_argument("--delay", type=float, default=1.2, help="Seconds between requests; keep this at least 1.0")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch cached seasons too")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")
    if args.delay < 1.0:
        raise SystemExit("Please keep --delay at 1.0 seconds or higher to avoid hammering the source")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    all_rows: list[dict] = []
    failures: list[str] = []
    total_tasks = len(LEAGUES) * (args.end_year - args.start_year + 1)
    completed_tasks = 0

    for slug, league_name in LEAGUES.items():
        for year in range(args.start_year, args.end_year + 1):
            cache_file = RAW_DIR / f"{slug}_{year}.json"
            url = f"{BASE_URL}getLeagueData/{slug}/{year}"
            data = None
            should_fetch = args.refresh or not cache_file.exists() or year == args.end_year
            try:
                if should_fetch:
                    data = fetch_json(session, url)
                    cache_file.write_text(json.dumps(data), encoding="utf-8")
                    time.sleep(args.delay)
                else:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                rows = parse_season(data, slug, league_name, year)
                all_rows.extend(rows)
                print(f"[{completed_tasks + 1:>3}/{total_tasks}] {league_name} {season_label(year)}: {len(rows)} completed xG matches")
            except Exception as exc:
                failures.append(f"{league_name} {season_label(year)}: {exc}")
                print(f"[{completed_tasks + 1:>3}/{total_tasks}] SKIP {league_name} {season_label(year)}: {exc}")
            completed_tasks += 1

    if not all_rows:
        print("\nNo xG match rows were imported. No output file was written.")
        print("Check the network/browser access to Understat, then rerun the script.")
        if failures:
            print("First failures:")
            for failure in failures[:8]:
                print(" -", failure)
        raise SystemExit(1)

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_xg", "away_xg"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.drop_duplicates(subset=["date", "league_slug", "home_team", "away_team"], keep="last")
    df = df.sort_values(["date", "league_slug", "home_team"]).reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)

    print("\nSUMMARY")
    print("Rows saved:", len(df))
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("League counts:\n" + df.groupby("league").size().to_string())
    print("Season range:", ", ".join(sorted(df["season"].unique())[:1] + sorted(df["season"].unique())[-1:]))
    print("Saved:", OUT_FILE)
    print("Note: Understat is not an official API; data coverage varies and can lag. Keep requests slow and inspect the match join report.")
    if failures:
        print(f"\nPartial result: {len(failures)} league-season requests failed. Cached successful rows were still saved.")
        for failure in failures[:8]:
            print(" -", failure)


if __name__ == "__main__":
    main()
