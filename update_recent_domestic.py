"""Update recent domestic results using macOS curl to avoid Python LibreSSL issues.

Run from the UCL project root:
    python3 scripts/update_recent_domestic.py

The script downloads free Football-Data.co.uk CSVs, appends completed 2026/27
matches later than the current domestic dataset, and preserves the old file.
It stores results, shots/cards where present, and available pre-match odds columns.
These extra fields are not yet used by the current predictor; feature engineering
must be added separately and validated with walk-forward backtesting.
"""
from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
import shutil
import subprocess

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "domestic_matches.csv"
BACKUP_FILE = ROOT / "data" / "domestic_matches_before_football_data.csv"
RAW_SAVE = ROOT / "data" / "raw" / "football_data_2026_27"
SEASON_CODE = "2627"
LEAGUES = {
    "E0": "English Premier League",
    "SC0": "Scottish Premiership",
    "D1": "German Bundesliga",
    "I1": "Italian Serie A",
    "SP1": "Spanish La Liga",
    "F1": "French Ligue 1",
    "N1": "Dutch Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Portuguese Primeira Liga",
    "T1": "Turkish Super Lig",
    "G1": "Greek Super League",
}
STAT_MAP = {
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HC": "home_corners", "AC": "away_corners",
    "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellow_cards", "AY": "away_yellow_cards",
    "HR": "home_red_cards", "AR": "away_red_cards",
    "AvgH": "odds_avg_home", "AvgD": "odds_avg_draw", "AvgA": "odds_avg_away",
    "B365H": "odds_b365_home", "B365D": "odds_b365_draw", "B365A": "odds_b365_away",
    "PSH": "odds_pinnacle_home", "PSD": "odds_pinnacle_draw", "PSA": "odds_pinnacle_away",
    "AvgCH": "odds_avg_close_home", "AvgCD": "odds_avg_close_draw", "AvgCA": "odds_avg_close_away",
}
REQUIRED = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"}


def download_csv(code: str) -> pd.DataFrame | None:
    url = f"https://www.football-data.co.uk/mmz4281/{SEASON_CODE}/{code}.csv"
    # On this Mac, Python 3.9 is built against LibreSSL and urllib has failed TLS.
    # curl uses the OS network stack and avoids that Python urllib failure path.
    cmd = [
        "curl", "--location", "--fail", "--silent", "--show-error",
        "--retry", "2", "--connect-timeout", "15", "--max-time", "45",
        "--user-agent", "Mozilla/5.0 UCL-Predictor-educational-project", url,
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, timeout=55)
        payload = result.stdout
        if not payload.strip():
            print(f"SKIP {code}: empty response")
            return None
        df = pd.read_csv(BytesIO(payload), encoding="latin1", low_memory=False)
        df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
        if not REQUIRED.issubset(df.columns):
            print(f"SKIP {code}: expected result columns missing; got {list(df.columns[:12])}")
            return None
        RAW_SAVE.mkdir(parents=True, exist_ok=True)
        (RAW_SAVE / f"{code}.csv").write_bytes(payload)
        return df
    except FileNotFoundError:
        raise SystemExit("curl was not found. On macOS it is normally installed; check by running: curl --version")
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        print(f"SKIP {code}: curl download failed ({detail or f'exit code {exc.returncode}'})")
        return None
    except Exception as exc:
        print(f"SKIP {code}: download/read failed ({exc})")
        return None


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit("data/domestic_matches.csv not found. Run scripts/import_football_json.py first.")
    existing = pd.read_csv(DATA_FILE, low_memory=False)
    required = {"date", "season", "league", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(existing.columns)
    if missing:
        raise SystemExit(f"data/domestic_matches.csv is missing required columns: {', '.join(sorted(missing))}")

    existing_dates = pd.to_datetime(existing["date"], errors="coerce")
    max_existing = existing_dates.max()
    if pd.isna(max_existing):
        raise SystemExit("Could not determine the latest date in data/domestic_matches.csv")
    cutoff = max_existing.normalize()
    today = pd.Timestamp(date.today()).normalize()
    print(f"Existing domestic data latest date: {cutoff.date()}")
    print(f"Downloading completed 2026/27 matches after {cutoff.date()} and no later than {today.date()}.")

    new_frames: list[pd.DataFrame] = []
    for code, league_name in LEAGUES.items():
        raw = download_csv(code)
        if raw is None or raw.empty:
            continue
        match_dates = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
        home_goals = pd.to_numeric(raw["FTHG"], errors="coerce")
        away_goals = pd.to_numeric(raw["FTAG"], errors="coerce")
        mask = match_dates.notna() & home_goals.notna() & away_goals.notna()
        mask &= (match_dates > cutoff) & (match_dates <= today)
        recent = raw.loc[mask].copy()
        if recent.empty:
            print(f"{code:>3} {league_name}: no completed matches after cutoff")
            continue
        selected_dates = match_dates.loc[mask]
        out = pd.DataFrame({
            "date": selected_dates.dt.strftime("%Y-%m-%d").to_numpy(),
            "season": "2026-27",
            "league": league_name,
            "home_team": recent["HomeTeam"].astype(str).str.strip().to_numpy(),
            "away_team": recent["AwayTeam"].astype(str).str.strip().to_numpy(),
            "home_goals": home_goals.loc[mask].astype(int).to_numpy(),
            "away_goals": away_goals.loc[mask].astype(int).to_numpy(),
            "round": "",
            "source_file": f"football-data.co.uk/{SEASON_CODE}/{code}.csv",
        })
        for source_col, target_col in STAT_MAP.items():
            if source_col in recent.columns:
                out[target_col] = pd.to_numeric(recent[source_col], errors="coerce").to_numpy()
            else:
                out[target_col] = float("nan")
        new_frames.append(out)
        print(f"{code:>3} {league_name}: appended-candidate matches = {len(out)}; latest = {out['date'].max()}")

    if not new_frames:
        print("No new completed matches were found. Existing domestic data was not changed.")
        print("If every league reports a curl/network error, run: curl --version")
        print("Then test one CSV: curl -L --fail --show-error 'https://www.football-data.co.uk/mmz4281/2627/E0.csv' -o /tmp/E0.csv")
        return

    new = pd.concat(new_frames, ignore_index=True, sort=False)
    new["date"] = pd.to_datetime(new["date"], errors="coerce")
    new = new.dropna(subset=["date"])
    new = new[(new["date"] > cutoff) & (new["date"] <= today)]
    new["date"] = new["date"].dt.strftime("%Y-%m-%d")
    new = new.drop_duplicates(subset=["date", "league", "home_team", "away_team", "home_goals", "away_goals"])

    if not BACKUP_FILE.exists():
        shutil.copy2(DATA_FILE, BACKUP_FILE)
        print(f"Backed up existing domestic data to {BACKUP_FILE.relative_to(ROOT)}")
    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    combined = combined.sort_values(["date", "league", "home_team"]).reset_index(drop=True)
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.to_csv(DATA_FILE, index=False)

    print(f"\nNew completed domestic matches appended: {len(new):,}")
    print(f"Domestic rows after update: {len(combined):,}")
    print(f"Latest date now: {combined['date'].max()}")
    print(f"Updated: {DATA_FILE.relative_to(ROOT)}")
    print("Note: odds and match statistics are stored for later modelling, not yet used by predictor.py.")


if __name__ == "__main__":
    main()
