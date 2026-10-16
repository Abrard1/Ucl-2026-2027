
from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {"date", "home_team", "away_team", "home_goals", "away_goals"}
UCL_COMP = "UEFA Champions League"
UCL_PRIOR_HOME = 1.53
UCL_PRIOR_AWAY = 1.20
FORM_WINDOW_DOMESTIC = 12
FORM_WINDOW_UCL = 6
FORM_PRIOR_WEIGHT = 3.0
MAX_GOALS = 10
HOME_ADVANTAGE_ELO = 55.0

# These aliases are applied after Unicode/punctuation/club-suffix cleanup.
ALIASES = {
    "paris sg": "paris saint germain", "psg": "paris saint germain",
    "paris saint germain": "paris saint germain",
    "inter milano": "inter", "inter milan": "inter", "internazionale": "inter",
    "fc internazionale": "inter", "inter": "inter",
    "atletico de madrid": "atletico madrid", "atletico madrid": "atletico madrid",
    "man utd": "manchester united", "man united": "manchester united",
    "manchester utd": "manchester united", "manchester united": "manchester united",
    "man city": "manchester city", "manchester city": "manchester city",
    "rb leipzig": "leipzig", "rasenballsport leipzig": "leipzig", "leipzig": "leipzig",
    "vfb stuttgart": "stuttgart", "stuttgart": "stuttgart",
    "slavia praha": "slavia prague", "slavia prague": "slavia prague",
    "bodo glimt": "bodo glimt", "bodoe glimt": "bodo glimt",
    "racing club de lens": "lens", "rc lens": "lens", "lens": "lens",
    "fenerbahce istanbul": "fenerbahce", "fenerbahce": "fenerbahce",
    "sabah masazir": "sabah", "sabah fc": "sabah", "sabah": "sabah",
    "real betis seville": "real betis", "real betis": "real betis",
    "fc porto": "porto", "porto": "porto",
    "fc barcelona": "barcelona", "barca": "barcelona", "barcelona": "barcelona",
    # Common name variants from Understat, Football-Data and OpenFootball.
    "ath bilbao": "athletic club", "athletic bilbao": "athletic club", "athletic club": "athletic club",
    "wolverhampton": "wolverhampton wanderers", "wolves": "wolverhampton wanderers",
    "wolverhampton wanderers": "wolverhampton wanderers",
    "nottm forest": "nottingham forest", "nottingham forest": "nottingham forest",
    "brighton": "brighton and hove albion", "brighton hove albion": "brighton and hove albion",
    "brighton and hove albion": "brighton and hove albion",
    "west ham": "west ham united", "west ham united": "west ham united",
    "spurs": "tottenham hotspur", "tottenham": "tottenham hotspur", "tottenham hotspur": "tottenham hotspur",
    "eintr frankfurt": "eintracht frankfurt", "eintracht frankfurt": "eintracht frankfurt",
    "gladbach": "borussia monchengladbach", "borussia monchengladbach": "borussia monchengladbach",
    "hoffenheim": "hoffenheim", "tsg hoffenheim": "hoffenheim",
    "athletic bilbao": "athletic club", "real sociedad": "real sociedad",
    "fc bayern munchen": "bayern munich", "bayern munchen": "bayern munich",
    "bayern munich": "bayern munich", "viking fk": "viking", "viking": "viking",
    "lask linz": "lask", "lask": "lask", "lille osc": "lille", "lille": "lille",
    "club brugge kv": "club brugge", "club brugge": "club brugge",
    "as roma": "roma", "roma": "roma",
    "fc shakhtar donetsk": "shakhtar donetsk", "shakhtar": "shakhtar donetsk",
    "shakhtar donetsk": "shakhtar donetsk",
    "sporting clube de portugal": "sporting cp", "sporting cp": "sporting cp",
    "como 1907": "como", "como": "como",
    # Additional aliases observed in Understat / Football-Data team labels.
    "leicester city": "leicester", "leicester": "leicester",
    "stoke city": "stoke", "stoke": "stoke",
    "swansea city": "swansea", "swansea": "swansea",
    "hull city": "hull", "hull": "hull",
    "queens park rangers": "qpr", "qpr": "qpr",
    "west bromwich albion": "west brom", "west brom": "west brom",
    "newcastle united": "newcastle", "newcastle": "newcastle",
    "norwich city": "norwich", "norwich": "norwich",
    "cardiff city": "cardiff", "cardiff": "cardiff",
    "wigan athletic": "wigan", "wigan": "wigan",
    "blackburn rovers": "blackburn", "blackburn": "blackburn",
    "bolton wanderers": "bolton", "bolton": "bolton",
    "sheffield united": "sheffield united", "sheffield utd": "sheffield united",
    "sheffield wednesday": "sheffield wednesday", "sheff wed": "sheffield wednesday",
    "bayer 04 leverkusen": "bayer leverkusen", "bayer leverkusen": "bayer leverkusen",
    "schalke 04": "schalke", "schalke": "schalke",
    "hannover 96": "hannover", "hannover": "hannover",
    "celta de vigo": "celta vigo", "celta vigo": "celta vigo",
    "deportivo alaves": "alaves", "alaves": "alaves", "deportivo alaves": "alaves",
    "ud almeria": "almeria", "almeria": "almeria",
    "cd leganes": "leganes", "leganes": "leganes",
    "real valladolid": "valladolid", "valladolid": "valladolid",
    "hellas verona": "verona", "verona": "verona",
    "udinese calcio": "udinese", "udinese": "udinese",
    "parma calcio 1913": "parma", "parma": "parma",
}


def canonical_team(name: Any) -> str:
    """Normalise team strings and remove OpenFootball country suffixes such as (ENG)."""
    text = str(name or "").strip()
    text = re.sub(r"\s*\([A-Za-z]{3}\)\s*$", "", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    # Remove common club designators, not identity-bearing words such as City or United.
    text = re.sub(r"\b(fc|cf|afc|sc|ssc|ac|as|fk|sk)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    df = pd.read_csv(path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_goals"] = pd.to_numeric(df["home_goals"], errors="coerce")
    df["away_goals"] = pd.to_numeric(df["away_goals"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df = df[(df["home_goals"] >= 0) & (df["away_goals"] >= 0)]
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    return df.reset_index(drop=True)


def load_matches(path: str = "data/matches.csv", current_path: Optional[str] = None) -> pd.DataFrame:
    """Load historical UCL rows plus curated current-season UCL results, if present."""
    base_path = Path(path)
    if not base_path.is_absolute():
        base_path = ROOT / base_path
    historical = _read_csv(base_path, REQUIRED_COLUMNS)
    if "season" not in historical.columns:
        historical["season"] = historical["date"].dt.year.astype(str)
    historical["source"] = "ucl"
    if current_path is None:
        current_file = ROOT / "data" / "ucl_2026_27_results.csv"
    else:
        current_file = Path(current_path)
        if not current_file.is_absolute():
            current_file = ROOT / current_file
    parts = [historical]
    if current_file.exists():
        current = _read_csv(current_file, REQUIRED_COLUMNS)
        if "season" not in current.columns:
            current["season"] = "2026-27"
        current["source"] = "ucl"
        parts.append(current)
    ucl = pd.concat(parts, ignore_index=True, sort=False)
    ucl["season"] = ucl["season"].fillna("").astype(str)
    ucl["_home_key"] = ucl["home_team"].map(canonical_team)
    ucl["_away_key"] = ucl["away_team"].map(canonical_team)
    ucl = ucl.drop_duplicates(subset=["date", "_home_key", "_away_key", "home_goals", "away_goals"])
    return ucl.drop(columns=["_home_key", "_away_key"]).sort_values("date").reset_index(drop=True)


def _merge_understat_xg(domestic: pd.DataFrame, xg_path: Optional[Path] = None) -> pd.DataFrame:
    """Attach Understat xG with safe team/score matching and +/-1-day date tolerance.

    Exact date + canonical home/away names + full-time score is preferred. If that
    fails, a match may be linked only when the same canonical teams and score have
    exactly one domestic candidate within one day. Ambiguous or score-mismatched
    rows are deliberately left unmatched to prevent false xG assignments.
    """
    if xg_path is None:
        xg_path = ROOT / "data" / "understat_matches.csv"
    df = domestic.copy().reset_index(drop=True)
    if not xg_path.exists():
        df["home_xg"] = np.nan
        df["away_xg"] = np.nan
        df["_xg_join_gap_days"] = np.nan
        return df

    xg = pd.read_csv(xg_path)
    required_xg = {"date", "home_team", "away_team", "home_xg", "away_xg", "home_goals", "away_goals"}
    missing = required_xg.difference(xg.columns)
    if missing:
        raise ValueError(f"{xg_path.name} missing xG join columns: {', '.join(sorted(missing))}")

    for col in ("home_xg", "away_xg", "home_goals", "away_goals"):
        xg[col] = pd.to_numeric(xg[col], errors="coerce")
    for col in ("home_goals", "away_goals"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_join_date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    xg["_join_date"] = pd.to_datetime(xg["date"], errors="coerce").dt.normalize()
    df["_join_home"] = df["home_team"].map(canonical_team)
    df["_join_away"] = df["away_team"].map(canonical_team)
    xg["_join_home"] = xg["home_team"].map(canonical_team)
    xg["_join_away"] = xg["away_team"].map(canonical_team)
    df["_hg"] = df["home_goals"].round().astype("Int64")
    df["_ag"] = df["away_goals"].round().astype("Int64")
    xg["_hg"] = xg["home_goals"].round().astype("Int64")
    xg["_ag"] = xg["away_goals"].round().astype("Int64")
    xg = xg.dropna(subset=["_join_date", "_join_home", "_join_away", "_hg", "_ag", "home_xg", "away_xg"])

    # Index domestic rows by teams and score; use date only to disambiguate nearby matches.
    candidate_map = defaultdict(list)
    exact_map = defaultdict(list)
    for di, row in df.iterrows():
        if pd.isna(row["_join_date"]) or pd.isna(row["_hg"]) or pd.isna(row["_ag"]):
            continue
        base = (row["_join_home"], row["_join_away"], int(row["_hg"]), int(row["_ag"]))
        candidate_map[base].append(di)
        exact_map[(row["_join_date"],) + base].append(di)

    # Assign each domestic row at most one xG record and each xG record at most once.
    assigned_domestic = set()
    mapped = []
    exact_count = 0
    tolerance_count = 0
    ambiguous_count = 0
    score_team_count = 0
    for xi, row in xg.sort_values("_join_date").iterrows():
        base = (row["_join_home"], row["_join_away"], int(row["_hg"]), int(row["_ag"]))
        xdate = row["_join_date"]
        exact_candidates = [di for di in exact_map.get((xdate,) + base, []) if di not in assigned_domestic]
        chosen = None
        gap = None
        if len(exact_candidates) == 1:
            chosen = exact_candidates[0]
            gap = 0
            exact_count += 1
        elif len(exact_candidates) > 1:
            ambiguous_count += 1
            continue
        else:
            # Candidate rows must match both teams in order and the final score.
            close = []
            for di in candidate_map.get(base, []):
                if di in assigned_domestic:
                    continue
                dd = df.at[di, "_join_date"]
                day_gap = abs((dd - xdate).days)
                if day_gap <= 1:
                    close.append((day_gap, di))
            if close:
                min_gap = min(v[0] for v in close)
                best = [di for g, di in close if g == min_gap]
                if len(best) == 1:
                    chosen = best[0]
                    gap = min_gap
                    tolerance_count += 1
                else:
                    ambiguous_count += 1
                    continue
            elif candidate_map.get(base):
                # Same teams and full-time score exist in the domestic source, but not nearby.
                score_team_count += 1
                continue
            else:
                continue

        if chosen is not None:
            assigned_domestic.add(chosen)
            mapped.append((chosen, float(row["home_xg"]), float(row["away_xg"]), int(gap), str(row.get("date", ""))))

    # Remove prior xG values before applying this deterministic join.
    df = df.drop(columns=[c for c in ("home_xg", "away_xg", "_xg_join_gap_days", "_xg_source_date") if c in df.columns])
    df["home_xg"] = np.nan
    df["away_xg"] = np.nan
    df["_xg_join_gap_days"] = np.nan
    df["_xg_source_date"] = ""
    for di, hxg, axg, gap, xdate in mapped:
        df.at[di, "home_xg"] = hxg
        df.at[di, "away_xg"] = axg
        df.at[di, "_xg_join_gap_days"] = gap
        df.at[di, "_xg_source_date"] = xdate

    # Keep a concise diagnostic for every run; this reveals whether the fix helps.
    total_xg = len(xg)
    print("Understat join: {}/{} xG rows mapped ({:.1%}); exact-date={}, +/-1-day={}, ambiguous={}, same team/score but >1 day={}".format(
        len(mapped), total_xg, len(mapped) / max(1, total_xg), exact_count, tolerance_count, ambiguous_count, score_team_count
    ))
    return df.drop(columns=["_join_date", "_join_home", "_join_away", "_hg", "_ag"])


def load_domestic_matches(path: str = "data/domestic_matches.csv") -> pd.DataFrame:
    """Load domestic results and attach Understat xG when data/understat_matches.csv exists."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    if not file_path.exists():
        return pd.DataFrame(columns=["date", "season", "league", "home_team", "away_team", "home_goals", "away_goals", "home_xg", "away_xg", "source"])
    required = REQUIRED_COLUMNS | {"league"}
    df = _read_csv(file_path, required)
    df["season"] = df["season"].fillna("").astype(str) if "season" in df.columns else ""
    df = _merge_understat_xg(df)
    df["source"] = "domestic"
    return df


def _season_year(value: Any) -> int:
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else 0


def _select_recent_seasons(ucl: pd.DataFrame, seasons: int) -> pd.DataFrame:
    if ucl.empty or seasons <= 0:
        return ucl.copy()
    labels = sorted(ucl["season"].fillna("").astype(str).unique(), key=lambda x: (_season_year(x), x))
    keep = set(labels[-seasons:])
    return ucl[ucl["season"].astype(str).isin(keep)].copy()


def prepare_events(ucl: pd.DataFrame, domestic: pd.DataFrame, as_of: Optional[Any] = None) -> pd.DataFrame:
    """Create chronological match events. If as_of is given, exclude later results."""
    u = ucl.copy()
    u["source"] = "ucl"
    u["competition"] = UCL_COMP
    if "season" not in u.columns:
        u["season"] = ""
    d = domestic.copy()
    d["source"] = "domestic"
    d["competition"] = d["league"].fillna("Domestic league").astype(str)
    if "season" not in d.columns:
        d["season"] = ""
    optional_features = ["home_xg", "away_xg"]
    for col in optional_features:
        if col not in u.columns:
            u[col] = np.nan
        if col not in d.columns:
            d[col] = np.nan
    cols = ["date", "season", "home_team", "away_team", "home_goals", "away_goals", "source", "competition"] + optional_features
    events = pd.concat([u[cols], d[cols]], ignore_index=True, sort=False)
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events = events.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])
    if as_of is not None:
        cutoff = pd.Timestamp(as_of).normalize()
        events = events[events["date"] <= cutoff]
    events["home_key"] = events["home_team"].map(canonical_team)
    events["away_key"] = events["away_team"].map(canonical_team)
    events = events[(events["home_key"] != "") & (events["away_key"] != "")]
    # Same competition/date/team/score duplicates are dropped, but cross-competition matches stay separate.
    events = events.drop_duplicates(subset=["date", "competition", "home_key", "away_key", "home_goals", "away_goals"])
    events["_source_order"] = events["source"].map({"domestic": 0, "ucl": 1}).fillna(2)
    return events.sort_values(["date", "_source_order", "competition", "home_key"]).reset_index(drop=True)


class _History:
    def __init__(self) -> None:
        self.ratings: dict[str, float] = {}
        self.domestic: dict[str, deque] = {}
        self.ucl: dict[str, deque] = {}
        self.comp_stats: dict[str, dict[str, float]] = {}
        self.matches_seen = 0

    def _deque(self, mapping: dict, key: str, maxlen: int) -> deque:
        if key not in mapping:
            mapping[key] = deque(maxlen=maxlen)
        return mapping[key]

    def team_rating(self, key: str) -> float:
        return self.ratings.get(key, 1500.0)

    def comp_average(self, competition: str) -> tuple[float, float]:
        # Priors prevent an early-season league average being based on one match.
        entry = self.comp_stats.get(competition)
        if entry is None:
            if competition == UCL_COMP:
                return UCL_PRIOR_HOME, UCL_PRIOR_AWAY
            return 1.48, 1.20
        prior_n = entry["prior_n"]
        home_mean = (entry["sum_home"] + prior_n * entry["prior_home"]) / (entry["n"] + prior_n)
        away_mean = (entry["sum_away"] + prior_n * entry["prior_away"]) / (entry["n"] + prior_n)
        return max(0.6, home_mean), max(0.5, away_mean)

    def comp_average_xg(self, competition: str) -> tuple[float, float]:
        """Smoothed competition xG rates based only on earlier matches with xG coverage."""
        entry = self.comp_stats.get(competition)
        if entry is None or entry.get("n_xg", 0.0) <= 0:
            return self.comp_average(competition)
        prior_n = entry["prior_n"]
        home_mean = (entry["sum_xg_home"] + prior_n * entry["prior_home"]) / (entry["n_xg"] + prior_n)
        away_mean = (entry["sum_xg_away"] + prior_n * entry["prior_away"]) / (entry["n_xg"] + prior_n)
        return max(0.4, home_mean), max(0.4, away_mean)

    def update(self, row: pd.Series) -> None:
        hk = str(row["home_key"])
        ak = str(row["away_key"])
        hg = int(row["home_goals"])
        ag = int(row["away_goals"])
        source = str(row["source"])
        competition = str(row["competition"])
        base_h, base_a = self.comp_average(competition)
        # Store performance relative to the competition baseline. Where historical
        # xG exists for this domestic match, blend 25% actual goals with 75% xG.
        # This keeps one-off finishing variance from dominating the rolling form signal.
        home_xg = pd.to_numeric(row.get("home_xg", np.nan), errors="coerce")
        away_xg = pd.to_numeric(row.get("away_xg", np.nan), errors="coerce")
        has_xg = source == "domestic" and pd.notna(home_xg) and pd.notna(away_xg)
        if has_xg:
            base_xg_h, base_xg_a = self.comp_average_xg(competition)
            h_att_goal = hg / base_h
            h_att_xg = float(home_xg) / base_xg_h
            h_def_goal = ag / base_a
            h_def_xg = float(away_xg) / base_xg_a
            a_att_goal = ag / base_a
            a_att_xg = float(away_xg) / base_xg_a
            a_def_goal = hg / base_h
            a_def_xg = float(home_xg) / base_xg_h
            h_att = 0.25 * h_att_goal + 0.75 * h_att_xg
            h_def = 0.25 * h_def_goal + 0.75 * h_def_xg
            a_att = 0.25 * a_att_goal + 0.75 * a_att_xg
            a_def = 0.25 * a_def_goal + 0.75 * a_def_xg
        else:
            h_att = hg / base_h
            h_def = ag / base_a
            a_att = ag / base_a
            a_def = hg / base_h
        h_att = float(np.clip(h_att, 0.0, 3.5))
        h_def = float(np.clip(h_def, 0.0, 3.5))
        a_att = float(np.clip(a_att, 0.0, 3.5))
        a_def = float(np.clip(a_def, 0.0, 3.5))
        entry_home = {"attack": h_att, "defense": h_def, "date": row["date"]}
        entry_away = {"attack": a_att, "defense": a_def, "date": row["date"]}
        if source == "ucl":
            self._deque(self.ucl, hk, FORM_WINDOW_UCL).append(entry_home)
            self._deque(self.ucl, ak, FORM_WINDOW_UCL).append(entry_away)
        else:
            self._deque(self.domestic, hk, FORM_WINDOW_DOMESTIC).append(entry_home)
            self._deque(self.domestic, ak, FORM_WINDOW_DOMESTIC).append(entry_away)

        # Elo update: domestic results are useful for current form; UCL matches carry a slightly larger K.
        rh = self.team_rating(hk)
        ra = self.team_rating(ak)
        expected_h = 1.0 / (1.0 + 10.0 ** (-((rh + HOME_ADVANTAGE_ELO) - ra) / 400.0))
        actual_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        k = 30.0 if source == "ucl" else 20.0
        margin = 1.0 + 0.12 * min(abs(hg - ag), 4)
        delta = k * margin * (actual_h - expected_h)
        self.ratings[hk] = rh + delta
        self.ratings[ak] = ra - delta

        if competition not in self.comp_stats:
            if competition == UCL_COMP:
                prior_home, prior_away, prior_n = UCL_PRIOR_HOME, UCL_PRIOR_AWAY, 10.0
            else:
                prior_home, prior_away, prior_n = 1.48, 1.20, 10.0
            self.comp_stats[competition] = {
                "sum_home": 0.0, "sum_away": 0.0, "n": 0.0,
                "sum_xg_home": 0.0, "sum_xg_away": 0.0, "n_xg": 0.0,
                "prior_home": prior_home, "prior_away": prior_away, "prior_n": prior_n,
            }
        comp = self.comp_stats[competition]
        # Upgrade an entry created by an earlier version in the same process if needed.
        comp.setdefault("sum_xg_home", 0.0)
        comp.setdefault("sum_xg_away", 0.0)
        comp.setdefault("n_xg", 0.0)
        comp["sum_home"] += hg
        comp["sum_away"] += ag
        comp["n"] += 1
        if has_xg:
            comp["sum_xg_home"] += float(home_xg)
            comp["sum_xg_away"] += float(away_xg)
            comp["n_xg"] += 1
        self.matches_seen += 1


def _smoothed_ratio(history: deque, metric: str) -> float:
    if not history:
        return 1.0
    values = [float(item[metric]) for item in history]
    # Recent matches get greater weight; prior neutral performances stop tiny samples dominating.
    weights = np.array([0.88 ** (len(values) - 1 - i) for i in range(len(values))], dtype=float)
    numerator = FORM_PRIOR_WEIGHT * 1.0 + float(np.dot(weights, np.array(values, dtype=float)))
    denominator = FORM_PRIOR_WEIGHT + float(weights.sum())
    return float(np.clip(numerator / denominator, 0.45, 2.2))


def _team_features(state: _History, key: str) -> dict[str, float]:
    dom_hist = state.domestic.get(key, deque())
    ucl_hist = state.ucl.get(key, deque())
    dom_att = _smoothed_ratio(dom_hist, "attack")
    dom_def = _smoothed_ratio(dom_hist, "defense")
    ucl_att = _smoothed_ratio(ucl_hist, "attack")
    ucl_def = _smoothed_ratio(ucl_hist, "defense")
    # Two UCL results should not outweigh a full run of domestic evidence; weight rises with UCL sample size.
    n_ucl = len(ucl_hist)
    weight_ucl = n_ucl / (n_ucl + 6.0) if n_ucl else 0.0
    attack = weight_ucl * ucl_att + (1.0 - weight_ucl) * dom_att
    defence_weakness = weight_ucl * ucl_def + (1.0 - weight_ucl) * dom_def
    return {
        "home_attack": float(attack), "away_attack": float(attack),
        "home_def_weakness": float(defence_weakness), "away_def_weakness": float(defence_weakness),
        "attack_index": float(attack), "defence_weakness_index": float(defence_weakness),
        "elo": float(state.team_rating(key)), "domestic_matches": float(len(dom_hist)),
        "ucl_matches": float(len(ucl_hist)),
    }


@dataclass
class GoalModel:
    avg_home_goals: float
    avg_away_goals: float
    team_stats: dict[str, dict[str, float]]
    match_count: int
    seasons_used: list[str]
    domestic_match_count: int = 0
    data_as_of: str = "unknown"


def _poisson_prediction(home_team: str, away_team: str, lam_home: float, lam_away: float,
                        home_known: bool, away_known: bool, home_elo: float = 1500.0,
                        away_elo: float = 1500.0) -> dict[str, Any]:
    lam_home = float(np.clip(lam_home, 0.10, 5.0))
    lam_away = float(np.clip(lam_away, 0.10, 5.0))
    goals = np.arange(MAX_GOALS + 1)
    p_h = poisson.pmf(goals, lam_home)
    p_a = poisson.pmf(goals, lam_away)
    matrix = np.outer(p_h, p_a)
    matrix /= max(float(matrix.sum()), 1e-12)
    p_home = float(np.tril(matrix, k=-1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, k=1).sum())
    total = p_home + p_draw + p_away
    p_home, p_draw, p_away = [v / total for v in (p_home, p_draw, p_away)]
    score_h, score_a = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return {
        "home_team": home_team, "away_team": away_team,
        "expected_home_goals": lam_home, "expected_away_goals": lam_away,
        "predicted_home_goals": int(score_h), "predicted_away_goals": int(score_a),
        "predicted_score": f"{score_h}–{score_a}",
        "home_win": p_home, "draw": p_draw, "away_win": p_away,
        "home_known": bool(home_known), "away_known": bool(away_known),
        "home_elo": float(home_elo), "away_elo": float(away_elo),
        "elo_difference_home_adjusted": float(home_elo + HOME_ADVANTAGE_ELO - away_elo),
    }


def _predict_with_state(state: _History, home_team: str, away_team: str,
                        baseline_home: Optional[float] = None,
                        baseline_away: Optional[float] = None) -> dict[str, Any]:
    hk, ak = canonical_team(home_team), canonical_team(away_team)
    if baseline_home is None or baseline_away is None:
        baseline_home, baseline_away = state.comp_average(UCL_COMP)
    hs = _team_features(state, hk)
    aws = _team_features(state, ak)
    rh, ra = hs["elo"], aws["elo"]
    # Elo is a restrained adjustment, not a direct claim about goal output.
    elo_factor = math.exp(float(np.clip((rh - ra) / 400.0 * 0.25, -0.40, 0.40)))
    lam_home = baseline_home * hs["home_attack"] * aws["away_def_weakness"] * elo_factor
    lam_away = baseline_away * aws["away_attack"] * hs["home_def_weakness"] / elo_factor
    result = _poisson_prediction(home_team, away_team, lam_home, lam_away,
                                 hs["ucl_matches"] + hs["domestic_matches"] > 0,
                                 aws["ucl_matches"] + aws["domestic_matches"] > 0,
                                 rh, ra)
    result["home_attack_index"] = hs["attack_index"]
    result["away_attack_index"] = aws["attack_index"]
    result["home_defence_weakness_index"] = hs["defence_weakness_index"]
    result["away_defence_weakness_index"] = aws["defence_weakness_index"]
    result["home_ucl_matches_used"] = int(hs["ucl_matches"])
    result["away_ucl_matches_used"] = int(aws["ucl_matches"])
    return result


def fit_model(matches: pd.DataFrame, seasons: int = 5,
              domestic_path: str = "data/domestic_matches.csv", as_of: Optional[Any] = None) -> GoalModel:
    """Build ratings/forms from all past data, and use recent UCL seasons for scoring baseline."""
    missing = REQUIRED_COLUMNS.difference(matches.columns)
    if missing:
        raise ValueError(f"Historical data is missing columns: {', '.join(sorted(missing))}")
    ucl = matches.copy()
    ucl["date"] = pd.to_datetime(ucl["date"], errors="coerce")
    if "season" not in ucl.columns:
        ucl["season"] = ucl["date"].dt.year.astype(str)
    ucl["season"] = ucl["season"].fillna("").astype(str)
    ucl = ucl.dropna(subset=["date", "home_goals", "away_goals"])
    selected = _select_recent_seasons(ucl, seasons)
    if selected.empty:
        raise ValueError("No completed UCL matches available to fit the model.")
    domestic = load_domestic_matches(domestic_path)
    today = pd.Timestamp(date.today()) if as_of is None else pd.Timestamp(as_of)
    events = prepare_events(ucl, domestic, as_of=today)
    if events.empty:
        raise ValueError("No valid events remain after date filtering.")
    state = _History()
    # Batch by date: all predictions for a matchday must be made before any results on that date are used.
    for _, group in events.groupby(events["date"].dt.normalize(), sort=True):
        for _, row in group.iterrows():
            state.update(row)
    team_stats = {k: _team_features(state, k) for k in set(state.ratings) | set(state.domestic) | set(state.ucl)}
    # Average goals from selected UCL seasons; current 2026/27 results are included if present.
    avg_home = float(selected["home_goals"].mean())
    avg_away = float(selected["away_goals"].mean())
    if not np.isfinite(avg_home) or avg_home <= 0: avg_home = UCL_PRIOR_HOME
    if not np.isfinite(avg_away) or avg_away <= 0: avg_away = UCL_PRIOR_AWAY
    seasons_used = sorted(selected["season"].dropna().astype(str).unique(), key=lambda x: (_season_year(x), x))
    domestic_count = int((events["source"] == "domestic").sum())
    data_as_of = str(events["date"].max().date())
    return GoalModel(avg_home, avg_away, team_stats, len(selected), seasons_used, domestic_count, data_as_of)


def predict_match(model: GoalModel, home_team: str, away_team: str) -> dict[str, Any]:
    """Predict one fixture from fitted club form and Elo features."""
    hk, ak = canonical_team(home_team), canonical_team(away_team)
    hs = model.team_stats.get(hk, {})
    aws = model.team_stats.get(ak, {})
    home_attack = float(hs.get("home_attack", 1.0))
    away_attack = float(aws.get("away_attack", 1.0))
    home_def = float(hs.get("home_def_weakness", 1.0))
    away_def = float(aws.get("away_def_weakness", 1.0))
    home_elo = float(hs.get("elo", 1500.0))
    away_elo = float(aws.get("elo", 1500.0))
    elo_factor = math.exp(float(np.clip((home_elo - away_elo) / 400.0 * 0.25, -0.40, 0.40)))
    lam_home = model.avg_home_goals * home_attack * away_def * elo_factor
    lam_away = model.avg_away_goals * away_attack * home_def / elo_factor
    out = _poisson_prediction(home_team, away_team, lam_home, lam_away,
                              bool(hs.get("domestic_matches", 0) or hs.get("ucl_matches", 0)),
                              bool(aws.get("domestic_matches", 0) or aws.get("ucl_matches", 0)),
                              home_elo, away_elo)
    out["home_attack_index"] = float(hs.get("attack_index", 1.0))
    out["away_attack_index"] = float(aws.get("attack_index", 1.0))
    out["home_defence_weakness_index"] = float(hs.get("defence_weakness_index", 1.0))
    out["away_defence_weakness_index"] = float(aws.get("defence_weakness_index", 1.0))
    out["home_ucl_matches_used"] = int(hs.get("ucl_matches", 0))
    out["away_ucl_matches_used"] = int(aws.get("ucl_matches", 0))
    return out


def _run_chronological_backtest(events: pd.DataFrame, target_seasons: set[str]) -> pd.DataFrame:
    """Predict selected historical UCL matches pre-kickoff while updating state chronologically."""
    state = _History()
    captured: list[dict[str, Any]] = []
    for _, group in events.groupby(events["date"].dt.normalize(), sort=True):
        # Use one common pre-match state for every fixture on the same date.
        base_h, base_a = state.comp_average(UCL_COMP)
        for _, row in group.iterrows():
            if row["source"] == "ucl" and str(row.get("season", "")) in target_seasons:
                pred = _predict_with_state(state, str(row["home_team"]), str(row["away_team"]), base_h, base_a)
                hg, ag = int(row["home_goals"]), int(row["away_goals"])
                actual = "H" if hg > ag else ("D" if hg == ag else "A")
                probs = [pred["home_win"], pred["draw"], pred["away_win"]]
                idx = {"H": 0, "D": 1, "A": 2}[actual]
                baseline = _poisson_prediction(str(row["home_team"]), str(row["away_team"]), base_h, base_a, True, True)
                bprobs = [baseline["home_win"], baseline["draw"], baseline["away_win"]]
                captured.append({
                    "date": row["date"].date().isoformat(), "season": str(row.get("season", "")),
                    "home_team": row["home_team"], "away_team": row["away_team"],
                    "home_goals": hg, "away_goals": ag, "actual_result": actual,
                    "predicted_score": pred["predicted_score"],
                    "home_win": probs[0], "draw": probs[1], "away_win": probs[2],
                    "baseline_home_win": bprobs[0], "baseline_draw": bprobs[1], "baseline_away_win": bprobs[2],
                    "actual_probability": probs[idx], "baseline_actual_probability": bprobs[idx],
                    "predicted_result": ["H", "D", "A"][int(np.argmax(probs))],
                    "baseline_predicted_result": ["H", "D", "A"][int(np.argmax(bprobs))],
                })
        for _, row in group.iterrows():
            state.update(row)
    return pd.DataFrame(captured)


def run_backtest(ucl: pd.DataFrame, domestic: pd.DataFrame, target_seasons: set[str]) -> pd.DataFrame:
    events = prepare_events(ucl, domestic, as_of=None)
    return _run_chronological_backtest(events, target_seasons)
