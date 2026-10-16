#!/usr/bin/env python3
"""Audit how well downloaded Understat xG rows join to domestic_matches.csv.

Run from project root or any folder:
    python3 scripts/audit_xg_coverage.py
It is read-only and does not modify datasets.
"""
from pathlib import Path
import sys
import unicodedata
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for req in (ROOT / "data" / "domestic_matches.csv", ROOT / "data" / "understat_matches.csv"):
    if not req.exists():
        raise SystemExit("Missing required file: {}".format(req))
sys.path.insert(0, str(ROOT))
try:
    from predictor_xg import canonical_team
except Exception:
    # Keep audit usable if the user copied only this script; mirrors core normalization.
    def canonical_team(name):
        text = str(name or "").strip()
        text = re.sub(r"\s*\([A-Za-z]{3}\)\s*$", "", text)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = text.lower().replace("&", " and ")
        text = re.sub(r"[^a-z0-9]+", " ", text).strip()
        text = re.sub(r"\b(fc|cf|afc|sc|ssc|ac|as|fk|sk)\b", " ", text)
        return re.sub(r"\s+", " ", text).strip()

xg = pd.read_csv(ROOT / "data" / "understat_matches.csv")
dom = pd.read_csv(ROOT / "data" / "domestic_matches.csv")
for frame, source in ((xg, "xG"), (dom, "Domestic")):
    required = {"date", "home_team", "away_team"}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit("{} file missing columns: {}".format(source, ", ".join(sorted(missing))))
    frame["_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["_home"] = frame["home_team"].map(canonical_team)
    frame["_away"] = frame["away_team"].map(canonical_team)
    frame["_key"] = frame["_date"].dt.strftime("%Y-%m-%d").fillna("") + "|" + frame["_home"] + "|" + frame["_away"]

# Exact match means date and both teams line up after normalization.
dom_keys = set(dom["_key"].dropna())
xg["exact_match"] = xg["_key"].isin(dom_keys)
exact_count = int(xg["exact_match"].sum())

print("Understat xG rows:", len(xg))
print("Domestic rows:", len(dom))
print("Exact xG rows matched to domestic data:", exact_count, "({:.1%})".format(exact_count / max(1, len(xg))))
print("Unique exact domestic keys:", dom["_key"].nunique())
print("Duplicate Understat date/team keys:", int(xg.duplicated("_key").sum()))
print("\nMatch rate by Understat league and season (xG rows matched to domestic rows):")
if "league" in xg.columns:
    groups = xg.groupby(["league", "season"], dropna=False) if "season" in xg.columns else xg.groupby(["league"], dropna=False)
    rows = []
    for name, g in groups:
        rows.append((name if isinstance(name, tuple) else (name,), len(g), int(g["exact_match"].sum()), float(g["exact_match"].mean())))
    for name, total, matched, rate in rows:
        print("  {}: {}/{} ({:.1%})".format(" | ".join(map(str, name)), matched, total, rate))

# Check whether most misses are team-name differences or dates.
miss = xg.loc[~xg["exact_match"]].copy()
# Pair-only matching can be many-to-many across seasons; keep only minimal columns and nearest dates.
dom_pair = dom[["_date", "_home", "_away", "home_team", "away_team"]].dropna(subset=["_date"])
dom_pair = dom_pair.rename(columns={"_date": "dom_date", "home_team": "dom_home_original", "away_team": "dom_away_original"})
if len(miss):
    pair_candidates = miss[["date", "season", "league", "home_team", "away_team", "_date", "_home", "_away"]].merge(
        dom_pair, on=["_home", "_away"], how="left")
    pair_candidates["day_gap"] = (pair_candidates["_date"] - pair_candidates["dom_date"]).abs().dt.days
    close = pair_candidates[pair_candidates["day_gap"].le(3)].sort_values("day_gap").drop_duplicates(
        subset=["date", "season", "league", "home_team", "away_team"])
    same_pair_any_date = pair_candidates.dropna(subset=["dom_date"]).drop_duplicates(
        subset=["date", "season", "league", "home_team", "away_team"]).shape[0]
    print("\nUnmatched xG rows:", len(miss))
    print("Unmatched rows with same normalized home/away pair within +/-3 days:", len(close))
    print("Unmatched rows with same normalized home/away pair somewhere in domestic data:", same_pair_any_date)
    if len(close):
        print("\nExamples where teams match but date differs (possible timezone/date discrepancy):")
        cols = ["date", "league", "home_team", "away_team", "dom_date", "dom_home_original", "dom_away_original", "day_gap"]
        print(close[cols].head(15).to_string(index=False))
    print("\nExamples with no close normalized team-pair match:")
    no_close_keys = set(zip(close["date"].astype(str), close["league"].astype(str), close["home_team"].astype(str), close["away_team"].astype(str)))
    examples = []
    for _, r in miss.iterrows():
        key = (str(r.get("date")), str(r.get("league")), str(r.get("home_team")), str(r.get("away_team")))
        if key not in no_close_keys:
            examples.append({"date": r.get("date"), "league": r.get("league"), "xg_home": r.get("home_team"), "xg_away": r.get("away_team"), "canon_home": r.get("_home"), "canon_away": r.get("_away")})
        if len(examples) >= 15:
            break
    if examples:
        print(pd.DataFrame(examples).to_string(index=False))
else:
    print("\nAll xG rows have an exact domestic match.")

print("\nThis audit is read-only; no files were changed.")
