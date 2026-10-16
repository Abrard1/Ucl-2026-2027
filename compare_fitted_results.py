#!/usr/bin/env python3
"""Compare already-saved fitted-model forecasts with prior forecasts.

This script does not refit models. It fixes the comparison step after the
fitted attack/defence run, so users don't have to rerun the expensive fit just
to compare saved predictions.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import predictor_xg as current
except Exception as exc:
    raise SystemExit(f"Could not import predictor_xg.py: {exc}")

HOLDOUT = {"2023-24", "2024-25", "2025-26"}
VALIDATION = {"2014-15", "2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23"}
KEYS = ["date", "season", "_home", "_away"]
PROB_COLS = ["home_win", "draw", "away_win"]


def load_predictions(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the corresponding backtest first.")
    d = pd.read_csv(path)
    required = {"date", "season", "home_team", "away_team", "home_goals", "away_goals", *PROB_COLS}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {sorted(missing)}")
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    d["season"] = d["season"].astype(str)
    d["_home"] = d["home_team"].map(current.canonical_team)
    d["_away"] = d["away_team"].map(current.canonical_team)
    d["home_goals"] = pd.to_numeric(d["home_goals"], errors="coerce")
    d["away_goals"] = pd.to_numeric(d["away_goals"], errors="coerce")
    for c in PROB_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=KEYS + ["home_goals", "away_goals"] + PROB_COLS).copy()
    d["actual_result"] = np.where(d["home_goals"] > d["away_goals"], "H", np.where(d["home_goals"] == d["away_goals"], "D", "A"))
    if d.duplicated(KEYS).any():
        dupes = int(d.duplicated(KEYS, keep=False).sum())
        raise ValueError(f"{path.name} has {dupes} rows with duplicate normalised fixture keys; won't silently pick one.")
    d["model"] = label
    return d[KEYS + ["actual_result"] + PROB_COLS + ["model"]].copy()


def calculate_metrics(d: pd.DataFrame):
    if d.empty:
        return {"n": 0, "accuracy": np.nan, "brier": np.nan, "log_loss": np.nan}
    p = d[PROB_COLS].to_numpy(float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    y = d["actual_result"].to_numpy(str)
    mapping = {"H": 0, "D": 1, "A": 2}
    yi = np.array([mapping[v] for v in y], dtype=int)
    onehot = np.eye(3)[yi]
    pred = np.array(["H", "D", "A"])[p.argmax(axis=1)]
    return {
        "n": int(len(d)),
        "accuracy": float((pred == y).mean()),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(p[np.arange(len(d)), yi]))),
    }


def main():
    data = ROOT / "data"
    models = {
        "Fitted attack/defence": load_predictions(data / "backtest_fitted_attack_defence.csv", "Fitted attack/defence"),
        "Existing heuristic": load_predictions(data / "backtest_live_equivalent_original.csv", "Existing heuristic"),
        "Existing xG candidate": load_predictions(data / "backtest_live_equivalent_xg.csv", "Existing xG candidate"),
    }
    # Inner join all models to ensure every reported comparison uses the exact same fixtures.
    joined = None
    for label, frame in models.items():
        subset = frame[KEYS + ["actual_result"] + PROB_COLS].rename(columns={c: f"{label}::{c}" for c in ["actual_result"] + PROB_COLS})
        joined = subset if joined is None else joined.merge(subset, on=KEYS, how="inner", validate="one_to_one")
    if joined is None or joined.empty:
        raise SystemExit("No common fixtures found across the three saved predictions. Check dates/team normalisation.")
    actual_cols = [f"{label}::actual_result" for label in models]
    if not (joined[actual_cols].nunique(axis=1) == 1).all():
        raise SystemExit("The saved prediction files disagree on actual match results for some fixture keys.")
    joined["actual_result"] = joined[actual_cols[0]]
    joined["season"] = joined["season"].astype(str)

    reports = []
    splits = {
        "ALL_COMMON_FIXTURES": set(joined["season"]),
        "VALIDATION": VALIDATION,
        "HOLDOUT": HOLDOUT,
    }
    for split_name, seasons in splits.items():
        part = joined[joined["season"].isin(seasons)].copy()
        print(f"\n{split_name} (identical fixtures for all models)")
        for label in models:
            d = part[["actual_result"] + [f"{label}::{c}" for c in PROB_COLS]].copy()
            d.columns = ["actual_result"] + PROB_COLS
            m = calculate_metrics(d)
            reports.append({"split": split_name, "season": "ALL", "model": label, **m})
            print(f"{label:<24} n={m['n']:>4} | accuracy={m['accuracy']:.1%} | Brier={m['brier']:.4f} | logloss={m['log_loss']:.4f}")

    print("\nHOLDOUT BY SEASON")
    for season in sorted(HOLDOUT):
        part = joined[joined["season"] == season]
        for label in models:
            d = part[["actual_result"] + [f"{label}::{c}" for c in PROB_COLS]].copy()
            d.columns = ["actual_result"] + PROB_COLS
            m = calculate_metrics(d)
            reports.append({"split": "HOLDOUT_BY_SEASON", "season": season, "model": label, **m})
            print(f"{season} | {label:<24} n={m['n']:>4} | accuracy={m['accuracy']:.1%} | Brier={m['brier']:.4f} | logloss={m['log_loss']:.4f}")

    out = data / "fitted_model_comparison_summary.csv"
    pd.DataFrame(reports).to_csv(out, index=False)
    print(f"\nCommon fixtures compared: {len(joined)}")
    print(f"Saved comparison summary: {out}")
    print("Note: the holdout seasons have already been inspected during earlier experiments. Treat this as diagnostic evidence, not a fresh untouched test.")

if __name__ == "__main__":
    main()
