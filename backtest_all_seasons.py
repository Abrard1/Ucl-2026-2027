"""Run a chronological UCL backtest over every completed season.

This script uses the project's existing predictor.py and inputs. It does not
modify app.py, predictor.py, matches.csv, or the existing backtest output.
It saves results to data/backtest_all_seasons.csv.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Union
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from predictor import load_matches, load_domestic_matches, run_backtest  # noqa: E402


def metrics(frame: pd.DataFrame, prefix: str = "") -> Dict[str, Union[float, int]]:
    if frame.empty:
        return {"n": 0, "accuracy": float("nan"), "brier": float("nan"), "log_loss": float("nan"), "rps": float("nan")}
    cols = [f"{prefix}home_win", f"{prefix}draw", f"{prefix}away_win"]
    probs = frame[cols].to_numpy(dtype=float)
    actual = frame["actual_result"].astype(str).to_numpy()
    idx = np.select([actual == "H", actual == "D", actual == "A"], [0, 1, 2], default=-1)
    if (idx < 0).any():
        raise ValueError("Found unexpected actual_result labels in backtest output")
    onehot = np.eye(3)[idx]
    actual_prob = np.clip(probs[np.arange(len(probs)), idx], 1e-15, 1.0)
    # Ranked Probability Score for ordered outcomes: away < draw < home.
    # Reverse columns to cumulative away and draw-or-away probabilities.
    p_ordered = probs[:, [2, 1, 0]]
    y_ordered = onehot[:, [2, 1, 0]]
    cumulative_p = np.cumsum(p_ordered, axis=1)[:, :2]
    cumulative_y = np.cumsum(y_ordered, axis=1)[:, :2]
    rps = np.mean(np.sum((cumulative_p - cumulative_y) ** 2, axis=1) / 2.0)
    pred_col = f"{prefix}predicted_result"
    accuracy = float((frame[pred_col].astype(str).to_numpy() == actual).mean())
    return {
        "n": int(len(frame)),
        "accuracy": accuracy,
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(actual_prob))),
        "rps": float(rps),
    }


def fmt(label: str, result: Dict[str, Union[float, int]]) -> str:
    return (
        f"{label:<12} n={result['n']:>5,} | acc={result['accuracy']:.1%} | "
        f"Brier={result['brier']:.4f} | log loss={result['log_loss']:.4f} | RPS={result['rps']:.4f}"
    )


def main() -> None:
    ucl = load_matches()
    domestic = load_domestic_matches()
    if "season" not in ucl.columns:
        raise SystemExit("data/matches.csv has no season column")

    seasons = sorted(
        {str(s) for s in ucl["season"].dropna().unique() if str(s) and str(s) != "2026-27"},
        key=lambda value: (int(value[:4]) if value[:4].isdigit() else 0, value),
    )
    if not seasons:
        raise SystemExit("No completed UCL seasons found")

    print("Chronological Champions League backtest")
    print("Target seasons:", ", ".join(seasons))
    print("Method: each match prediction uses only events strictly before that match date.")
    print("Note: a same-day group is predicted before results from that date are applied.")
    results = run_backtest(ucl, domestic, set(seasons))
    if results.empty:
        raise SystemExit("No backtest results generated")

    out = ROOT / "data" / "backtest_all_seasons.csv"
    results.to_csv(out, index=False)
    print("\nOVERALL")
    print(fmt("Model", metrics(results)))
    print(fmt("Baseline", metrics(results, prefix="baseline_")))
    print("\nBY SEASON")
    records = []
    for season, group in results.groupby("season", sort=True):
        m = metrics(group)
        b = metrics(group, prefix="baseline_")
        records.append({"season": season, **{f"model_{k}": v for k, v in m.items()}, **{f"baseline_{k}": v for k, v in b.items()}})
        print(f"{season}: " + fmt("Model", m) + " || " + fmt("Base", b))
    pd.DataFrame(records).to_csv(ROOT / "data" / "backtest_season_summary.csv", index=False)
    print("\nSaved match-level predictions:", out)
    print("Saved season summary:", ROOT / "data" / "backtest_season_summary.csv")
    print("Interpretation: diagnostic only; historical performance does not guarantee future accuracy.")


if __name__ == "__main__":
    main()
