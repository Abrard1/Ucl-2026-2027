#!/usr/bin/env python3
"""Compare current result/Elo model against the candidate xG-enhanced model.

Run from any working directory after placing this script in <project>/scripts/ and
predictor_xg.py in <project>/. Does not replace predictor.py or overwrite prior reports.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import predictor as baseline_model  # noqa: E402
import predictor_xg as xg_model  # noqa: E402


def metrics(frame: pd.DataFrame, prediction_columns: tuple[str, str, str]) -> dict[str, float | int]:
    if frame.empty:
        return {"n": 0, "accuracy": float("nan"), "brier": float("nan"), "log_loss": float("nan"), "rps": float("nan")}
    probs = frame[list(prediction_columns)].to_numpy(dtype=float)
    actual = frame["actual_result"].astype(str).to_numpy()
    idx = np.select([actual == "H", actual == "D", actual == "A"], [0, 1, 2], default=-1)
    if (idx < 0).any() or not np.isfinite(probs).all():
        raise ValueError("Backtest has unexpected outcomes or non-finite probability values")
    onehot = np.eye(3)[idx]
    p_actual = np.clip(probs[np.arange(len(probs)), idx], 1e-15, 1.0)
    p_ordered = probs[:, [2, 1, 0]]
    y_ordered = onehot[:, [2, 1, 0]]
    rps = np.mean(np.sum((np.cumsum(p_ordered, axis=1)[:, :2] - np.cumsum(y_ordered, axis=1)[:, :2]) ** 2, axis=1) / 2.0)
    return {
        "n": int(len(frame)),
        "accuracy": float((np.array(["H", "D", "A"])[np.argmax(probs, axis=1)] == actual).mean()),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(p_actual))),
        "rps": float(rps),
    }


def fmt(name: str, m: dict[str, float | int]) -> str:
    return (f"{name:<18} n={m['n']:>5,} | acc={m['accuracy']:.1%} | "
            f"Brier={m['brier']:.4f} | log loss={m['log_loss']:.4f} | RPS={m['rps']:.4f}")


def main() -> None:
    xg_file = ROOT / "data" / "understat_matches.csv"
    if not xg_file.exists():
        raise SystemExit("Missing data/understat_matches.csv. First run: python3 scripts/import_understat_xg.py")

    raw_xg = pd.read_csv(xg_file)
    if raw_xg.empty or not {"date", "home_team", "away_team", "home_xg", "away_xg"}.issubset(raw_xg.columns):
        raise SystemExit("data/understat_matches.csv is empty or lacks the required xG columns.")

    ucl = baseline_model.load_matches()
    domestic_base = baseline_model.load_domestic_matches()
    domestic_xg = xg_model.load_domestic_matches()
    xg_rows = int(domestic_xg["home_xg"].notna().sum() & domestic_xg["away_xg"].notna().sum())
    xg_matched = int((domestic_xg["home_xg"].notna() & domestic_xg["away_xg"].notna()).sum())
    seasons = sorted({str(s) for s in ucl["season"].dropna().unique() if str(s) and str(s) != "2026-27"},
                     key=lambda s: (int(s[:4]) if s[:4].isdigit() else 0, s))
    if not seasons:
        raise SystemExit("No completed Champions League seasons available for backtesting.")
    targets = set(seasons)

    print("Understat xG rows downloaded:", len(raw_xg))
    print("Domestic match rows with matched xG:", xg_matched, "of", len(domestic_xg))
    print("xG date range:", raw_xg["date"].min(), "to", raw_xg["date"].max())
    print("Backtest seasons:", ", ".join(seasons))
    print("Running original model...")
    old = baseline_model.run_backtest(ucl, domestic_base, targets)
    print("Running xG-enhanced candidate...")
    new = xg_model.run_backtest(ucl, domestic_xg, targets)
    if old.empty or new.empty:
        raise SystemExit("No backtest rows produced. Check the UCL dataset and season labels.")

    # Assert that both models evaluated exactly the same UCL fixtures, in the same order.
    keys = ["date", "season", "home_team", "away_team", "actual_result"]
    old_keys = old[keys].astype(str).reset_index(drop=True)
    new_keys = new[keys].astype(str).reset_index(drop=True)
    if len(old_keys) != len(new_keys) or not old_keys.equals(new_keys):
        raise SystemExit("Models did not score identical match rows; stop before comparing metrics.")

    old_m = metrics(old, ("home_win", "draw", "away_win"))
    new_m = metrics(new, ("home_win", "draw", "away_win"))
    print("\nOVERALL: original vs xG-enhanced candidate")
    print(fmt("Original model", old_m))
    print(fmt("xG candidate", new_m))
    print("Delta (candidate - original):")
    for key in ("accuracy", "brier", "log_loss", "rps"):
        print(f"  {key}: {new_m[key] - old_m[key]:+.4f} ({'better' if (new_m[key] > old_m[key] if key == 'accuracy' else new_m[key] < old_m[key]) else 'worse or equal'})")

    per_season = []
    print("\nBY SEASON")
    for season in seasons:
        old_g = old[old["season"].astype(str) == season]
        new_g = new[new["season"].astype(str) == season]
        a = metrics(old_g, ("home_win", "draw", "away_win"))
        b = metrics(new_g, ("home_win", "draw", "away_win"))
        per_season.append({"season": season, **{f"original_{k}": v for k, v in a.items()}, **{f"xg_{k}": v for k, v in b.items()}})
        print(f"{season}: original acc={a['accuracy']:.1%} Brier={a['brier']:.4f} logloss={a['log_loss']:.4f} | "
              f"xG acc={b['accuracy']:.1%} Brier={b['brier']:.4f} logloss={b['log_loss']:.4f}")

    out_old = ROOT / "data" / "comparison_original_model.csv"
    out_new = ROOT / "data" / "comparison_xg_model.csv"
    out_summary = ROOT / "data" / "comparison_xg_summary.csv"
    old.to_csv(out_old, index=False)
    new.to_csv(out_new, index=False)
    pd.DataFrame(per_season).to_csv(out_summary, index=False)
    print("\nSaved original predictions:", out_old)
    print("Saved xG candidate predictions:", out_new)
    print("Saved per-season comparison:", out_summary)
    print("Do not adopt the candidate on one overall metric alone; inspect seasons, calibration and sample coverage.")


if __name__ == "__main__":
    main()
