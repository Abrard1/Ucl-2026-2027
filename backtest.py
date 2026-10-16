from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from predictor import load_matches, load_domestic_matches, run_backtest

def summarize(df: pd.DataFrame, prefix: str = "") -> dict:
    if df.empty:
        return {"matches": 0}
    onehot_home = (df["actual_result"] == "H").astype(float).to_numpy()
    onehot_draw = (df["actual_result"] == "D").astype(float).to_numpy()
    onehot_away = (df["actual_result"] == "A").astype(float).to_numpy()
    probs = df[["home_win", "draw", "away_win"]].to_numpy() if not prefix else df[["baseline_home_win", "baseline_draw", "baseline_away_win"]].to_numpy()
    labels = [onehot_home, onehot_draw, onehot_away]
    brier = ((probs - __import__("numpy").column_stack(labels)) ** 2).sum(axis=1).mean()
    p = df["actual_probability"].to_numpy() if not prefix else df["baseline_actual_probability"].to_numpy()
    logloss = -__import__("numpy").log(__import__("numpy").clip(p, 1e-12, 1.0)).mean()
    pred = df["predicted_result"] if not prefix else df["baseline_predicted_result"]
    accuracy = (pred.to_numpy() == df["actual_result"].to_numpy()).mean()
    return {"matches": len(df), "accuracy": float(accuracy), "brier": float(brier), "log_loss": float(logloss)}


def main() -> None:
    ucl = load_matches()
    domestic = load_domestic_matches()
    available = sorted(ucl["season"].fillna("").astype(str).unique(), key=lambda x: (int(x[:4]) if x[:4].isdigit() else 0, x))
    # Evaluate completed recent seasons only, deliberately excluding the current 2026/27 season.
    targets = {s for s in available if s in {"2024-25", "2025-26"}}
    if not targets:
        targets = set(available[-3:-1])
    results = run_backtest(ucl, domestic, targets)
    if results.empty:
        raise SystemExit("No backtest rows produced; check season labels and imported data.")
    out = ROOT / "data" / "backtest_predictions.csv"
    results.to_csv(out, index=False)
    model = summarize(results)
    baseline = summarize(results, prefix="baseline")
    print("Chronological UCL backtest (each prediction is made before the match date)")
    print("Seasons:", ", ".join(sorted(targets)))
    print(f"Model:    n={model['matches']:,} | accuracy={model['accuracy']:.1%} | Brier={model['brier']:.4f} | log loss={model['log_loss']:.4f}")
    print(f"Baseline: n={baseline['matches']:,} | accuracy={baseline['accuracy']:.1%} | Brier={baseline['brier']:.4f} | log loss={baseline['log_loss']:.4f}")
    print("Saved match-level predictions:", out)
    print("Read these as a diagnostic, not proof of predictive skill; compare on more seasons and avoid tuning against the test seasons.")

if __name__ == "__main__":
    main()
