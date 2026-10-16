#!/usr/bin/env python3
"""Walk-forward UCL backtest using the live predictor's recent-season goal baseline.

Copies prediction logic from predictor._predict_with_state / predictor_xg._predict_with_state,
but computes each match's UCL scoring baseline from only the five most recent UCL seasons
with matches strictly before that match date. Same-day matches are all predicted first.

Read-only with respect to source files. Writes distinct reports under data/.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import predictor as base_model
    import predictor_xg as xg_model
except Exception as exc:
    raise SystemExit(
        "Could not import predictor.py and predictor_xg.py. Put this script in your project's scripts/ folder.\n"
        + str(exc)
    )

TEST_SEASONS_EXCLUDE = {"", "2026-27"}
N_BASELINE_SEASONS = 5


def season_key(value: Any) -> tuple[int, str]:
    return (base_model._season_year(value), str(value))


def recent_ucl_baseline(past_ucl: list[dict[str, Any]], module: Any) -> tuple[float, float]:
    """Home/away scoring means from up to five UCL seasons available before date."""
    if not past_ucl:
        return float(module.UCL_PRIOR_HOME), float(module.UCL_PRIOR_AWAY)
    frame = pd.DataFrame(past_ucl)
    season_labels = sorted(
        {str(s) for s in frame["season"].dropna().unique() if str(s) not in TEST_SEASONS_EXCLUDE},
        key=season_key,
    )
    selected_labels = set(season_labels[-N_BASELINE_SEASONS:])
    selected = frame[frame["season"].astype(str).isin(selected_labels)] if selected_labels else frame
    if selected.empty:
        return float(module.UCL_PRIOR_HOME), float(module.UCL_PRIOR_AWAY)
    h = float(pd.to_numeric(selected["home_goals"], errors="coerce").mean())
    a = float(pd.to_numeric(selected["away_goals"], errors="coerce").mean())
    if not np.isfinite(h) or h <= 0:
        h = float(module.UCL_PRIOR_HOME)
    if not np.isfinite(a) or a <= 0:
        a = float(module.UCL_PRIOR_AWAY)
    return h, a


def run_walk_forward(module: Any, ucl: pd.DataFrame, domestic: pd.DataFrame,
                     target_seasons: set[str], label: str) -> pd.DataFrame:
    events = module.prepare_events(ucl, domestic, as_of=None)
    if events.empty:
        return pd.DataFrame()
    state = module._History()
    past_ucl: list[dict[str, Any]] = []
    captured: list[dict[str, Any]] = []

    # Every fixture on a calendar date is scored against the same pre-date state.
    for _, group in events.groupby(events["date"].dt.normalize(), sort=True):
        base_h, base_a = recent_ucl_baseline(past_ucl, module)
        day_ucl_rows: list[dict[str, Any]] = []
        for _, row in group.iterrows():
            if row["source"] == "ucl":
                season = str(row.get("season", ""))
                if season in target_seasons:
                    pred = module._predict_with_state(
                        state, str(row["home_team"]), str(row["away_team"]), base_h, base_a
                    )
                    hg, ag = int(row["home_goals"]), int(row["away_goals"])
                    actual = "H" if hg > ag else ("D" if hg == ag else "A")
                    probs = [float(pred["home_win"]), float(pred["draw"]), float(pred["away_win"])]
                    actual_idx = {"H": 0, "D": 1, "A": 2}[actual]
                    baseline = module._poisson_prediction(
                        str(row["home_team"]), str(row["away_team"]), base_h, base_a,
                        True, True,
                    )
                    baseline_probs = [float(baseline["home_win"]), float(baseline["draw"]), float(baseline["away_win"])]
                    captured.append({
                        "model": label,
                        "date": pd.Timestamp(row["date"]).date().isoformat(),
                        "season": season,
                        "home_team": str(row["home_team"]),
                        "away_team": str(row["away_team"]),
                        "home_goals": hg,
                        "away_goals": ag,
                        "actual_result": actual,
                        "predicted_score": str(pred["predicted_score"]),
                        "home_win": probs[0], "draw": probs[1], "away_win": probs[2],
                        "baseline_home_win": baseline_probs[0],
                        "baseline_draw": baseline_probs[1],
                        "baseline_away_win": baseline_probs[2],
                        "actual_probability": probs[actual_idx],
                        "baseline_actual_probability": baseline_probs[actual_idx],
                        "predicted_result": ["H", "D", "A"][int(np.argmax(probs))],
                        "baseline_predicted_result": ["H", "D", "A"][int(np.argmax(baseline_probs))],
                        "baseline_home_goals_rate": base_h,
                        "baseline_away_goals_rate": base_a,
                    })
                day_ucl_rows.append({
                    "season": season,
                    "home_goals": int(row["home_goals"]),
                    "away_goals": int(row["away_goals"]),
                })
        # Apply every result only after all predictions for this date have been made.
        for _, row in group.iterrows():
            state.update(row)
        past_ucl.extend(day_ucl_rows)
    return pd.DataFrame(captured)


def metrics(frame: pd.DataFrame, prefix: str = "") -> dict[str, float | int]:
    if frame.empty:
        return {"n": 0, "accuracy": float("nan"), "brier": float("nan"), "log_loss": float("nan"), "rps": float("nan")}
    probability_columns = [f"{prefix}home_win", f"{prefix}draw", f"{prefix}away_win"]
    p = frame[probability_columns].to_numpy(dtype=float)
    actual = frame["actual_result"].astype(str).to_numpy()
    idx = np.select([actual == "H", actual == "D", actual == "A"], [0, 1, 2], default=-1)
    if (idx < 0).any() or not np.isfinite(p).all():
        raise ValueError("Invalid outcome labels or non-finite probabilities found")
    onehot = np.eye(3)[idx]
    actual_probability = np.clip(p[np.arange(len(p)), idx], 1e-15, 1.0)
    ordered_p = p[:, [2, 1, 0]]
    ordered_y = onehot[:, [2, 1, 0]]
    rps = float(np.mean(np.sum((np.cumsum(ordered_p, axis=1)[:, :2] - np.cumsum(ordered_y, axis=1)[:, :2]) ** 2, axis=1) / 2.0))
    pred_result_column = f"{prefix}predicted_result"
    accuracy = float((frame[pred_result_column].astype(str).to_numpy() == actual).mean())
    return {
        "n": int(len(frame)),
        "accuracy": accuracy,
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(actual_probability))),
        "rps": rps,
    }


def fmt(name: str, result: dict[str, float | int]) -> str:
    return (f"{name:<19} n={result['n']:>5,} | acc={result['accuracy']:.1%} | "
            f"Brier={result['brier']:.4f} | log loss={result['log_loss']:.4f} | RPS={result['rps']:.4f}")


def main() -> None:
    ucl = base_model.load_matches()
    domestic_base = base_model.load_domestic_matches()
    domestic_xg = xg_model.load_domestic_matches()
    seasons = sorted(
        {str(s) for s in ucl["season"].dropna().unique() if str(s) not in TEST_SEASONS_EXCLUDE},
        key=season_key,
    )
    target_seasons = set(seasons)
    if not target_seasons:
        raise SystemExit("No completed UCL seasons found")

    print("Chronological backtest using a recent-five-season UCL scoring baseline")
    print("This corrects the prior backtest's cumulative-all-history scoring baseline mismatch.")
    print("Target seasons:", ", ".join(seasons))
    print("Every target match is predicted before any results dated on the same day are applied.")

    original = run_walk_forward(base_model, ucl, domestic_base, target_seasons, "original")
    candidate = run_walk_forward(xg_model, ucl, domestic_xg, target_seasons, "xg_candidate")
    if original.empty or candidate.empty:
        raise SystemExit("Backtest produced no predictions")
    key_cols = ["date", "season", "home_team", "away_team", "actual_result"]
    if not original[key_cols].astype(str).reset_index(drop=True).equals(candidate[key_cols].astype(str).reset_index(drop=True)):
        raise SystemExit("Models were not tested on identical fixtures; refusing comparison")

    for frame in (original, candidate):
        frame["baseline_home_win"] = frame["baseline_home_win"]
        frame["baseline_draw"] = frame["baseline_draw"]
        frame["baseline_away_win"] = frame["baseline_away_win"]

    combined_metrics = []
    print("\nOVERALL")
    for label, frame in (("Original model", original), ("xG candidate", candidate)):
        print(fmt(label, metrics(frame)))
        print(fmt("Simple baseline", metrics(frame, prefix="baseline_")))
        m = metrics(frame)
        combined_metrics.append({"model": label, **m})

    print("\nBY SEASON: model metrics")
    season_rows = []
    for season in seasons:
        row = {"season": season}
        for label, frame in (("original", original), ("xg_candidate", candidate)):
            part = frame[frame["season"].astype(str) == season]
            m = metrics(part)
            row.update({f"{label}_{k}": v for k, v in m.items()})
            print(f"{season} {label}: " + fmt("Model", m))
        season_rows.append(row)

    original_path = ROOT / "data" / "backtest_live_equivalent_original.csv"
    candidate_path = ROOT / "data" / "backtest_live_equivalent_xg.csv"
    summary_path = ROOT / "data" / "backtest_live_equivalent_summary.csv"
    original.to_csv(original_path, index=False)
    candidate.to_csv(candidate_path, index=False)
    pd.DataFrame(season_rows).to_csv(summary_path, index=False)
    print("\nSaved original match predictions:", original_path)
    print("Saved xG candidate predictions:", candidate_path)
    print("Saved per-season summary:", summary_path)
    print("Do not choose a winner from accuracy alone; inspect probability metrics and season-to-season stability.")


if __name__ == "__main__":
    main()
