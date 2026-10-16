#!/usr/bin/env python3
"""Walk-forward Dixon-Coles rho sweep for the UCL Predictor project.

Uses predictor_xg's current event preparation, form/Elo updates, and recent-five-UCL-season
scoring baseline. Applies the standard Dixon-Coles low-score correction to the score matrix.

Parameter selection is restricted to 2014-15 through 2022-23. The final holdout is 2023-24
through 2025-26. All predictions are made before results dated on the same day are applied.
No source datasets are modified. Output CSVs have new names.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import predictor_xg as model
except Exception as exc:
    raise SystemExit(
        "Could not import predictor_xg.py. Keep this script in your project's scripts/ folder.\n"
        + str(exc)
    )

# Set these before looking at the final holdout results. Do not expand the grid using holdout metrics.
RHO_GRID = [-0.15, -0.10, -0.05, 0.0]
VALIDATION_SEASONS = {"2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
                      "2019-20", "2020-21", "2021-22", "2022-23"}
HOLDOUT_SEASONS = {"2023-24", "2024-25", "2025-26"}
EXCLUDED_SEASONS = {"", "2026-27"}
N_BASELINE_SEASONS = 5


def season_sort_key(value: Any) -> Tuple[int, str]:
    match = re.search(r"(19|20)\d{2}", str(value))
    return (int(match.group(0)) if match else 0, str(value))


def recent_ucl_baseline(past_ucl: List[Dict[str, Any]]) -> Tuple[float, float]:
    if not past_ucl:
        return float(model.UCL_PRIOR_HOME), float(model.UCL_PRIOR_AWAY)
    frame = pd.DataFrame(past_ucl)
    labels = sorted(
        {str(s) for s in frame["season"].dropna().unique() if str(s) not in EXCLUDED_SEASONS},
        key=season_sort_key,
    )
    selected_labels = set(labels[-N_BASELINE_SEASONS:])
    selected = frame[frame["season"].astype(str).isin(selected_labels)] if selected_labels else frame
    if selected.empty:
        return float(model.UCL_PRIOR_HOME), float(model.UCL_PRIOR_AWAY)
    home_mean = float(pd.to_numeric(selected["home_goals"], errors="coerce").mean())
    away_mean = float(pd.to_numeric(selected["away_goals"], errors="coerce").mean())
    if not np.isfinite(home_mean) or home_mean <= 0:
        home_mean = float(model.UCL_PRIOR_HOME)
    if not np.isfinite(away_mean) or away_mean <= 0:
        away_mean = float(model.UCL_PRIOR_AWAY)
    return home_mean, away_mean


def make_dixon_coles_predictor(original_poisson, rho: float):
    """Wrap the existing score-probability function with a Dixon-Coles tau correction."""
    def wrapped(home_team: str, away_team: str, lam_home: float, lam_away: float,
                home_known: bool, away_known: bool, home_elo: float = 1500.0,
                away_elo: float = 1500.0) -> Dict[str, Any]:
        # Preserve metadata and base expected-goal fields from the existing implementation.
        result = original_poisson(home_team, away_team, lam_home, lam_away,
                                  home_known, away_known, home_elo, away_elo)
        lam_h = float(np.clip(lam_home, 0.10, 5.0))
        lam_a = float(np.clip(lam_away, 0.10, 5.0))
        goals = np.arange(model.MAX_GOALS + 1)
        p_h = poisson.pmf(goals, lam_h)
        p_a = poisson.pmf(goals, lam_a)
        matrix = np.outer(p_h, p_a)

        # Standard Dixon-Coles low-score tau correction; rows are home goals,
        # columns are away goals. Rho=0 exactly recovers independent Poisson.
        matrix[0, 0] *= 1.0 - lam_h * lam_a * rho
        if len(goals) > 1:
            matrix[0, 1] *= 1.0 + lam_h * rho
            matrix[1, 0] *= 1.0 + lam_a * rho
            matrix[1, 1] *= 1.0 - rho
        if (matrix < -1e-12).any() or not np.isfinite(matrix).all():
            raise ValueError("Dixon-Coles correction produced invalid score probabilities; check rho grid.")
        matrix = np.clip(matrix, 0.0, None)
        total = float(matrix.sum())
        if total <= 0:
            raise ValueError("Dixon-Coles score matrix has zero probability mass.")
        matrix /= total

        # Home goals are matrix rows. Home wins lie below the diagonal.
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        probs_total = home_win + draw + away_win
        result["home_win"] = home_win / probs_total
        result["draw"] = draw / probs_total
        result["away_win"] = away_win / probs_total
        score_h, score_a = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        result["predicted_home_goals"] = int(score_h)
        result["predicted_away_goals"] = int(score_a)
        result["predicted_score"] = f"{score_h}–{score_a}"
        result["dixon_coles_rho"] = float(rho)
        return result
    return wrapped


def walk_forward(target_seasons: set, rho: float) -> pd.DataFrame:
    ucl = model.load_matches()
    domestic = model.load_domestic_matches()
    events = model.prepare_events(ucl, domestic, as_of=None)
    original_poisson = model._poisson_prediction
    model._poisson_prediction = make_dixon_coles_predictor(original_poisson, rho)
    state = model._History()
    past_ucl: List[Dict[str, Any]] = []
    captured: List[Dict[str, Any]] = []
    try:
        for _, group in events.groupby(events["date"].dt.normalize(), sort=True):
            base_h, base_a = recent_ucl_baseline(past_ucl)
            day_ucl_rows: List[Dict[str, Any]] = []
            for _, row in group.iterrows():
                if str(row.get("source", "")) != "ucl":
                    continue
                season = str(row.get("season", ""))
                hg, ag = int(row["home_goals"]), int(row["away_goals"])
                day_ucl_rows.append({"season": season, "home_goals": hg, "away_goals": ag})
                if season not in target_seasons:
                    continue
                pred = model._predict_with_state(
                    state, str(row["home_team"]), str(row["away_team"]), base_h, base_a
                )
                actual = "H" if hg > ag else ("D" if hg == ag else "A")
                probs = [float(pred["home_win"]), float(pred["draw"]), float(pred["away_win"])]
                actual_idx = {"H": 0, "D": 1, "A": 2}[actual]
                captured.append({
                    "date": pd.Timestamp(row["date"]).date().isoformat(),
                    "season": season,
                    "home_team": str(row["home_team"]),
                    "away_team": str(row["away_team"]),
                    "home_goals": hg,
                    "away_goals": ag,
                    "actual_result": actual,
                    "predicted_score": str(pred["predicted_score"]),
                    "home_win": probs[0], "draw": probs[1], "away_win": probs[2],
                    "actual_probability": probs[actual_idx],
                    "predicted_result": ["H", "D", "A"][int(np.argmax(probs))],
                    "rho": float(rho),
                })
            # Results from all competitions on this date are applied only after all predictions.
            for _, row in group.iterrows():
                state.update(row)
            past_ucl.extend(day_ucl_rows)
    finally:
        model._poisson_prediction = original_poisson
    return pd.DataFrame(captured)


def score_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {"n": 0, "accuracy": float("nan"), "brier": float("nan"),
                "log_loss": float("nan"), "rps": float("nan")}
    p = frame[["home_win", "draw", "away_win"]].to_numpy(dtype=float)
    actual = frame["actual_result"].astype(str).to_numpy()
    idx = np.select([actual == "H", actual == "D", actual == "A"], [0, 1, 2], default=-1)
    if (idx < 0).any() or not np.isfinite(p).all():
        raise ValueError("Invalid result labels or non-finite probabilities in predictions")
    onehot = np.eye(3)[idx]
    actual_prob = np.clip(p[np.arange(len(frame)), idx], 1e-15, 1.0)
    # RPS compares ordered outcomes: away, draw-or-away, and home/draw/away cumulative probabilities.
    ordered_p = p[:, [2, 1, 0]]
    ordered_y = onehot[:, [2, 1, 0]]
    rps = float(np.mean(np.sum(
        (np.cumsum(ordered_p, axis=1)[:, :2] - np.cumsum(ordered_y, axis=1)[:, :2]) ** 2,
        axis=1,
    ) / 2.0))
    return {
        "n": int(len(frame)),
        "accuracy": float((frame["predicted_result"].astype(str).to_numpy() == actual).mean()),
        "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(actual_prob))),
        "rps": rps,
    }


def print_metrics(name: str, frame: pd.DataFrame) -> None:
    m = score_metrics(frame)
    print(f"{name:<18} n={m['n']:>4} | acc={m['accuracy']:.1%} | "
          f"Brier={m['brier']:.4f} | logloss={m['log_loss']:.4f} | RPS={m['rps']:.4f}")


def main() -> None:
    print("Dixon-Coles low-score correction sweep (xG candidate features held fixed)")
    print("Validation: 2014-15 through 2022-23; untouched holdout: 2023-24 through 2025-26")
    print("rho=0 is the independent-Poisson version of the same xG candidate.")
    print("No datasets are modified; this may take a little while because each rho is walked forward.")

    # Confirm expected seasons are present before running.
    ucl = model.load_matches()
    available = set(ucl["season"].astype(str).unique())
    validation = VALIDATION_SEASONS.intersection(available)
    holdout = HOLDOUT_SEASONS.intersection(available)
    if len(validation) < 6 or len(holdout) < 2:
        raise SystemExit(
            f"Insufficient seasons for the prescribed split. Validation present={sorted(validation)}; "
            f"holdout present={sorted(holdout)}"
        )

    frames: Dict[float, pd.DataFrame] = {}
    selection_rows = []
    for rho in RHO_GRID:
        frame = walk_forward(validation | holdout, rho)
        if frame.empty:
            raise SystemExit(f"No predictions generated for rho={rho}")
        frames[rho] = frame
        val = frame[frame["season"].isin(validation)]
        test = frame[frame["season"].isin(holdout)]
        vm = score_metrics(val)
        tm = score_metrics(test)
        selection_rows.append({
            "rho": rho,
            **{"validation_" + k: v for k, v in vm.items()},
            **{"holdout_" + k: v for k, v in tm.items()},
        })
        print(f"\nrho={rho:+.2f}")
        print_metrics("Validation", val)
        print_metrics("HOLDOUT", test)

    summary = pd.DataFrame(selection_rows)
    # Select only on validation log loss; report the held-out metrics for that selected rho separately.
    best = summary.sort_values(["validation_log_loss", "validation_brier"], ascending=True).iloc[0]
    chosen_rho = float(best["rho"])
    chosen_frame = frames[chosen_rho]
    all_preds_path = ROOT / "data" / "dixon_coles_sweep_predictions.csv"
    summary_path = ROOT / "data" / "dixon_coles_sweep_summary.csv"
    chosen_frame.to_csv(all_preds_path, index=False)
    summary.to_csv(summary_path, index=False)
    print("\nSELECTION")
    print(f"rho selected using validation log loss only: {chosen_rho:+.2f}")
    print_metrics("Selected validation", chosen_frame[chosen_frame["season"].isin(validation)])
    print_metrics("Selected HOLDOUT", chosen_frame[chosen_frame["season"].isin(holdout)])
    print("\nSaved sweep summary:", summary_path)
    print("Saved predictions for selected rho:", all_preds_path)
    print("Important: the holdout is reported, not used to choose rho. Do not retune against it.")


if __name__ == "__main__":
    main()
