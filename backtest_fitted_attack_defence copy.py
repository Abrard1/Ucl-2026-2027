#!/usr/bin/env python3
"""Experimental, time-decayed, maximum-likelihood team attack/defence model.

This is a separate candidate experiment. It does not edit predictor.py or any
source datasets. It uses prior domestic and Champions League goals, competition-
specific home/away scoring baselines, and ridge shrinkage for teams with limited
history. It compares the candidate with saved heuristic-model predictions when
those files are present.

Python 3.9+; uses the existing predictor_xg.py and local data files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    import predictor_xg as current
except Exception as exc:
    raise SystemExit(
        "Could not import predictor_xg.py. Keep this script in ucl/scripts/ and "
        "predictor_xg.py in the project root.\n" + str(exc)
    )

TRAIN_WINDOW_DAYS = 1825        # five years of prior match history
DECAY_HALF_LIFE_DAYS = 365.0   # fixed before the holdout is scored
TEAM_RIDGE = 6.0               # shrink team attack/defence towards league-neutral
BASELINE_RIDGE = 4.0           # shrink competition baselines towards observed averages
MAX_GOALS = 10
UCL_COMP = "UEFA Champions League"
VALIDATION_SEASONS = {
    "2014-15", "2015-16", "2016-17", "2017-18", "2018-19",
    "2019-20", "2020-21", "2021-22", "2022-23",
}
HOLDOUT_SEASONS = {"2023-24", "2024-25", "2025-26"}


def competition_family(row: pd.Series) -> str:
    if str(row.get("source", "")) == "ucl":
        return UCL_COMP
    value = str(row.get("competition", "Domestic league")).strip()
    # Merge season-specific labels (e.g. "English Premier League 2026/27")
    # into a stable competition family so the model can learn across seasons.
    value = re.sub(r"\s+(?:(?:19|20)\d{2}[-/]\d{2,4}|(?:19|20)\d{2})$", "", value).strip()
    return value or "Domestic league"


@dataclass
class FittedTeamModel:
    teams: List[str]
    competitions: List[str]
    attack: np.ndarray
    defence: np.ndarray
    home_base: np.ndarray
    away_base: np.ndarray
    fit_rows: int
    train_start: str
    train_end: str

    def predict(self, home_key: str, away_key: str, comp: str = UCL_COMP) -> Dict[str, float]:
        team_index = {t: i for i, t in enumerate(self.teams)}
        comp_index = {c: i for i, c in enumerate(self.competitions)}
        hi = team_index.get(home_key)
        ai = team_index.get(away_key)
        ci = comp_index.get(comp)
        # Unknown teams get a neutral, shrunk-to-average rating.
        ah = self.attack[hi] if hi is not None else 0.0
        dh = self.defence[hi] if hi is not None else 0.0
        aa = self.attack[ai] if ai is not None else 0.0
        da = self.defence[ai] if ai is not None else 0.0
        if ci is None:
            raise ValueError("No prior matches for the Champions League competition baseline.")
        lam_h = float(np.exp(self.home_base[ci] + ah - da))
        lam_a = float(np.exp(self.away_base[ci] + aa - dh))
        lam_h = float(np.clip(lam_h, 0.05, 6.0))
        lam_a = float(np.clip(lam_a, 0.05, 6.0))
        goals = np.arange(MAX_GOALS + 1)
        matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
        matrix /= max(float(matrix.sum()), 1e-15)
        p_home = float(np.tril(matrix, k=-1).sum())
        p_draw = float(np.trace(matrix))
        p_away = float(np.triu(matrix, k=1).sum())
        total = p_home + p_draw + p_away
        score_h, score_a = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        return {
            "home_win": p_home / total,
            "draw": p_draw / total,
            "away_win": p_away / total,
            "expected_home_goals": lam_h,
            "expected_away_goals": lam_a,
            "predicted_home_goals": int(score_h),
            "predicted_away_goals": int(score_a),
            "predicted_score": f"{score_h}–{score_a}",
        }


def _prior_log_rates(y: np.ndarray, comp_idx: np.ndarray, weights: np.ndarray,
                     n_comp: int, fallback: float, pseudo_n: float = 8.0) -> np.ndarray:
    out = np.zeros(n_comp, dtype=float)
    global_mean = float(np.average(y, weights=weights)) if len(y) else fallback
    if not np.isfinite(global_mean) or global_mean <= 0:
        global_mean = fallback
    for j in range(n_comp):
        mask = comp_idx == j
        if mask.any():
            wsum = float(weights[mask].sum())
            mean = (float(np.dot(weights[mask], y[mask])) + pseudo_n * global_mean) / (wsum + pseudo_n)
        else:
            mean = global_mean
        out[j] = np.log(float(np.clip(mean, 0.25, 3.8)))
    return out


def fit_attack_defence(train: pd.DataFrame, as_of: pd.Timestamp) -> FittedTeamModel:
    if train.empty:
        raise ValueError("No training matches available before target date.")
    frame = train.copy()
    frame["competition_model"] = frame.apply(competition_family, axis=1)
    frame["home_key"] = frame["home_team"].map(current.canonical_team)
    frame["away_key"] = frame["away_team"].map(current.canonical_team)
    frame["home_goals"] = pd.to_numeric(frame["home_goals"], errors="coerce")
    frame["away_goals"] = pd.to_numeric(frame["away_goals"], errors="coerce")
    frame = frame.dropna(subset=["date", "home_goals", "away_goals"])
    frame = frame[(frame["home_goals"] >= 0) & (frame["away_goals"] >= 0)]
    if len(frame) < 50:
        raise ValueError(f"Not enough training matches before {as_of.date()}: {len(frame)}")

    teams = sorted(set(frame["home_key"]) | set(frame["away_key"]))
    competitions = sorted(set(frame["competition_model"]))
    team_index = {v: i for i, v in enumerate(teams)}
    comp_index = {v: i for i, v in enumerate(competitions)}
    hi = frame["home_key"].map(team_index).to_numpy(dtype=int)
    ai = frame["away_key"].map(team_index).to_numpy(dtype=int)
    ci = frame["competition_model"].map(comp_index).to_numpy(dtype=int)
    yh = frame["home_goals"].to_numpy(dtype=float)
    ya = frame["away_goals"].to_numpy(dtype=float)
    age_days = (as_of.normalize() - frame["date"].dt.normalize()).dt.days.to_numpy(dtype=float)
    weights = np.exp(-np.log(2.0) * np.maximum(age_days, 0.0) / DECAY_HALF_LIFE_DAYS)
    weights *= len(weights) / max(float(weights.sum()), 1e-12)

    n_team, n_comp = len(teams), len(competitions)
    bh_prior = _prior_log_rates(yh, ci, weights, n_comp, fallback=1.45)
    ba_prior = _prior_log_rates(ya, ci, weights, n_comp, fallback=1.20)
    # Parameter vector: all team attack effects, team defence effects, then
    # competition-specific log home and away goal baselines.
    def unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (x[:n_team], x[n_team:2*n_team],
                x[2*n_team:2*n_team+n_comp], x[2*n_team+n_comp:2*n_team+2*n_comp])

    x0 = np.concatenate([np.zeros(n_team), np.zeros(n_team), bh_prior, ba_prior])
    bounds = ([(-1.2, 1.2)] * (2*n_team) + [(-1.4, 1.4)] * (2*n_comp))

    def objective(x: np.ndarray) -> Tuple[float, np.ndarray]:
        att, deff, bh, ba = unpack(x)
        log_h = bh[ci] + att[hi] - deff[ai]
        log_a = ba[ci] + att[ai] - deff[hi]
        # Bounds keep rates in a numerically safe range; this is still the standard
        # weighted Poisson negative log likelihood up to response-only constants.
        log_h = np.clip(log_h, -5.0, 4.0)
        log_a = np.clip(log_a, -5.0, 4.0)
        lam_h, lam_a = np.exp(log_h), np.exp(log_a)
        loss = float(np.sum(weights * (lam_h - yh * log_h + lam_a - ya * log_a)))
        loss += 0.5 * TEAM_RIDGE * float(np.dot(att, att) + np.dot(deff, deff))
        loss += 0.5 * BASELINE_RIDGE * float(np.dot(bh - bh_prior, bh - bh_prior) + np.dot(ba - ba_prior, ba - ba_prior))

        err_h = weights * (lam_h - yh) * ((bh[ci] + att[hi] - deff[ai] > -5.0) & (bh[ci] + att[hi] - deff[ai] < 4.0))
        err_a = weights * (lam_a - ya) * ((ba[ci] + att[ai] - deff[hi] > -5.0) & (ba[ci] + att[ai] - deff[hi] < 4.0))
        grad_att = np.bincount(hi, weights=err_h, minlength=n_team) + np.bincount(ai, weights=err_a, minlength=n_team) + TEAM_RIDGE * att
        grad_def = -np.bincount(ai, weights=err_h, minlength=n_team) - np.bincount(hi, weights=err_a, minlength=n_team) + TEAM_RIDGE * deff
        grad_bh = np.bincount(ci, weights=err_h, minlength=n_comp) + BASELINE_RIDGE * (bh - bh_prior)
        grad_ba = np.bincount(ci, weights=err_a, minlength=n_comp) + BASELINE_RIDGE * (ba - ba_prior)
        grad = np.concatenate([grad_att, grad_def, grad_bh, grad_ba])
        return loss, grad

    result = minimize(objective, x0, jac=True, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 160, "ftol": 1e-8})
    if not np.isfinite(result.fun):
        raise RuntimeError(f"Optimisation failed for cutoff {as_of.date()}: non-finite objective")
    att, deff, bh, ba = unpack(result.x)
    return FittedTeamModel(
        teams=teams, competitions=competitions, attack=att.copy(), defence=deff.copy(),
        home_base=bh.copy(), away_base=ba.copy(), fit_rows=len(frame),
        train_start=frame["date"].min().date().isoformat(),
        train_end=frame["date"].max().date().isoformat(),
    )


def metrics(frame: pd.DataFrame) -> Dict[str, float]:
    if frame.empty:
        return {"n": 0, "accuracy": np.nan, "brier": np.nan, "log_loss": np.nan, "rps": np.nan}
    probs = frame[["home_win", "draw", "away_win"]].to_numpy(dtype=float)
    actual = frame["actual_result"].astype(str).to_numpy()
    idx = np.select([actual == "H", actual == "D", actual == "A"], [0, 1, 2], default=-1)
    if (idx < 0).any() or not np.isfinite(probs).all():
        raise ValueError("Invalid target labels/probabilities in score calculation")
    onehot = np.eye(3)[idx]
    actual_p = np.clip(probs[np.arange(len(frame)), idx], 1e-15, 1.0)
    ordered_p = probs[:, [2, 1, 0]]
    ordered_y = onehot[:, [2, 1, 0]]
    rps = float(np.mean(np.sum((np.cumsum(ordered_p, axis=1)[:, :2] - np.cumsum(ordered_y, axis=1)[:, :2]) ** 2, axis=1) / 2.0))
    return {
        "n": int(len(frame)),
        "accuracy": float((frame["predicted_result"].to_numpy() == actual).mean()),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "log_loss": float(-np.mean(np.log(actual_p))),
        "rps": rps,
    }


def print_metrics(label: str, frame: pd.DataFrame) -> None:
    m = metrics(frame)
    print(f"{label:<22} n={m['n']:>4} | acc={m['accuracy']:.1%} | Brier={m['brier']:.4f} | logloss={m['log_loss']:.4f} | RPS={m['rps']:.4f}")


def _score_file(path: Path, label: str, keys: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    needed = {"date", "season", "home_team", "away_team", "home_goals", "away_goals", "home_win", "draw", "away_win"}
    if not needed.issubset(frame.columns):
        print(f"Skipping saved comparator {path.name}: expected columns are missing")
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype(str)
    frame["_home"] = frame["home_team"].map(current.canonical_team)
    frame["_away"] = frame["away_team"].map(current.canonical_team)
    out = frame.merge(keys, on=["date", "season", "_home", "_away"], how="inner", suffixes=("", "_new"))
    if out.empty:
        print(f"Saved comparator {path.name} did not match candidate rows")
        return None
    out["actual_result"] = np.where(out["home_goals"] > out["away_goals"], "H", np.where(out["home_goals"] == out["away_goals"], "D", "A"))
    out["predicted_result"] = np.array(["H", "D", "A"])[np.argmax(out[["home_win", "draw", "away_win"]].to_numpy(float), axis=1)]
    return out


def main() -> None:
    print("Fitted, time-decayed attack/defence Poisson model — experimental candidate")
    print("Training: prior 5 years only; half-life 365 days; MLE with ridge shrinkage")
    print("No Dixon-Coles correction (the prior rho sweep selected zero). This is a different model architecture.")
    ucl = current.load_matches()
    domestic = current.load_domestic_matches()
    events = current.prepare_events(ucl, domestic, as_of=None)
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    events["season"] = events["season"].fillna("").astype(str)
    events["home_key"] = events["home_team"].map(current.canonical_team)
    events["away_key"] = events["away_team"].map(current.canonical_team)
    events = events.dropna(subset=["date", "home_goals", "away_goals"]).sort_values("date").reset_index(drop=True)
    ucl_targets = events[(events["source"] == "ucl") & (events["season"].isin(VALIDATION_SEASONS | HOLDOUT_SEASONS))].copy()
    if ucl_targets.empty:
        raise SystemExit("No UCL target matches found in the expected 2014/15–2025/26 seasons.")

    output: List[Dict[str, object]] = []
    date_groups = list(ucl_targets.groupby(ucl_targets["date"].dt.normalize(), sort=True))
    for n, (date_value, target_day) in enumerate(date_groups, start=1):
        cutoff = pd.Timestamp(date_value).normalize()
        start = cutoff - pd.Timedelta(days=TRAIN_WINDOW_DAYS)
        train = events[(events["date"] < cutoff) & (events["date"] >= start)].copy()
        # Ignore any future rows and ensure same-day outcomes cannot leak into predictions.
        fitted = fit_attack_defence(train, cutoff)
        print(f"[{n:03d}/{len(date_groups):03d}] {cutoff.date()} | train={fitted.fit_rows:,} matches", flush=True)
        for _, row in target_day.iterrows():
            pred = fitted.predict(str(row["home_key"]), str(row["away_key"]), UCL_COMP)
            hg, ag = int(row["home_goals"]), int(row["away_goals"])
            actual = "H" if hg > ag else ("D" if hg == ag else "A")
            probs = [pred["home_win"], pred["draw"], pred["away_win"]]
            output.append({
                "date": cutoff.date().isoformat(), "season": str(row["season"]),
                "home_team": str(row["home_team"]), "away_team": str(row["away_team"]),
                "home_goals": hg, "away_goals": ag, "actual_result": actual,
                "home_win": probs[0], "draw": probs[1], "away_win": probs[2],
                "expected_home_goals": pred["expected_home_goals"],
                "expected_away_goals": pred["expected_away_goals"],
                "predicted_score": pred["predicted_score"],
                "predicted_result": ["H", "D", "A"][int(np.argmax(probs))],
                "training_rows": fitted.fit_rows, "training_start": fitted.train_start,
                "training_end": fitted.train_end,
            })

    result = pd.DataFrame(output)
    out_path = ROOT / "data" / "backtest_fitted_attack_defence.csv"
    result.to_csv(out_path, index=False)
    # Normalised identity keys are required by the saved-model comparator below.
    # They are added after saving so the public result CSV stays compact.
    result["_home"] = result["home_team"].map(current.canonical_team)
    result["_away"] = result["away_team"].map(current.canonical_team)
    val = result[result["season"].isin(VALIDATION_SEASONS)]
    hold = result[result["season"].isin(HOLDOUT_SEASONS)]
    print("\nRESULTS")
    print_metrics("Validation candidate", val)
    print_metrics("HOLDOUT candidate", hold)

    for fname, label in [
        ("backtest_live_equivalent_original.csv", "Existing heuristic"),
        ("backtest_live_equivalent_xg.csv", "Existing xG candidate"),
    ]:
        comparator_keys = result[["season", "_home", "_away"]].copy()
        comparator_keys["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date.astype(str).to_numpy()
        prior = _score_file(ROOT / "data" / fname, label, comparator_keys[["date", "season", "_home", "_away"]])
        if prior is not None:
            prior_hold = prior[prior["season"].isin(HOLDOUT_SEASONS)]
            print_metrics(label + " holdout", prior_hold)

    summary = []
    for season, group in result.groupby("season", sort=True):
        m = metrics(group)
        summary.append({"season": season, **m})
    summary_path = ROOT / "data" / "backtest_fitted_attack_defence_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print("\nSaved match predictions:", out_path)
    print("Saved season summary:", summary_path)
    print("This is an experimental fitted-goals-only candidate. It does not automatically replace the live predictor.")


if __name__ == "__main__":
    main()
