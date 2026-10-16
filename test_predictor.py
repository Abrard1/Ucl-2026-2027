import pandas as pd
from predictor import fit_model, predict_match, canonical_team


def sample_matches():
    return pd.DataFrame([
        {"date":"2024-09-01", "season":"2024-25", "home_team":"Alpha FC", "away_team":"Beta FC", "home_goals":2, "away_goals":0},
        {"date":"2024-09-02", "season":"2024-25", "home_team":"Beta FC", "away_team":"Alpha FC", "home_goals":1, "away_goals":1},
        {"date":"2025-09-01", "season":"2025-26", "home_team":"Alpha FC", "away_team":"Gamma FC", "home_goals":3, "away_goals":1},
        {"date":"2025-09-02", "season":"2025-26", "home_team":"Gamma FC", "away_team":"Beta FC", "home_goals":0, "away_goals":2},
    ])


def test_canonical_team_ignores_common_prefixes():
    assert canonical_team("FC Barcelona") == canonical_team("Barcelona")


def test_predictor_probabilities_sum_to_one():
    model = fit_model(sample_matches(), seasons=5)
    p = predict_match(model, "Alpha FC", "Beta FC")
    assert abs(p["home_win"] + p["draw"] + p["away_win"] - 1.0) < 1e-9
    assert 0 <= p["home_win"] <= 1
    assert 0 <= p["draw"] <= 1
    assert 0 <= p["away_win"] <= 1
    assert "–" in p["predicted_score"]


def test_unseen_teams_fall_back_to_league_average():
    model = fit_model(sample_matches(), seasons=5)
    p = predict_match(model, "Unknown Home", "Unknown Away")
    assert not p["home_known"]
    assert not p["away_known"]
    assert abs(p["expected_home_goals"] - model.avg_home_goals) < 1e-9
