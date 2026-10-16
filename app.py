from __future__ import annotations
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor import fit_model, load_matches, predict_match

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIXTURE_FILE = DATA_DIR / "fixtures.csv"
MATCH_FILE = DATA_DIR / "matches.csv"
DB_FILE = ROOT / "predictions.db"

st.set_page_config(page_title="UCL Predictor 26/27", page_icon="⚽", layout="wide")
st.markdown("""
<style>
    .block-container {padding-top: 1.7rem; max-width: 1400px;}
    [data-testid="stMetric"] {background: rgba(120,140,160,.08); border: 1px solid rgba(120,140,160,.18); padding: 14px; border-radius: 12px;}
    .hero {padding: 1rem 0 .5rem 0;}
    .subtle {color: #8b95a7;}
</style>
""", unsafe_allow_html=True)


def init_db() -> None:
    with sqlite3.connect(DB_FILE) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS saved_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            saved_at TEXT DEFAULT CURRENT_TIMESTAMP,
            match_date TEXT NOT NULL,
            home_team TEXT NOT NULL,
            away_team TEXT NOT NULL,
            predicted_score TEXT NOT NULL,
            home_win REAL NOT NULL,
            draw_prob REAL NOT NULL,
            away_win REAL NOT NULL,
            expected_home_goals REAL NOT NULL,
            expected_away_goals REAL NOT NULL,
            model_seasons TEXT,
            UNIQUE(match_date, home_team, away_team, model_seasons)
        )""")


def saved_rows() -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_FILE) as con:
        return pd.read_sql_query("SELECT * FROM saved_predictions ORDER BY match_date, home_team", con)


def save_all(predictions: pd.DataFrame, seasons_label: str) -> int:
    init_db()
    inserted = 0
    with sqlite3.connect(DB_FILE) as con:
        for _, r in predictions.iterrows():
            cursor = con.execute("""INSERT OR IGNORE INTO saved_predictions
                (match_date, home_team, away_team, predicted_score, home_win, draw_prob, away_win,
                 expected_home_goals, expected_away_goals, model_seasons)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(r["date"]), r["home_team"], r["away_team"], r["predicted_score"],
                 float(r["home_win"]), float(r["draw"]), float(r["away_win"]),
                 float(r["expected_home_goals"]), float(r["expected_away_goals"]), seasons_label))
            inserted += max(cursor.rowcount, 0)
    return inserted


st.markdown("<div class='hero'>", unsafe_allow_html=True)
st.title("⚽ UCL Predictor 26/27")
st.caption("A transparent football prediction lab · historical data in, testable probabilities out")
st.markdown("</div>", unsafe_allow_html=True)

if not FIXTURE_FILE.exists():
    st.error("Fixture file not found: data/fixtures.csv")
    st.stop()
fixtures = pd.read_csv(FIXTURE_FILE, parse_dates=["date"])

if not MATCH_FILE.exists() or MATCH_FILE.stat().st_size == 0:
    st.warning("Historical results have not been imported yet. Follow the three setup steps below to load free OpenFootball data.")
    st.code("git clone https://github.com/openfootball/champions-league.git data/raw/champions-league\npython scripts/import_openfootball.py", language="powershell")
    st.markdown("**Next:** install the packages, run the data importer, then launch `streamlit run app.py`. The app will calculate predictions only after historical results are available; it will not invent probabilities from missing data.")
    st.subheader("Upcoming fixtures")
    view = fixtures.copy()
    view["date"] = view["date"].dt.strftime("%a %d %b %Y")
    st.dataframe(view[["date", "kickoff", "home_team", "away_team"]], use_container_width=True, hide_index=True)
    st.stop()

try:
    matches = load_matches(str(MATCH_FILE))
    lookback = st.sidebar.slider("Historical seasons to use", min_value=1, max_value=10, value=5,
                                 help="Uses the most recent seasons in the imported dataset.")
    model = fit_model(matches, seasons=lookback)
except Exception as e:
    st.error(f"Could not prepare the model: {e}")
    st.info("Check data/matches.csv and run the importer again if needed.")
    st.stop()

pred_rows = []
for _, fixture in fixtures.iterrows():
    p = predict_match(model, fixture["home_team"], fixture["away_team"])
    pred_rows.append({**fixture.to_dict(), **p})
predictions = pd.DataFrame(pred_rows)
season_txt = ", ".join(model.seasons_used) if model.seasons_used else f"last {lookback} seasons"

c1, c2, c3, c4 = st.columns(4)
c1.metric("UCL matches in baseline", f"{model.match_count:,}")
c2.metric("Teams in model", f"{len(model.team_stats):,}")
c3.metric("Fixtures", f"{len(predictions):,}")
c4.metric("Model", "Form + Elo + Poisson")
st.caption(f"UCL seasons used for scoring baseline: {season_txt}. Team features combine UCL history, recent domestic form and Elo ratings. Data available through {model.data_as_of}.")
if model.domestic_match_count == 0:
    st.warning("Domestic match data was not loaded. Run scripts/import_football_json.py to enable domestic-form features.")

page = st.radio("View", ["Predictions", "Saved predictions", "About the model"], horizontal=True, label_visibility="collapsed")
if page == "Predictions":
    st.subheader("Matchday 3 predictions")
    table = predictions.copy()
    table["date"] = table["date"].dt.strftime("%d %b")
    table["H win"] = (table["home_win"] * 100).round(1).astype(str) + "%"
    table["Draw"] = (table["draw"] * 100).round(1).astype(str) + "%"
    table["A win"] = (table["away_win"] * 100).round(1).astype(str) + "%"
    table["Expected goals"] = table.apply(lambda r: f"{r['expected_home_goals']:.2f} – {r['expected_away_goals']:.2f}", axis=1)
    table = table.rename(columns={"date": "Date", "kickoff": "Kickoff", "home_team": "Home", "away_team": "Away", "predicted_score": "Likeliest score"})
    st.dataframe(table[["Date", "UK kickoff", "Home", "Away", "Likeliest score", "Expected goals", "H win", "Draw", "A win"]], use_container_width=True, hide_index=True)

    st.subheader("Explore a fixture")
    selected = st.selectbox("Choose match", predictions.index,
        format_func=lambda i: f"{predictions.loc[i, 'home_team']} vs {predictions.loc[i, 'away_team']} — {predictions.loc[i, 'date']:%d %b}")
    r = predictions.loc[selected]
    left, centre, right = st.columns([2, 1, 2])
    with left:
        st.markdown(f"### {r['home_team']}")
        st.metric("Home-win probability", f"{r['home_win']:.1%}")
        st.metric("Expected home goals", f"{r['expected_home_goals']:.2f}")
    with centre:
        st.markdown("### Score")
        st.markdown(f"<h1 style='text-align:center'>{r['predicted_score']}</h1>", unsafe_allow_html=True)
        st.caption("Most likely exact score")
    with right:
        st.markdown(f"### {r['away_team']}")
        st.metric("Away-win probability", f"{r['away_win']:.1%}")
        st.metric("Expected away goals", f"{r['expected_away_goals']:.2f}")
    st.bar_chart(pd.DataFrame({"Probability": [r["home_win"], r["draw"], r["away_win"]]}, index=["Home win", "Draw", "Away win"]))
    unknown = []
    if not r["home_known"]: unknown.append(str(r["home_team"]))
    if not r["away_known"]: unknown.append(str(r["away_team"]))
    if unknown:
        st.warning("No matching historic team record was found for " + " and ".join(unknown) + ". Average league scoring rates are used for the missing team. Check team-name aliases before trusting this fixture.")

    if st.button("Save all fixture predictions to history", type="primary"):
        count = save_all(predictions, season_txt)
        st.success(f"Saved {count} new predictions. Existing duplicates were skipped.")

elif page == "Saved predictions":
    st.subheader("Prediction log")
    saved = saved_rows()
    if saved.empty:
        st.info("No saved predictions yet. Open Predictions and click 'Save all fixture predictions to history'.")
    else:
        for col in ["home_win", "draw_prob", "away_win"]:
            saved[col] = (saved[col] * 100).round(1)
        st.dataframe(saved.rename(columns={"match_date": "Date", "home_team": "Home", "away_team": "Away", "predicted_score": "Predicted score", "home_win": "Home win %", "draw_prob": "Draw %", "away_win": "Away win %", "saved_at": "Saved at"}), use_container_width=True, hide_index=True)
        st.caption("Next planned feature: enter completed scores and evaluate the predictions after each matchday.")

else:
    st.subheader("How the baseline model works")
    st.markdown("""
    1. Uses the selected recent UCL seasons to set the scoring baseline.
    2. Builds rolling attack/defence form from the most recent domestic and UCL matches. Each team rate is normalized against the relevant competition baseline.
    3. Updates a global Elo rating chronologically from domestic and UCL results.
    4. Uses the team-form rates and a restrained Elo adjustment to estimate expected goals.
    5. Applies Poisson goal distributions to calculate home-win, draw and away-win probabilities, plus the likeliest score.

    **Limitations:** this version does not use xG, shots, line-ups, injuries, travel, or weather. The domestic source currently ends on 20 September 2026; current-season UCL Matchdays 1 and 2 are included separately. Club aliases are normalized, but all data coverage still needs manual checking. The backtest uses chronological predictions and should be compared with its baseline before model performance is trusted.
    """)
