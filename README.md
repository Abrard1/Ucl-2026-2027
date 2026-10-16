# UCL Predictor 26/27

A small, explainable Python project for predicting UEFA Champions League fixture outcomes from historic Champions League results. It starts with a transparent Poisson goal model and a Streamlit dashboard.

## Data source (free)

This project is designed around the OpenFootball Champions League repository: <https://github.com/openfootball/champions-league>. The repository describes its datasets as public domain / CC0. The data is not bundled in this ZIP; you download it directly from the source.

## Setup on Windows / VS Code

Open a terminal in this project folder and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download the free historical data into the expected location (requires Git):

```powershell
git clone https://github.com/openfootball/champions-league.git data/raw/champions-league
python scripts/import_openfootball.py
```

The importer reads completed match lines from season `.txt` files and creates `data/matches.csv`. Review the output CSV once before relying on predictions because football-data formats can vary between seasons.

If Git is not installed, download the repository ZIP from the GitHub page, extract it, and place its season folders under `data/raw/champions-league/`.

## Launch

```powershell
streamlit run app.py
```

Streamlit will print a local URL (normally `http://localhost:8501`). Open it in your browser.

## What's included

- Upcoming Matchday 3 fixture slate for 20–21 October 2026 (kickoff times displayed in UK time)
- OpenFootball text-data importer
- Smoothed home/away attack and defence estimates
- Poisson scoreline and home/draw/away probabilities
- Streamlit dashboard and SQLite prediction history
- Small tests for probability and fallback behaviour

## Model warning

This is a baseline model, not a betting recommendation and not a claim of reliable prediction. It uses historical Champions League goals and does not yet include line-ups, injuries, domestic league form, Elo ratings, or betting-market probabilities. Backtest chronologically before assessing performance. Unknown team names fall back toward league-average rates; check warnings in the app.

## Four-session build plan

1. Data: run the importer, inspect coverage, fix team-name mapping.
2. Model: backtest against held-out seasons and compare with simple home/draw/away baselines.
3. Interface: improve fixture details and add form charts.
4. Evaluation: store final scores after matchday, calculate log loss / Brier score, document limitations.
