# ──────────────────────────────────────────────────────────────────────────────
# Euro Soccer Value Model (EPL + EFL Championship) – Starter Scaffold
# Author: You
# License: MIT (adjust as you prefer)
# ──────────────────────────────────────────────────────────────────────────────
# This single document contains a minimal, production‑leaning scaffold for a
# European football (soccer) model focusing on:
#   • 1X2 probabilities (home/draw/away)
#   • Totals (Over/Under) via Poisson goal modeling
#   • EV (expected value) detection vs. bookmaker odds
#   • Support for EPL and the EFL Championship (England Tier 1 & 2)
#   • FastAPI (JSON) + Streamlit (dashboard) frontends
#
# Files below are presented in a monorepo style. Copy each block into its own
# file matching the path in the header, then follow the README at the end.
# ──────────────────────────────────────────────────────────────────────────────

# ============================================================================
# File: requirements.txt
# ============================================================================
# Pin loosely for dev; tighten when deploying.
# core
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
statsmodels>=0.14
# web/api
requests>=2.31
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.6
httpx>=0.27
# ui
streamlit>=1.36
# plots (optional)
plotly>=5.20
# config
python-dotenv>=1.0
# auth & billing
python-jose[cryptography]>=3.3
supabase>=2.4
stripe>=6.0
cachetools>=5.3
# testing
pytest>=8.2
pytest-cov>=4.1

# ============================================================================
# File: .env.example
# ============================================================================
API_FOOTBALL_KEY="replace_with_your_key"

# Supabase (Auth + DB)
SUPABASE_URL="https://<project-ref>.supabase.co"
SUPABASE_ANON_KEY="public_anon_key_for_frontend"
SUPABASE_SERVICE_ROLE="service_role_key_server_only"
# JWKS URL usually: ${SUPABASE_URL}/auth/v1/keys
SUPABASE_JWKS_URL="https://<project-ref>.supabase.co/auth/v1/keys"

# Stripe (Billing)
STRIPE_API_KEY="sk_live_or_test"
STRIPE_WEBHOOK_SECRET="whsec_..."
STRIPE_PRICE_ID="price_..."   # your subscription price id
SITE_URL="https://footyev.com"

ENV="dev"

# ============================================================================

# Pin loosely for dev; tighten when deploying.
# core
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
statsmodels>=0.14
# web/api
requests>=2.31
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.6
# ui
streamlit>=1.36
# plots (optional)
plotly>=5.20
# config
python-dotenv>=1.0
# testing
pytest>=8.2
pytest-cov>=4.1

# ============================================================================

# Pin loosely for dev; tighten when deploying.
# core
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
statsmodels>=0.14
# web/api
requests>=2.31
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.6
# ui
streamlit>=1.36
# plots (optional)
plotly>=5.20
# config
python-dotenv>=1.0

# ============================================================================
# File: .env.example
# ============================================================================
API_FOOTBALL_KEY="replace_with_your_key"
SPORTMONKS_TOKEN="optional_if_using_sportmonks"
# Deployment / runtime
ENV="dev"

# ============================================================================

# Pin loosely for dev; tighten when deploying.
# core
pandas>=2.0
numpy>=1.24
scipy>=1.10
scikit-learn>=1.3
# web/api
requests>=2.31
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.6
# ui
streamlit>=1.36
# plots (optional)
plotly>=5.20
# config
python-dotenv>=1.0

# ============================================================================
# File: .env.example
# ============================================================================
API_FOOTBALL_KEY="replace_with_your_key"
SPORTMONKS_TOKEN="optional_if_using_sportmonks"
# Deployment / runtime
ENV="dev"

# ============================================================================
# File: config.py
# ============================================================================
from dataclasses import dataclass
import os

@dataclass
class Settings:
    api_football_key: str = os.getenv("API_FOOTBALL_KEY", "")
    sportmonks_token: str = os.getenv("SPORTMONKS_TOKEN", "")
    env: str = os.getenv("ENV", "dev")
    # data dirs
    data_dir: str = os.getenv("DATA_DIR", "./data")
    raw_dir: str = os.path.join(data_dir, "raw")
    proc_dir: str = os.path.join(data_dir, "processed")

settings = Settings()

os.makedirs(settings.raw_dir, exist_ok=True)
os.makedirs(settings.proc_dir, exist_ok=True)

# ============================================================================
# File: data_schemas.py
# ============================================================================
from typing import TypedDict, Optional

class MatchRow(TypedDict):
    match_id: int
    league: str          # e.g., 'EPL', 'CHAMP'
    season: int          # e.g., 2024
    date: str            # ISO date
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    # odds (closing or latest pre‑match)
    book_home_odds: Optional[float]
    book_draw_odds: Optional[float]
    book_away_odds: Optional[float]

# ============================================================================
# File: ingest_api.py
# ============================================================================
"""Lightweight ingestion for fixtures/results/odds from API‑Football (or stub).

This module includes a simple client and two helpers:
  • fetch_results(league_id, season)
  • fetch_odds(match_id)
Store raw JSON under data/raw, and create tidy CSVs under data/processed.
"""
from __future__ import annotations
import os, time, json
import pandas as pd
import requests
from config import settings

API_BASE = "https://v3.football.api-sports.io"
LEAGUES = {
    # API‑Football league IDs (verify): EPL=39, Championship=40
    "EPL": 39,
    "CHAMP": 40,
}

HEADERS = {"x-rapidapi-key": settings.api_football_key}

class APIFootball:
    def __init__(self, api_key: str | None = None):
        key = api_key or settings.api_football_key
        if not key:
            raise RuntimeError("API_FOOTBALL_KEY missing. Set it in .env")
        self.headers = {"x-rapidapi-key": key}

    def get(self, path: str, params: dict) -> dict:
        url = f"{API_BASE}/{path}"
        r = requests.get(url, params=params, headers=self.headers, timeout=20)
        r.raise_for_status()
        return r.json()


def fetch_results(league_code: str, season: int) -> pd.DataFrame:
    api = APIFootball()
    league_id = LEAGUES[league_code]
    data = api.get("fixtures", {"league": league_id, "season": season, "status": "FT"})
    rows = []
    for it in data.get("response", []):
        fixture = it.get("fixture", {})
        teams = it.get("teams", {})
        goals = it.get("goals", {})
        rows.append({
            "match_id": fixture.get("id"),
            "league": league_code,
            "season": season,
            "date": fixture.get("date", "").split("T")[0],
            "home_team": teams.get("home", {}).get("name"),
            "away_team": teams.get("away", {}).get("name"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(settings.proc_dir, f"matches_{league_code}_{season}.csv")
    df.to_csv(out, index=False)
    return df


def fetch_odds_for_matches(match_ids: list[int]) -> pd.DataFrame:
    api = APIFootball()
    rows = []
    for mid in match_ids:
        data = api.get("odds", {"fixture": mid, "bookmaker": 8})  # 8=Bet365 as example
        # Parse the 1X2 market if present
        for it in data.get("response", []):
            leagues = it.get("league", {})
            bkts = it.get("bookmakers", [])
            if not bkts:
                continue
            markets = bkts[0].get("bets", [])
            m1x2 = next((m for m in markets if m.get("name") in ["Match Winner", "1X2"]), None)
            if not m1x2:
                continue
            price_map = {o["value"].upper(): float(o["odd"]) for o in m1x2.get("values", [])}
            rows.append({
                "match_id": mid,
                "book_home_odds": price_map.get("1"),
                "book_draw_odds": price_map.get("X"),
                "book_away_odds": price_map.get("2"),
            })
        time.sleep(0.5)  # be gentle
    return pd.DataFrame(rows)


# ============================================================================
# File: features.py
# ============================================================================
"""Feature engineering for Poisson-style goal models and calibration layers."""
from __future__ import annotations
import pandas as pd
import numpy as np

HOME_ADVANTAGE = 0.15  # goals boost; tune via backtest


def build_team_strengths(results: pd.DataFrame) -> pd.DataFrame:
    """Estimate simple attack/defense ratings per team using goal differentials.
    Returns a DataFrame: team, atk, def.
    """
    df = results.copy()
    # Long format: for each match, create two rows (home & away perspectives)
    home = df[["home_team", "away_team", "home_goals", "away_goals"]].copy()
    home.columns = ["team", "opp", "gf", "ga"]
    away = df[["away_team", "home_team", "away_goals", "home_goals"]].copy()
    away.columns = ["team", "opp", "gf", "ga"]
    long = pd.concat([home, away], ignore_index=True)

    grp = long.groupby("team", as_index=False).agg(gf=("gf", "mean"), ga=("ga", "mean"))
    # Attack ~ goals for vs league mean; Defense ~ goals against vs league mean
    g_mean = grp["gf"].mean()
    g_against_mean = grp["ga"].mean()
    grp["atk"] = grp["gf"] / g_mean
    grp["def"] = grp["ga"] / g_against_mean
    return grp[["team", "atk", "def"]]


def expected_goals(row, strengths, is_home=True):
    atk = strengths.loc[strengths.team == (row.home_team if is_home else row.away_team), "atk"].values
    dfn = strengths.loc[strengths.team == (row.away_team if is_home else row.home_team), "def"].values
    if len(atk) == 0 or len(dfn) == 0:
        return np.nan
    base = atk[0] / dfn[0]
    return max(0.05, base + (HOME_ADVANTAGE if is_home else 0))


def make_prediction_frame(matches: pd.DataFrame, strengths: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy()
    df["xg_home"] = df.apply(lambda r: expected_goals(r, strengths, True), axis=1)
    df["xg_away"] = df.apply(lambda r: expected_goals(r, strengths, False), axis=1)
    return df

# ============================================================================
# File: model_poisson.py
# ============================================================================
"""Poisson goals to 1X2 + Totals probabilities."""
import numpy as np
import pandas as pd
from math import exp
from itertools import product

MAX_GOALS = 10  # tail truncate


def poisson_pmf(k: int, lam: float) -> float:
    # simple Poisson pmf
    return (lam**k) * exp(-lam) / (np.math.factorial(k))


def score_matrix(lam_home: float, lam_away: float) -> np.ndarray:
    # P(HomeGoals=i, AwayGoals=j) under independent Poisson
    H = np.array([poisson_pmf(i, lam_home) for i in range(MAX_GOALS+1)])
    A = np.array([poisson_pmf(j, lam_away) for j in range(MAX_GOALS+1)])
    return np.outer(H, A)


def win_draw_probs(lam_home: float, lam_away: float) -> tuple[float, float, float]:
    M = score_matrix(lam_home, lam_away)
    ph = np.tril(M, k=-1).sum()  # home wins
    pd = np.trace(M)             # draw
    pa = np.triu(M, k=1).sum()   # away wins
    return ph, pd, pa


def over_under_prob(lam_home: float, lam_away: float, line: float = 2.5) -> tuple[float, float]:
    # returns P(Over line), P(Under line)
    M = score_matrix(lam_home, lam_away)
    probs = {}
    for i, j in product(range(MAX_GOALS+1), range(MAX_GOALS+1)):
        total = i + j
        probs[total] = probs.get(total, 0.0) + M[i, j]
    over_p = sum(v for k, v in probs.items() if k > line)
    under_p = 1 - over_p
    return over_p, under_p


def implied_prob_from_decimal(odds: float) -> float:
    return 1.0 / odds if odds and odds > 0 else np.nan


# ============================================================================
# File: ev_scanner.py
# ============================================================================
import pandas as pd
from model_poisson import win_draw_probs, over_under_prob, implied_prob_from_decimal


def compute_ev_row(row) -> dict:
    ph, pd, pa = win_draw_probs(row.xg_home, row.xg_away)
    evs = {}
    # 1X2 EVs (stake=1 unit)
    for label, p, odds in [
        ("home", ph, row.book_home_odds),
        ("draw", pd, row.book_draw_odds),
        ("away", pa, row.book_away_odds),
    ]:
        if odds and odds > 1.0:
            implied = implied_prob_from_decimal(odds)
            edge = p - implied
            ev = p * (odds - 1) - (1 - p)  # expected profit per 1 unit stake
            evs[f"ev_{label}"] = ev
            evs[f"edge_{label}"] = edge
        else:
            evs[f"ev_{label}"] = None
            evs[f"edge_{label}"] = None
    return evs | {"p_home": ph, "p_draw": pd, "p_away": pa}


def scan_ev(pred_df: pd.DataFrame, min_edge: float = 0.03) -> pd.DataFrame:
    out = pred_df.copy()
    ev_cols = out.apply(compute_ev_row, axis=1, result_type="expand")
    out = pd.concat([out, ev_cols], axis=1)
    # Rank by best single‑side EV
    out["best_ev"] = out[["ev_home", "ev_draw", "ev_away"]].max(axis=1)
    out["best_side"] = out[["ev_home", "ev_draw", "ev_away"]].idxmax(axis=1).str.replace("ev_", "")

    # filter by min edge on chosen side
    def chosen_edge(r):
        return r[f"edge_{r.best_side}"] if pd.notna(r.best_side) else None

    out["chosen_edge"] = out.apply(chosen_edge, axis=1)
    return out[out["chosen_edge"] >= min_edge].sort_values("best_ev", ascending=False)


# ============================================================================
# File: backtest.py
# ============================================================================
"""Rolling backtest to validate the model on past seasons.

Usage: backtest league‑by‑league (EPL, CHAMP) for season ranges, compare:
  • Hit rate calibration
  • CLV proxy (your probs vs. book implied)
  • EV realized return with a simple staking rule
"""
from __future__ import annotations
import os
import pandas as pd
from config import settings
from features import build_team_strengths, make_prediction_frame
from ev_scanner import scan_ev


def load_matches(league: str, season: int) -> pd.DataFrame:
    path = os.path.join(settings.proc_dir, f"matches_{league}_{season}.csv")
    return pd.read_csv(path)


def simple_backtest(league: str, seasons: list[int], min_edge: float = 0.03) -> pd.DataFrame:
    all_rows = []
    for season in seasons:
        df = load_matches(league, season)
        strengths = build_team_strengths(df)
        # use same season strengths for simplicity; improve with expanding window
        pred = make_prediction_frame(df, strengths)
        # join odds if you have them stored; here we assume odds cols exist
        picks = scan_ev(pred, min_edge=min_edge)
        # mark realized profit for the chosen side using actual result
        def realized_ev(r):
            side = r.best_side
            if pd.isna(side):
                return 0.0
            odds = r[f"book_{side}_odds"]
            if pd.isna(odds) or odds <= 1.0:
                return 0.0
            # did that side actually win?
            home, away = r.home_goals, r.away_goals
            won = ((side == "home" and home > away) or
                   (side == "draw" and home == away) or
                   (side == "away" and away > home))
            return (odds - 1.0) if won else -1.0

        picks["realized"] = picks.apply(realized_ev, axis=1)
        picks["season"] = season
        picks["league"] = league
        all_rows.append(picks)
    res = pd.concat(all_rows, ignore_index=True)
    summary = res.groupby(["league"]).agg(
        n_picks=("realized", "size"),
        avg_edge=("chosen_edge", "mean"),
        ev_mean=("best_ev", "mean"),
        roi=("realized", "mean"),
        total_units=("realized", "sum"),
    ).reset_index()
    out = os.path.join(settings.proc_dir, f"bt_summary_{league}.csv")
    summary.to_csv(out, index=False)
    return summary


# ============================================================================
# File: api.py (FastAPI)
# ============================================================================
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from features import build_team_strengths, make_prediction_frame
from ev_scanner import scan_ev

app = FastAPI(title="Euro Soccer Model API")

class PredictRequest(BaseModel):
    league: str
    season: int
    min_edge: float = 0.03

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(req: PredictRequest):
    # Simplified: load processed matches for league/season, compute EV list
    path = f"data/processed/matches_{req.league}_{req.season}.csv"
    df = pd.read_csv(path)
    strengths = build_team_strengths(df)
    pred = make_prediction_frame(df, strengths)
    ev = scan_ev(pred, min_edge=req.min_edge)
    # Return top rows as JSON
    cols = [
        "match_id","league","season","date","home_team","away_team",
        "book_home_odds","book_draw_odds","book_away_odds",
        "xg_home","xg_away","p_home","p_draw","p_away",
        "best_side","best_ev","chosen_edge"
    ]
    return ev[cols].head(50).to_dict(orient="records")


# ============================================================================
# File: app.py (Streamlit dashboard)
# ============================================================================
import streamlit as st
import pandas as pd
from features import build_team_strengths, make_prediction_frame
from ev_scanner import scan_ev

st.set_page_config(page_title="Euro Soccer Value – EPL + Championship", layout="wide")

st.title("⚽ Euro Soccer Value – EPL + Championship")
with st.sidebar:
    st.markdown("### Filters")
    league = st.selectbox("League", ["EPL", "CHAMP"])
    season = st.number_input("Season", min_value=2018, max_value=2030, value=2024, step=1)
    min_edge = st.slider("Min Edge (prob gap)", 0.0, 0.10, 0.03, 0.01)
    st.caption("Edge = model prob – market implied prob on chosen side")

path = f"data/processed/matches_{league}_{season}.csv"
try:
    df = pd.read_csv(path)
    strengths = build_team_strengths(df)
    pred = make_prediction_frame(df, strengths)
    ev = scan_ev(pred, min_edge=min_edge)
    st.success(f"Loaded {len(df)} matches; showing {len(ev)} EV opportunities ≥ {min_edge:.0%}")
    st.dataframe(ev[[
        "date","home_team","away_team",
        "book_home_odds","book_draw_odds","book_away_odds",
        "xg_home","xg_away","p_home","p_draw","p_away",
        "best_side","chosen_edge","best_ev"
    ]].sort_values("best_ev", ascending=False), use_container_width=True)
except FileNotFoundError:
    st.warning("No processed file found yet. Run ingest + process first.")

# ============================================================================
# File: README.md (quick start)
# ============================================================================
# Euro Soccer Value (EPL + Championship) – Quick Start

<!-- Badges (replace <OWNER>/<REPO> and domain) -->
[![Nightly Ingest](https://github.com/<OWNER>/<REPO>/actions/workflows/nightly_ingest.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/nightly_ingest.yml)
[![Publish Public Feeds](https://github.com/<OWNER>/<REPO>/actions/workflows/publish_pages.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/publish_pages.yml)
[![Daily Email](https://github.com/<OWNER>/<REPO>/actions/workflows/daily_email.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/daily_email.yml)
[![Docker Build](https://github.com/<OWNER>/<REPO>/actions/workflows/deploy_streamlit.yml/badge.svg)](https://github.com/<OWNER>/<REPO>/actions/workflows/deploy_streamlit.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](#)
[**Live Feeds**](https://footyev.com) · [meta.json](https://footyev.com/meta.json)

> Replace `<OWNER>/<REPO>` with your GitHub org/user and repo name, and update the domain link to your Pages domain.


## 0) Setup
```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # add your API keys
```

## 1) Ingest historical match results (per league & season)
Edit `ingest_api.py` if needed, then in a scratch script / notebook:
```python
from ingest_api import fetch_results, fetch_odds_for_matches
from config import settings
import pandas as pd

# Example: pull 2023 & 2024 seasons for EPL + Championship
for lg in ["EPL", "CHAMP"]:
    for yr in [2023, 2024]:
        df = fetch_results(lg, yr)
        # Optionally add odds
        odds = fetch_odds_for_matches(df["match_id"].dropna().astype(int).tolist()[:50])  # sample for dev
        if not odds.empty:
            merged = df.merge(odds, on="match_id", how="left")
            merged.to_csv(f"{settings.proc_dir}/matches_{lg}_{yr}.csv", index=False)
```

> Tip: For backtests, use closing odds where possible; otherwise latest pre‑match.

## 2) Backtest
```python
from backtest import simple_backtest
print(simple_backtest("CHAMP", [2023, 2024], min_edge=0.03))
```

## 3) Run API
```bash
uvicorn api:app --reload --port 8001
```

## 4) Run Dashboard
```bash
streamlit run app.py
```

## 5) Productizing
- Add daily cron to ingest upcoming fixtures + odds, run model, publish JSON/CSV to a public endpoint.
- Add Stripe to gate premium features (alerts, more leagues, deeper props).
- Add affiliate links next to each book line (check jurisdiction rules).

## 6) Validation & Credibility
- Track and publish: pick log, ROI, hit-rate by league, CLV vs closing lines.
- Show calibration plots (predicted vs actual frequency).
- Keep your write‑ups modest and transparent (no “locks”).

## 7) Extending the Model
- Replace naive strengths with regularized attack/defense ratings (GLM).
- Add lineup‑strength adjustments using player xG+xA on/off.
- Add referee, weather, congestion, travel, and rest features.
- Add totals model calibration and Asian handicap mapping.

# ──────────────────────────────────────────────────────────────────────────────
# End of scaffold
# ──────────────────────────────────────────────────────────────────────────────


# ============================================================================
# File: scripts/test_ingest.py
# ============================================================================
"""Tiny script to exercise end-to-end ingest for EPL + Championship, one season.
Usage:
  python scripts/test_ingest.py --season 2024 --leagues EPL CHAMP --with-odds
"""
import argparse
import os
import pandas as pd
from ingest_api import fetch_results, fetch_odds_for_matches
from config import settings

parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, default=2024)
parser.add_argument("--leagues", nargs="+", default=["EPL", "CHAMP"]) 
parser.add_argument("--with-odds", action="store_true")
args = parser.parse_args()

for lg in args.leagues:
    print(f"[INGEST] {lg} {args.season}")
    df = fetch_results(lg, args.season)
    if args.with_odds:
        mids = df["match_id"].dropna().astype(int).tolist()[:150]
        odds = fetch_odds_for_matches(mids)
        if not odds.empty:
            df = df.merge(odds, on="match_id", how="left")
    out = os.path.join(settings.proc_dir, f"matches_{lg}_{args.season}.csv")
    df.to_csv(out, index=False)
    print(f"[OK] wrote {out} ({len(df)} rows)")


# ============================================================================
# File: models/glm_ratings.py
# ============================================================================
"""Regularized attack/defense ratings using GLM (Poisson regression).
This improves on naive strengths by fitting team fixed effects for goals.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Note: add to requirements.txt -> statsmodels>=0.14


def fit_glm_goals(df_matches: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Fit Poisson GLM for goals with home advantage and team effects.
    Returns: ratings DF (team, atk, def) and home_adv (float goals).
    """
    # Build long format: one row per team per match with indicator for home
    home = df_matches[["home_team","away_team","home_goals"]].copy()
    home.columns = ["team","opp","goals"]
    home["is_home"] = 1

    away = df_matches[["away_team","home_team","away_goals"]].copy()
    away.columns = ["team","opp","goals"]
    away["is_home"] = 0

    long = pd.concat([home, away], ignore_index=True)

    # Design matrix: team fixed effects for attack, opponent effects for defense
    X_attack = pd.get_dummies(long["team"], prefix="atk")
    X_defense = pd.get_dummies(long["opp"], prefix="def")
    X = pd.concat([X_attack, X_defense, long[["is_home"]]], axis=1)
    y = long["goals"].astype(float)

    # Add small ridge-like penalty via L2 using sm.GLM.fit_regularized
    model = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson())
    res = model.fit_regularized(alpha=0.5, L1_wt=0.0, maxiter=200)

    params = res.params
    home_adv = params.get("is_home", 0.15)

    # Convert params back to per-team attack/defense ratios in goal-space
    atk_cols = [c for c in params.index if c.startswith("atk_")]
    def_cols = [c for c in params.index if c.startswith("def_")]

    atk_df = pd.DataFrame({
        "team": [c.replace("atk_", "") for c in atk_cols],
        "atk": np.exp(params[atk_cols].values),
    })
    def_df = pd.DataFrame({
        "team": [c.replace("def_", "") for c in def_cols],
        "def": np.exp(params[def_cols].values),
    })
    ratings = atk_df.merge(def_df, on="team", how="outer").fillna(1.0)
    return ratings, float(np.exp(home_adv) - 1.0)  # convert to goal-equivalent boost


def predict_xg_glm(row, ratings: pd.DataFrame, home_bonus: float) -> tuple[float,float]:
    h = ratings.loc[ratings.team == row.home_team]
    a = ratings.loc[ratings.team == row.away_team]
    if h.empty or a.empty:
        return np.nan, np.nan
    lam_h = h.at[h.index[0], "atk"] / a.at[a.index[0], "def"] + home_bonus
    lam_a = a.at[a.index[0], "atk"] / h.at[h.index[0], "def"]
    return max(lam_h, 0.05), max(lam_a, 0.05)


def make_prediction_frame_glm(matches: pd.DataFrame) -> pd.DataFrame:
    ratings, home_bonus = fit_glm_goals(matches)
    df = matches.copy()
    xh, xa = [], []
    for _, r in df.iterrows():
        lam_h, lam_a = predict_xg_glm(r, ratings, home_bonus)
        xh.append(lam_h)
        xa.append(lam_a)
    df["xg_home"], df["xg_away"] = xh, xa
    return df


# ============================================================================
# File: app_glm.py (Streamlit dashboard using GLM ratings)
# ============================================================================
import streamlit as st
import pandas as pd
from models.glm_ratings import make_prediction_frame_glm
from ev_scanner import scan_ev

st.set_page_config(page_title="Euro Soccer Value – GLM (EPL + Championship)", layout="wide")

st.title("⚽ Euro Soccer Value – GLM (EPL + Championship)")
with st.sidebar:
    league = st.selectbox("League", ["EPL", "CHAMP"])
    season = st.number_input("Season", min_value=2016, max_value=2030, value=2024, step=1)
    min_edge = st.slider("Min Edge (prob gap)", 0.0, 0.10, 0.03, 0.01)

path = f"data/processed/matches_{league}_{season}.csv"
try:
    df = pd.read_csv(path)
    pred = make_prediction_frame_glm(df)
    ev = scan_ev(pred, min_edge=min_edge)
    st.success(f"Loaded {len(df)} matches; GLM EV ≥ {min_edge:.0%}: {len(ev)}")
    st.dataframe(ev[[
        "date","home_team","away_team",
        "book_home_odds","book_draw_odds","book_away_odds",
        "xg_home","xg_away","p_home","p_draw","p_away",
        "best_side","chosen_edge","best_ev"
    ]].sort_values("best_ev", ascending=False), use_container_width=True)
except FileNotFoundError:
    st.warning("No processed file found yet. Run scripts/test_ingest.py first.")


# ============================================================================
# File: Dockerfile
# ============================================================================
# Multi-stage build: API + Streamlit images
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Default: run Streamlit dashboard (app_glm.py). Override CMD for API service.
EXPOSE 8501 8001
CMD ["streamlit", "run", "app_glm.py", "--server.port", "8501", "--server.address", "0.0.0.0"]


# ============================================================================
# File: docker-compose.yml
# ============================================================================
version: "3.9"
services:
  dashboard:
    build: .
    ports:
      - "8501:8501"
    environment:
      - API_FOOTBALL_KEY=${API_FOOTBALL_KEY}
      - ENV=prod
    volumes:
      - ./:/app
  api:
    build: .
    command: ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]
    ports:
      - "8001:8001"
    environment:
      - API_FOOTBALL_KEY=${API_FOOTBALL_KEY}
      - ENV=prod
    volumes:
      - ./:/app


# ============================================================================
# File: DEPLOY.md
# ============================================================================
# Deploy guide (Docker, Render/Vercel, Cron)

## Local (Docker Compose)
```bash
cp .env.example .env   # add API_FOOTBALL_KEY
docker compose up --build
# dashboard -> http://localhost:8501
# api       -> http://localhost:8001/health
```

## Render.com (one-click style)
- Create a new **Web Service** from your Git repo
- Docker build: auto-detects Dockerfile
- Environment variable: `API_FOOTBALL_KEY`
- Start command (dashboard): `streamlit run app_glm.py --server.port $PORT --server.address 0.0.0.0`
- (Optional) Second service for API: `uvicorn api:app --host 0.0.0.0 --port $PORT`

## Vercel (dashboard only)
- Wrap Streamlit via `vercel.json` or use Streamlit Cloud for simplest path
- Or export a static marketing page that calls your API hosted elsewhere (Render/Fly.io)

## Cron / Scheduler
- Use GitHub Actions or Render Cron to run nightly:
  - ingest latest fixtures/odds
  - recompute predictions → upload JSON/CSV artifact

## Observability
- Add basic logging to `ingest_api.py`
- Track daily pick logs and CLV stats; publish a public results page


# ============================================================================
# File: vercel.json (optional if deploying a React front-end later)
# ============================================================================
{
  "version": 2,
  "builds": [
    { "src": "api.py", "use": "@vercel/python" }
  ],
  "routes": [
    { "src": "/api/(.*)", "dest": "/api.py" }
  ]
}


# ============================================================================
# File: .github/workflows/nightly_ingest.yml
# ============================================================================
name: Nightly Ingest & Backtest

on:
  schedule:
    # Run at 02:15 UTC daily
    - cron: "15 2 * * *"
  workflow_dispatch:
    inputs:
      season:
        description: "Season year (e.g., 2024)"
        required: false
        default: "2024"
      leagues:
        description: "Space-separated league codes (EPL CHAMP)"
        required: false
        default: "EPL CHAMP"
      with_odds:
        description: "Fetch odds as well"
        required: false
        default: "true"

jobs:
  run:
    runs-on: ubuntu-latest
    env:
      PYTHONUNBUFFERED: 1
      SEASON: ${{ inputs.season || '2024' }}
      LEAGUES: ${{ inputs.leagues || 'EPL CHAMP' }}
      WITH_ODDS: ${{ inputs.with_odds || 'true' }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

      - name: Configure API keys
        env:
          API_FOOTBALL_KEY: ${{ secrets.API_FOOTBALL_KEY }}
        run: |
          echo "API_FOOTBALL_KEY=${API_FOOTBALL_KEY}" >> $GITHUB_ENV

      - name: Ingest fixtures/results (+odds)
        run: |
          source .venv/bin/activate
          if [ "$WITH_ODDS" = "true" ]; then
            python scripts/test_ingest.py --season "$SEASON" --leagues $LEAGUES --with-odds
          else
            python scripts/test_ingest.py --season "$SEASON" --leagues $LEAGUES
          fi

      - name: Backtest (Championship & EPL)
        run: |
          source .venv/bin/activate
          python - << 'PY'
from backtest import simple_backtest
print(simple_backtest('CHAMP', [int('${{ env.SEASON }}')], min_edge=0.03))
print(simple_backtest('EPL', [int('${{ env.SEASON }}')], min_edge=0.03))
PY

      - name: Upload processed data & summaries
        uses: actions/upload-artifact@v4
        with:
          name: processed-${{ env.SEASON }}
          path: |
            data/processed/*.csv

      - name: Trigger Render deploy (optional)
        if: ${{ secrets.RENDER_DEPLOY_HOOK_URL != '' }}
        run: |
          curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"

# ============================================================================
# File: .github/workflows/deploy_streamlit.yml
# ============================================================================
name: Build & Publish Docker (Streamlit)

on:
  workflow_dispatch:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up QEMU
        uses: docker/setup-qemu-action@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build & Push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/euro-soccer-value:latest
          build-args: |
            API_FOOTBALL_KEY=${{ secrets.API_FOOTBALL_KEY }}

# Notes:
# 1) Add repository secrets: API_FOOTBALL_KEY, optionally RENDER_DEPLOY_HOOK_URL
# 2) Nightly workflow uploads CSV artifacts; you can fetch them or host them via a CDN.
# 3) For automatic commits of generated CSVs, add a PAT and a commit step (optional).


# ============================================================================
# File: scripts/generate_public_feed.py
# ============================================================================
"""Generate public JSON/CSV feeds of today's EV picks for EPL + Championship.
Writes outputs under `public/` for GitHub Pages publishing.
Also writes a small `meta.json` with last-updated timestamp and counts.
"""
from __future__ import annotations
import os, json, datetime as dt
import pandas as pd
from config import settings
from features import build_team_strengths, make_prediction_frame
from models.glm_ratings import make_prediction_frame_glm
from ev_scanner import scan_ev

LEAGUES = ["EPL", "CHAMP"]
SEASON_DEFAULT = int(os.getenv("FEED_SEASON", dt.date.today().year))
MIN_EDGE = float(os.getenv("FEED_MIN_EDGE", 0.03))
USE_GLM = os.getenv("FEED_USE_GLM", "1") == "1"

os.makedirs("public", exist_ok=True)

def load_matches(league: str, season: int) -> pd.DataFrame:
    path = os.path.join(settings.proc_dir, f"matches_{league}_{season}.csv")
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. Run ingest first.")
    return pd.read_csv(path)

all_ev = []
for lg in LEAGUES:
    df = load_matches(lg, SEASON_DEFAULT)
    pred = make_prediction_frame_glm(df) if USE_GLM else make_prediction_frame(df, build_team_strengths(df))
    ev = scan_ev(pred, min_edge=MIN_EDGE)
    ev["league"] = lg
    all_ev.append(ev)

res = pd.concat(all_ev, ignore_index=True)
# keep relevant columns
cols = [
    "date","league","home_team","away_team",
    "book_home_odds","book_draw_odds","book_away_odds",
    "xg_home","xg_away","p_home","p_draw","p_away",
    "best_side","chosen_edge","best_ev"
]
res = res[cols].sort_values(["date","best_ev"], ascending=[True, False])

# Write CSV & JSON
csv_path = "public/ev_picks.csv"
json_path = "public/ev_picks.json"
res.to_csv(csv_path, index=False)
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(res.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

# Write META (last-updated UTC and counts)
meta = {
    "last_updated_utc": dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    "season": SEASON_DEFAULT,
    "leagues": LEAGUES,
    "min_edge": MIN_EDGE,
    "model": "glm" if USE_GLM else "baseline",
    "items": int(len(res)),
}
with open("public/meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"Wrote {csv_path}, {json_path} and public/meta.json with {meta['items']} rows")


# ============================================================================
# File: scripts/email_top_ev.py
# ============================================================================
"""Email the top EV matches (EPL + Championship) using SMTP or Resend API.

Set one of:
  • SMTP_* secrets (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO, EMAIL_FROM)
  • RESEND_API_KEY and EMAIL_TO, EMAIL_FROM

This script picks top N rows from the public feed, formats a simple email, and sends.
"""
from __future__ import annotations
import os, smtplib, ssl, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

TOP_N = int(os.getenv("EMAIL_TOP_N", 10))
FEED_JSON = os.getenv("FEED_JSON", "public/ev_picks.json")

with open(FEED_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
rows = data[:TOP_N]

lines = [
    "Top EV Matches (EPL + Championship)
",
]
for r in rows:
    line = (
        f"{r['date']} — {r['league']}: {r['home_team']} vs {r['away_team']}
"
        f"  Model probs H/D/A: {r['p_home']:.2%}/{r['p_draw']:.2%}/{r['p_away']:.2%}
"
        f"  Best side: {r['best_side']}  | Edge: {r['chosen_edge']:.2%}  | EV/u: {r['best_ev']:.2f}
"
        f"  Book odds H/D/A: {r['book_home_odds']} / {r['book_draw_odds']} / {r['book_away_odds']}
"
    )
    lines.append(line)
body_text = "
".join(lines)

subject = os.getenv("EMAIL_SUBJECT", "Daily EV Picks – EPL + Championship")
email_to = os.getenv("EMAIL_TO")
email_from = os.getenv("EMAIL_FROM", email_to or "noreply@example.com")

# Try Resend API first if key present
resend_key = os.getenv("RESEND_API_KEY")
if resend_key and email_to:
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
        json={
            "from": email_from,
            "to": [email_to],
            "subject": subject,
            "text": body_text,
        },
        timeout=20,
    )
    resp.raise_for_status()
    print("Sent via Resend.")
else:
    # Fallback: SMTP
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pw = os.getenv("SMTP_PASS")
    if not (host and user and pw and email_to):
        raise SystemExit("Missing SMTP_* or RESEND_API_KEY/EMAIL_TO secrets.")
    msg = MIMEMultipart()
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain"))
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(user, pw)
        server.sendmail(email_from, [email_to], msg.as_string())
    print("Sent via SMTP.")

# ============================================================================
"""Email the top EV matches (EPL + Championship) using SMTP or Resend API.

Set one of:
  • SMTP_* secrets (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, EMAIL_TO, EMAIL_FROM)
  • RESEND_API_KEY and EMAIL_TO, EMAIL_FROM

This script picks top N rows from the public feed, formats a simple email, and sends.
"""
from __future__ import annotations
import os, smtplib, ssl, json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

TOP_N = int(os.getenv("EMAIL_TOP_N", 10))
FEED_JSON = os.getenv("FEED_JSON", "public/ev_picks.json")

with open(FEED_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)
rows = data[:TOP_N]

lines = [
    "Top EV Matches (EPL + Championship)
",
]
for r in rows:
    line = (
        f"{r['date']} — {r['league']}: {r['home_team']} vs {r['away_team']}
"
        f"  Model probs H/D/A: {r['p_home']:.2%}/{r['p_draw']:.2%}/{r['p_away']:.2%}
"
        f"  Best side: {r['best_side']}  | Edge: {r['chosen_edge']:.2%}  | EV/u: {r['best_ev']:.2f}
"
        f"  Book odds H/D/A: {r['book_home_odds']} / {r['book_draw_odds']} / {r['book_away_odds']}
"
    )
    lines.append(line)
body_text = "
".join(lines)

subject = os.getenv("EMAIL_SUBJECT", "Daily EV Picks – EPL + Championship")
email_to = os.getenv("EMAIL_TO")
email_from = os.getenv("EMAIL_FROM", email_to or "noreply@example.com")

# Try Resend API first if key present
resend_key = os.getenv("RESEND_API_KEY")
if resend_key and email_to:
    resp = requests.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
        json={
            "from": email_from,
            "to": [email_to],
            "subject": subject,
            "text": body_text,
        },
        timeout=20,
    )
    resp.raise_for_status()
    print("Sent via Resend.")
else:
    # Fallback: SMTP
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pw = os.getenv("SMTP_PASS")
    if not (host and user and pw and email_to):
        raise SystemExit("Missing SMTP_* or RESEND_API_KEY/EMAIL_TO secrets.")
    msg = MIMEMultipart()
    msg["From"] = email_from
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain"))
    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(user, pw)
        server.sendmail(email_from, [email_to], msg.as_string())
    print("Sent via SMTP.")


# ============================================================================
# File: .github/workflows/publish_pages.yml
# ============================================================================
name: Publish Public Feeds (GitHub Pages)

on:
  schedule:
    - cron: "45 2 * * *"   # after nightly ingest
  workflow_dispatch: {}

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    env:
      PUBLIC_BASE_URL: ${{ vars.PUBLIC_BASE_URL || 'https://footyev.com' }}
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

      - name: Generate public feeds (JSON/CSV)
        env:
          FEED_SEASON: 2024
          FEED_MIN_EDGE: 0.03
        run: |
          source .venv/bin/activate
          python scripts/generate_public_feed.py

      - name: Generate sitemap.xml & robots.txt
        env:
          PUBLIC_BASE_URL: ${{ env.PUBLIC_BASE_URL }}
        run: |
          source .venv/bin/activate
          python scripts/generate_sitemap.py

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: public

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4

# After first successful run, enable Pages in repo Settings → Pages → GitHub Actions.

# ============================================================================
# File: .github/workflows/daily_email.yml
# ============================================================================
name: Send Daily EV Email

on:
  schedule:
    - cron: "0 3 * * *"   # after pages publish
  workflow_dispatch: {}

jobs:
  send:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

      - name: Download latest public feed (optional)
        run: |
          # If publishing to Pages, you can curl your public JSON here instead of local file.
          # curl -L "${{ vars.PUBLIC_BASE_URL || 'https://footyev.com' }}/ev_picks.json" -o public/ev_picks.json
          mkdir -p public
          if [ ! -f public/ev_picks.json ]; then echo "[]" > public/ev_picks.json; fi

      - name: Send Email
        env:
          RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}
          EMAIL_TO: ${{ secrets.EMAIL_TO }}
          EMAIL_FROM: ${{ secrets.EMAIL_FROM }}
          SMTP_HOST: ${{ secrets.SMTP_HOST }}
          SMTP_PORT: ${{ secrets.SMTP_PORT }}
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASS: ${{ secrets.SMTP_PASS }}
        run: |
          source .venv/bin/activate
          python scripts/email_top_ev.py

# ============================================================================
name: Send Daily EV Email

on:
  schedule:
    - cron: "0 3 * * *"   # after pages publish
  workflow_dispatch: {}

jobs:
  send:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install deps
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

      - name: Download latest public feed (optional)
        run: |
          # If publishing to Pages, you can curl your public JSON here instead of local file.
          # curl -L "https://<user>.github.io/<repo>/ev_picks.json" -o public/ev_picks.json
          mkdir -p public
          if [ ! -f public/ev_picks.json ]; then echo "[]" > public/ev_picks.json; fi

      - name: Send Email
        env:
          RESEND_API_KEY: ${{ secrets.RESEND_API_KEY }}
          EMAIL_TO: ${{ secrets.EMAIL_TO }}
          EMAIL_FROM: ${{ secrets.EMAIL_FROM }}
          SMTP_HOST: ${{ secrets.SMTP_HOST }}
          SMTP_PORT: ${{ secrets.SMTP_PORT }}
          SMTP_USER: ${{ secrets.SMTP_USER }}
          SMTP_PASS: ${{ secrets.SMTP_PASS }}
        run: |
          source .venv/bin/activate
          python scripts/email_top_ev.py

# Notes:
# - For Substack: set up a Zapier/Make automation that watches a specific inbox or webhook and auto-creates a post.
#   Point EMAIL_TO to that automation address to turn the daily email into a Substack draft/post.
# - Alternatively, send to your own inbox and manually post until you’re ready to automate.


# ============================================================================
# File: public/index.html
# ============================================================================
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Euro Soccer Value — EPL + Championship</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <header class="container head">
    <div class="brand">
      <img src="logo.svg" alt="Euro Soccer Value" width="42" height="42" />
      <div>
        <h1>Euro Soccer Value</h1>
        <p class="sub">Daily EV picks for <strong>EPL</strong> & <strong>EFL Championship</strong></p>
      </div>
    </div>
    <nav class="links">
      <a href="ev_picks.json">JSON feed</a>
      <a href="ev_picks.csv">CSV feed</a>
      <a href="privacy.html">Privacy</a>
      <a href="terms.html">Terms</a>
      <span id="badge" class="badge">Updated: —</span>
    </nav>
  </header>

  <main class="container">
    <div id="status">Loading today’s matches…</div>
    <table id="tbl" class="hidden">
      <thead>
        <tr>
          <th>Date</th><th>League</th><th>Home</th><th>Away</th>
          <th>H/D/A (model)</th>
          <th>Best Side</th>
          <th>Edge</th>
          <th>EV / unit</th>
          <th>Book H/D/A</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </main>

  <footer class="container foot">
    <small>© <span id="yr"></span> Euro Soccer Value. Data for informational purposes only. No guarantees. Gamble responsibly.</small>
  </footer>

  <script>
    const feedUrl = (new URLSearchParams(location.search).get('feed')) || 'ev_picks.json';
    const status = document.getElementById('status');
    const tbl = document.getElementById('tbl');
    const tbody = tbl.querySelector('tbody');
    const badge = document.getElementById('badge');
    document.getElementById('yr').textContent = new Date().getFullYear();

    // fetch meta for status badge
    fetch('meta.json').then(r=>r.json()).then(meta => {
      if (meta && meta.last_updated_utc) {
        badge.textContent = `Updated: ${meta.last_updated_utc} UTC · ${meta.items} picks`;
      }
    }).catch(()=>{});

    // fetch feed table
    fetch(feedUrl)
      .then(r => r.json())
      .then(rows => {
        if (!rows || !rows.length) {
          status.textContent = 'No EV picks found. Check back later today.';
          return;
        }
        status.classList.add('hidden');
        tbl.classList.remove('hidden');
        for (const r of rows) {
          const tr = document.createElement('tr');
          const probs = `${(r.p_home*100).toFixed(0)}% / ${(r.p_draw*100).toFixed(0)}% / ${(r.p_away*100).toFixed(0)}%`;
          const odds = `${r.book_home_odds ?? '-'} / ${r.book_draw_odds ?? '-'} / ${r.book_away_odds ?? '-'}`;
          tr.innerHTML = `
            <td>${r.date}</td>
            <td>${r.league}</td>
            <td>${r.home_team}</td>
            <td>${r.away_team}</td>
            <td>${probs}</td>
            <td class="tag">${r.best_side?.toUpperCase() || '-'}</td>
            <td>${(r.chosen_edge*100).toFixed(1)}%</td>
            <td>${r.best_ev?.toFixed(2)}</td>
            <td>${odds}</td>
          `;
          tbody.appendChild(tr);
        }
      })
      .catch(e => {
        status.textContent = 'Failed to load feed.';
        console.error(e);
      });
  </script>
</body>
</html>

# ============================================================================

<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Euro Soccer Value — EPL + Championship</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <header class="container">
    <h1>⚽ Euro Soccer Value</h1>
    <p class="sub">Daily EV picks for <strong>EPL</strong> & <strong>EFL Championship</strong></p>
    <nav class="links">
      <a href="ev_picks.json">JSON feed</a>
      <a href="ev_picks.csv">CSV feed</a>
    </nav>
  </header>

  <main class="container">
    <div id="status">Loading today’s matches…</div>
    <table id="tbl" class="hidden">
      <thead>
        <tr>
          <th>Date</th><th>League</th><th>Home</th><th>Away</th>
          <th>H/D/A (model)</th>
          <th>Best Side</th>
          <th>Edge</th>
          <th>EV / unit</th>
          <th>Book H/D/A</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </main>

  <footer class="container foot">
    <small>© <span id="yr"></span> Euro Soccer Value. Data for informational purposes only. No guarantees. Gamble responsibly.</small>
  </footer>

  <script>
    const feedUrl = (new URLSearchParams(location.search).get('feed')) || 'ev_picks.json';
    const status = document.getElementById('status');
    const tbl = document.getElementById('tbl');
    const tbody = tbl.querySelector('tbody');
    document.getElementById('yr').textContent = new Date().getFullYear();

    fetch(feedUrl)
      .then(r => r.json())
      .then(rows => {
        if (!rows || !rows.length) {
          status.textContent = 'No EV picks found. Check back later today.';
          return;
        }
        status.classList.add('hidden');
        tbl.classList.remove('hidden');
        for (const r of rows) {
          const tr = document.createElement('tr');
          const probs = `${(r.p_home*100).toFixed(0)}% / ${(r.p_draw*100).toFixed(0)}% / ${(r.p_away*100).toFixed(0)}%`;
          const odds = `${r.book_home_odds ?? '-'} / ${r.book_draw_odds ?? '-'} / ${r.book_away_odds ?? '-'}`;
          tr.innerHTML = `
            <td>${r.date}</td>
            <td>${r.league}</td>
            <td>${r.home_team}</td>
            <td>${r.away_team}</td>
            <td>${probs}</td>
            <td class="tag">${r.best_side?.toUpperCase() || '-'}</td>
            <td>${(r.chosen_edge*100).toFixed(1)}%</td>
            <td>${r.best_ev?.toFixed(2)}</td>
            <td>${odds}</td>
          `;
          tbody.appendChild(tr);
        }
      })
      .catch(e => {
        status.textContent = 'Failed to load feed.';
        console.error(e);
      });
  </script>
</body>
</html>


# ============================================================================
# File: public/styles.css
# ============================================================================
:root{ --bg:#0b0d10; --fg:#e5ecf3; --muted:#a5b3c2; --card:#12161b; --accent:#25a18e; }
*{ box-sizing:border-box; }
body{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.5 system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial; }
.container{ max-width:1100px; margin:0 auto; padding:24px; }
.head{ display:flex; align-items:center; justify-content:space-between; gap:16px; }
.brand{ display:flex; align-items:center; gap:12px; }
h1{ margin:0; font-size:28px; }
.sub{ color:var(--muted); margin:6px 0 12px; }
.links a{ color:var(--accent); margin-right:16px; text-decoration:none; }
#status{ color:var(--muted); }
.hidden{ display:none; }
.badge{ display:inline-block; background:linear-gradient(90deg, #25a18e, #38c6b0); color:#07211c; padding:6px 10px; border-radius:999px; font-weight:700; font-size:12px; }

table{ width:100%; border-collapse:separate; border-spacing:0 8px; }
th, td{ text-align:left; padding:12px 10px; }
thead th{ color:var(--muted); font-weight:600; border-bottom:1px solid #26313d; }
tbody tr{ background:var(--card); border-radius:12px; }
tbody tr td:first-child{ border-top-left-radius:12px; border-bottom-left-radius:12px; }
tbody tr td:last-child{ border-top-right-radius:12px; border-bottom-right-radius:12px; }
.tag{ font-weight:700; color:#e9fbf7; }
.foot{ color:var(--muted); }


# ============================================================================
# File: web/package.json  (React landing page)
# ============================================================================
{
  "name": "euro-soccer-value",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1"
  },
  "devDependencies": {
    "vite": "^5.4.0"
  }
}

# ============================================================================
# File: web/index.html
# ============================================================================
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Euro Soccer Value — Dashboard</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>

# ============================================================================
# File: web/src/main.jsx
# ============================================================================
import React from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)

# ============================================================================
# File: web/src/App.jsx
# ============================================================================
import React, { useEffect, useState } from 'react'

const FEED_URL = import.meta.env.VITE_FEED_URL || '/ev_picks.json'
const META_URL = import.meta.env.VITE_META_URL || (FEED_URL.includes('ev_picks.json')
  ? FEED_URL.replace('ev_picks.json', 'meta.json')
  : '/meta.json')

export default function App(){
  const [rows, setRows] = useState(null)
  const [err, setErr] = useState(null)
  const [meta, setMeta] = useState(null)

  useEffect(() => {
    fetch(FEED_URL)
      .then(r => r.json())
      .then(setRows)
      .catch(e => setErr(e.message))

    fetch(META_URL)
      .then(r => r.json())
      .then(setMeta)
      .catch(() => {})
  }, [])

  const badge = meta ? `Updated: ${meta.last_updated_utc} UTC · ${meta.items} picks` : 'Updated: —'

  return (
    <div style={{maxWidth:1100, margin:'0 auto', padding:24, fontFamily:'system-ui'}}>
      <header style={{display:'flex', alignItems:'center', justifyContent:'space-between', gap:16}}>
        <div style={{display:'flex', alignItems:'center', gap:12}}>
          <img src={import.meta.env.BASE_URL + 'logo.svg'} width={38} height={38} alt="logo"/>
          <div>
            <h1 style={{margin:0}}>⚽ Euro Soccer Value</h1>
            <p style={{color:'#667788', margin:'6px 0 0'}}>EPL + Championship — model vs. market EV</p>
          </div>
        </div>
        <nav>
          <a href="/ev_picks.json" style={{marginRight:12}}>JSON</a>
          <a href="/ev_picks.csv" style={{marginRight:12}}>CSV</a>
          <span style={{display:'inline-block', background:'linear-gradient(90deg,#25a18e,#38c6b0)', color:'#07211c', padding:'6px 10px', borderRadius:999, fontWeight:700, fontSize:12}}>{badge}</span>
        </nav>
      </header>

      {!rows && !err && <p>Loading…</p>}
      {err && <p style={{color:'crimson'}}>Failed to load feed: {String(err)}</p>}
      {rows && rows.length === 0 && <p>No EV picks at the moment.</p>}
      {rows && rows.length > 0 && (
        <table width="100%" cellPadding="8" style={{borderCollapse:'separate', borderSpacing:'0 8px'}}>
          <thead>
            <tr>
              <th align="left">Date</th>
              <th align="left">League</th>
              <th align="left">Home</th>
              <th align="left">Away</th>
              <th align="left">H/D/A (model)</th>
              <th align="left">Best Side</th>
              <th align="left">Edge</th>
              <th align="left">EV/u</th>
              <th align="left">Book H/D/A</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={{background:'#f5f7fa'}}>
                <td>{r.date}</td>
                <td>{r.league}</td>
                <td>{r.home_team}</td>
                <td>{r.away_team}</td>
                <td>{`${(r.p_home*100).toFixed(0)}% / ${(r.p_draw*100).toFixed(0)}% / ${(r.p_away*100).toFixed(0)}%`}</td>
                <td style={{fontWeight:700}}>{r.best_side?.toUpperCase()}</td>
                <td>{(r.chosen_edge*100).toFixed(1)}%</td>
                <td>{r.best_ev?.toFixed(2)}</td>
                <td>{`${r.book_home_odds ?? '-'} / ${r.book_draw_odds ?? '-'} / ${r.book_away_odds ?? '-'}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

# ============================================================================
import React, { useEffect, useState } from 'react'

const FEED_URL = import.meta.env.VITE_FEED_URL || '/ev_picks.json'

export default function App(){
  const [rows, setRows] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    fetch(FEED_URL)
      .then(r => r.json())
      .then(setRows)
      .catch(e => setErr(e.message))
  }, [])

  return (
    <div style={{maxWidth:1100, margin:'0 auto', padding:24, fontFamily:'system-ui'}}>
      <h1>⚽ Euro Soccer Value</h1>
      <p style={{color:'#667788'}}>EPL + Championship — model vs. market EV</p>
      <div style={{marginBottom:12}}>
        <a href="/ev_picks.json">JSON</a>{' '}·{' '}
        <a href="/ev_picks.csv">CSV</a>
      </div>
      {!rows && !err && <p>Loading…</p>}
      {err && <p style={{color:'crimson'}}>Failed to load feed: {String(err)}</p>}
      {rows && rows.length === 0 && <p>No EV picks at the moment.</p>}
      {rows && rows.length > 0 && (
        <table width="100%" cellPadding="8" style={{borderCollapse:'separate', borderSpacing:'0 8px'}}>
          <thead>
            <tr>
              <th align="left">Date</th>
              <th align="left">League</th>
              <th align="left">Home</th>
              <th align="left">Away</th>
              <th align="left">H/D/A (model)</th>
              <th align="left">Best Side</th>
              <th align="left">Edge</th>
              <th align="left">EV/u</th>
              <th align="left">Book H/D/A</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} style={{background:'#f5f7fa'}}>
                <td>{r.date}</td>
                <td>{r.league}</td>
                <td>{r.home_team}</td>
                <td>{r.away_team}</td>
                <td>{`${(r.p_home*100).toFixed(0)}% / ${(r.p_draw*100).toFixed(0)}% / ${(r.p_away*100).toFixed(0)}%`}</td>
                <td style={{fontWeight:700}}>{r.best_side?.toUpperCase()}</td>
                <td>{(r.chosen_edge*100).toFixed(1)}%</td>
                <td>{r.best_ev?.toFixed(2)}</td>
                <td>{`${r.book_home_odds ?? '-'} / ${r.book_draw_odds ?? '-'} / ${r.book_away_odds ?? '-'}`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

# ============================================================================
# README additions — Front-end hosting
# ============================================================================
# GitHub Pages marketing page
- The `public/` folder is published by the `publish_pages.yml` workflow.
- `public/index.html` lists today’s EV picks and links to JSON/CSV.

# React landing page (optional)
- The `web/` folder contains a Vite + React SPA.
- It expects the feeds (`ev_picks.json`/`.csv`) to be available at the site root.
- For local dev (from repo root):
  ```bash
  cd web
  npm install
  npm run dev  # http://localhost:5173
  ```
- To point the app to a remote feed, set `VITE_FEED_URL` at build time:
  ```bash
  VITE_FEED_URL="https://<user>.github.io/<repo>/ev_picks.json" npm run build
  ```
- Deploy to **Vercel** or **Netlify** by connecting the `web/` directory and setting the environment variable if needed.


# ============================================================================
# File: public/logo.svg
# ============================================================================
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0%" stop-color="#25a18e"/>
      <stop offset="100%" stop-color="#38c6b0"/>
    </linearGradient>
  </defs>
  <circle cx="32" cy="32" r="30" fill="#0b0d10" stroke="url(#g)" stroke-width="4"/>
  <!-- stylized pitch -->
  <rect x="16" y="18" width="32" height="28" fill="none" stroke="#e5ecf3" stroke-width="2" rx="3"/>
  <line x1="32" y1="18" x2="32" y2="46" stroke="#e5ecf3" stroke-width="2"/>
  <circle cx="32" cy="32" r="3" fill="url(#g)"/>
  <!-- EV spark -->
  <path d="M26 50 l6-10 6 10" fill="none" stroke="url(#g)" stroke-width="3" stroke-linecap="round"/>
</svg>

# ============================================================================
# File: public/privacy.html
# ============================================================================
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Privacy — Euro Soccer Value</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <main class="container">
    <h1>Privacy Policy</h1>
    <p>We respect your privacy. This site publishes model outputs (probabilities, fair odds, EV estimates) and static feeds. We do not collect personal information unless you voluntarily provide it (e.g., subscribing to an email list).</p>
    <h3>Analytics</h3>
    <p>We may use privacy-friendly analytics to measure aggregated traffic. No personally identifiable information is stored.</p>
    <h3>Emails</h3>
    <p>If you opt-in to daily emails, your address is used solely to deliver those emails. You can unsubscribe anytime.</p>
    <h3>Data sources</h3>
    <p>Match, odds, and statistics data are sourced from third-party providers. Their privacy practices apply to their services.</p>
    <h3>Contact</h3>
    <p>Questions? Email <a href="mailto:info@example.com">info@example.com</a>.</p>
  </main>
</body>
</html>

# ============================================================================
# File: public/terms.html
# ============================================================================
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Terms — Euro Soccer Value</title>
  <link rel="stylesheet" href="styles.css" />
</head>
<body>
  <main class="container">
    <h1>Terms of Use</h1>
    <p>This site provides informational analytics only. No guarantees are made regarding accuracy or profitability. Use at your own risk.</p>
    <h3>Not investment advice</h3>
    <p>Model outputs are not financial advice or a solicitation to gamble. Check your local laws before placing any wagers.</p>
    <h3>Limitation of liability</h3>
    <p>We are not liable for any losses or damages resulting from use of the data or site.</p>
    <h3>Attribution</h3>
    <p>Data provided by third-party APIs and sources; all trademarks belong to their respective owners.</p>
    <h3>Contact</h3>
    <p>Questions? Email <a href="mailto:info@example.com">info@example.com</a>.</p>
  </main>
</body>
</html>


# ============================================================================
# File: public/CNAME
# ============================================================================
footyev.com



# ============================================================================
# DEPLOY.md additions — Custom Domain for GitHub Pages
# ============================================================================
## GitHub Pages custom domain
1. Decide your subdomain, e.g. `example.yourdomain.com` (edit `public/CNAME` to match).
2. In your DNS provider, create a **CNAME** record:
   - **Host/Name**: `example` (or the subdomain label you chose)
   - **Type**: `CNAME`
   - **Value/Target**: `<username>.github.io`  *(or `<org>.github.io` for org repos)*
   - **TTL**: automatic or 30 min
3. Commit and push `public/CNAME`.
4. In GitHub → **Settings → Pages**:
   - Under “Custom domain”, enter `example.yourdomain.com` and save.
   - Enable **Enforce HTTPS** once the certificate is provisioned (can take a few minutes).
5. Re-run **Publish Public Feeds** workflow to ensure the `CNAME` ships with the site.

### Notes
- If you’re using apex (root) domain like `yourdomain.com`, use **ALIAS/ANAME** or **A** records per your DNS provider’s guidance pointing to GitHub Pages IPs. GitHub’s docs list the current IP addresses.
- Some DNS providers cache aggressively; propagation can take up to a few hours. You can verify with `dig CNAME example.yourdomain.com`.
- The `CNAME` file must be present in the published `public/` folder every deploy; our workflow uploads the entire folder so it’s preserved.


# ============================================================================
# File: scripts/generate_sitemap.py
# ============================================================================
"""Generate sitemap.xml and robots.txt for GitHub Pages / static hosting.

Env:
  PUBLIC_BASE_URL: e.g., https://footyev.com (no trailing slash)
Output:
  public/sitemap.xml
  public/robots.txt
"""
from __future__ import annotations
import os
import datetime as dt

BASE = os.getenv("PUBLIC_BASE_URL", "https://footyev.com").rstrip("/")
now = dt.datetime.utcnow().date().isoformat()

pages = [
    ("/", "daily"),
    ("/privacy.html", "monthly"),
    ("/terms.html", "monthly"),
    ("/ev_picks.json", "hourly"),
    ("/ev_picks.csv", "hourly"),
]

os.makedirs("public", exist_ok=True)

# Write sitemap.xml
sitemap_path = os.path.join("public", "sitemap.xml")
with open(sitemap_path, "w", encoding="utf-8") as f:
    f.write("""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
""")
    for path, freq in pages:
        f.write(f"  <url>
    <loc>{BASE}{path}</loc>
    <lastmod>{now}</lastmod>
    <changefreq>{freq}</changefreq>
  </url>
")
    f.write("</urlset>
")

# Write robots.txt (points to sitemap)
robots_path = os.path.join("public", "robots.txt")
with open(robots_path, "w", encoding="utf-8") as f:
    f.write("""User-agent: *
Allow: /
Sitemap: {base}/sitemap.xml
""".format(base=BASE))

print(f"Wrote {sitemap_path} and {robots_path} for {BASE}")


# ============================================================================
# DEPLOY.md additions — SEO
# ============================================================================
## SEO
- The Pages workflow now calls `scripts/generate_sitemap.py` to build `public/sitemap.xml` and `public/robots.txt`.
- Set repo **Variables** → `PUBLIC_BASE_URL` to your site (e.g., `https://footyev.com`) so the sitemap contains correct absolute URLs.
- Re-run **Publish Public Feeds** after setting the variable.


# ============================================================================
# DEPLOY.md additions — One‑click deploy (Vercel & Netlify)
# ============================================================================

## Vercel — one‑click (React SPA in `web/`)
Use this button after pushing your repo to GitHub/GitLab/Bitbucket (replace `<YOUR_REPO_URL>` in the link if you fork first):

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=<YOUR_REPO_URL>&project-name=euro-soccer-value&repository-name=euro-soccer-value&root-directory=web&build-command=npm%20run%20build&output-directory=dist&env=VITE_FEED_URL,VITE_META_URL)

**Required env vars**
- `VITE_FEED_URL` → e.g., `https://footyev.com/ev_picks.json`
- `VITE_META_URL` → e.g., `https://footyev.com/meta.json`

**Project settings** (Vercel UI → Project → Settings → General)
- **Framework preset**: Vite
- **Root Directory**: `web`
- **Build Command**: `npm run build`
- **Output Directory**: `dist`

> Note: The Python API/Streamlit dashboard are best hosted on Render/Streamlit Cloud. You can still deploy the API to Vercel using the provided `vercel.json` (Python serverless), but it’s optional if you’re only shipping the SPA.

---

## Netlify — one‑click (React SPA in `web/`)
Use this button (replace `<YOUR_REPO_URL>`):

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=<YOUR_REPO_URL>)

**Build settings** (Netlify UI → Site settings → Build & deploy)
- **Base directory**: `web`
- **Build command**: `npm run build`
- **Publish directory**: `dist`

**Environment variables**
- `VITE_FEED_URL` → `https://footyev.com/ev_picks.json`
- `VITE_META_URL` → `https://footyev.com/meta.json`

**Optional**: You can commit `web/netlify.toml` (below) to bake in the config.

```toml
# web/netlify.toml
[build]
  base = "web"
  command = "npm run build"
  publish = "dist"

[build.environment]
  VITE_FEED_URL = "https://footyev.com/ev_picks.json"
  VITE_META_URL = "https://footyev.com/meta.json"
```

---

## After deploy
1) Verify the SPA loads the feed and badge:
   - `${VITE_FEED_URL}` (JSON) and `${VITE_META_URL}` (meta) must be publicly reachable.
2) Point these to your **GitHub Pages** domain if you’re hosting the feeds there:
   - Example: `https://<username>.github.io/<repo>/ev_picks.json`
3) (Optional) Add a `/_headers` or `netlify.toml` cache policy for the JSON if you want short TTLs.


# ============================================================================
# File: LICENSE
# ============================================================================
MIT License

Copyright (c) 2025 Your Name or Company

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# ============================================================================
# File: CONTRIBUTING.md
# ============================================================================
# Contributing Guide

Thanks for taking the time to contribute! This project powers a public EV feed and dashboards for **EPL + EFL Championship**. Contributions that improve data quality, modeling, UX, docs, or deploy automation are welcome.

## Getting Started
1. **Fork** the repo and create a feature branch: `git checkout -b feat/my-change`  
2. **Set up** local env (Python 3.11 + Node 18+):
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   cd web && npm install && cd ..
   cp .env.example .env  # add API_FOOTBALL_KEY
   ```
3. **Smoke test**:
   ```bash
   python scripts/test_ingest.py --season 2024 --leagues EPL CHAMP --with-odds
   streamlit run app_glm.py  # http://localhost:8501
   uvicorn api:app --reload --port 8001  # http://localhost:8001/health
   npm --prefix web run build
   ```

## Development Guidelines
- **Python style**: PEP8; type hints where practical. Prefer small, pure functions.
- **Modeling**: add unit tests or notebooks that show calibration/ROI deltas; avoid look‑ahead bias.
- **Config & Secrets**: never commit secrets. Use `.env` locally and GitHub **Secrets/Variables** in CI.
- **Data**: do not commit provider datasets with license restrictions. Commit *schemas* and *sample rows* only.
- **Commits**: use Conventional Commits where possible, e.g. `feat: add GLM regularization to away defense`.
- **Docs**: update `README.md` and `DEPLOY.md` when behavior or deploy steps change.

## PR Checklist
- [ ] Code builds locally (dashboard/API), and `scripts/test_ingest.py` runs without error.  
- [ ] Added/updated docs (README/DEPLOY) and examples if needed.  
- [ ] No secrets, tokens, or private data added.  
- [ ] If changing the model, include a brief **before/after** calibration or ROI snapshot.  
- [ ] Accessibility: basic keyboard nav & readable contrasts for UI changes.

## Issues & Discussions
- **Bug reports**: include OS, Python/Node versions, logs, and reproduction steps.  
- **Feature requests**: describe the problem and proposed UX.  
- **Model ideas**: share a quick offline validation (CSV or notebook) if possible.

## Code of Conduct
This project follows the **Contributor Covenant v2.1**. Be kind, inclusive, and constructive.  
Report unacceptable behavior to `info@example.com`.

---

# ============================================================================
# File: SECURITY.md
# ============================================================================
# Security Policy

We take data, dependency, and credential security seriously. Please follow the guidance below.

## Supported Versions
We actively maintain the `main` branch. Security fixes will be patched there and released via GitHub Actions.

## Reporting a Vulnerability
- Email: **security@example.com** (replace with your address)  
- Or use **GitHub Security Advisories** for a private report.  
Please include reproduction steps, impact, and suggested fixes if known. We aim to acknowledge within 72 hours.

## Secrets & Credentials
- Never commit API keys or tokens. Use `.env` locally and GitHub **Secrets** in workflows.  
- Rotate keys if you suspect exposure.  
- Public artifacts (JSON/CSV, Pages) must **not** contain PII or provider‑restricted content.

## Dependency Security
- Keep Python/Node deps updated. We recommend enabling **Dependabot** for `pip` and `npm`.  
- If a CVE affects a core dependency, open an issue with upgrade plan and test notes.

## Data Providers & Rate Limits
- Respect all provider terms (API‑Football/Sportmonks/etc.).  
- Handle rate limits gracefully; avoid aggressive parallelization.  
- Do not redistribute provider raw datasets unless the license allows it.

## Email & Automation
- Outbound email (`scripts/email_top_ev.py`) should use verified senders (Resend/SMTP).  
- Add unsubscribe links for mailing lists and comply with local regulations.

## Disclosure
If a fix requires coordinated disclosure, we will work with reporters to agree on a timeline, typically 7–30 days, depending on severity.


# ============================================================================
# File: .github/dependabot.yml
# ============================================================================
version: 2
updates:
  # Python dependencies (root)
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "04:00"
      timezone: "UTC"
    open-pull-requests-limit: 10
    labels: ["deps", "python"]
    groups:
      minor-and-patch:
        update-types: ["minor", "patch"]
    # Example: ignore major bumps for key libs until manually reviewed
    ignore:
      - dependency-name: "pandas"
        update-types: ["version-update:semver-major"]
      - dependency-name: "numpy"
        update-types: ["version-update:semver-major"]

  # Node dependencies for React SPA (web/)
  - package-ecosystem: "npm"
    directory: "/web"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "04:10"
      timezone: "UTC"
    open-pull-requests-limit: 10
    labels: ["deps", "frontend"]
    groups:
      minor-and-patch:
        update-types: ["minor", "patch"]

  # GitHub Actions workflow dependencies
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "04:20"
      timezone: "UTC"
    labels: ["deps", "ci"]


# ============================================================================
# File: .github/workflows/dependabot-automerge.yml
# ============================================================================
name: Dependabot Auto‑Merge (safe updates)

on:
  pull_request_target:
    types: [opened, synchronize, reopened]

permissions:
  contents: write
  pull-requests: write

jobs:
  test:
    if: github.actor == 'dependabot[bot]'
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install Python deps
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt
          # quick import smoke test for key libs
          python - << 'PY'
import pandas, numpy, scipy, statsmodels, sklearn
print('python-deps-ok')
PY

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: web/package-lock.json

      - name: Build React SPA (web/)
        run: |
          npm ci --prefix web
          npm run build --prefix web

  automerge:
    if: github.actor == 'dependabot[bot]'
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Fetch Dependabot metadata
        id: meta
        uses: dependabot/fetch-metadata@v2
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}

      - name: Approve PR (non‑major updates only)
        if: steps.meta.outputs.update-type != 'version-update:semver-major'
        uses: peter-evans/approve-pull-request@v5
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          pull-request-number: ${{ github.event.pull_request.number }}
          message: "Auto‑approved by CI (Dependabot safe update)."

      - name: Enable auto‑merge (squash)
        if: steps.meta.outputs.update-type != 'version-update:semver-major'
        uses: peter-evans/enable-pull-request-automerge@v3
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          pull-request-number: ${{ github.event.pull_request.number }}
          merge-method: squash

      - name: Comment when skipping major updates
        if: steps.meta.outputs.update-type == 'version-update:semver-major'
        uses: thollander/actions-comment-pull-request@v2
        with:
          message: |
            Skipping auto‑merge: **major** update detected (`${{ steps.meta.outputs.dependency-names }}` @ `${{ steps.meta.outputs.new-version }}`).
            Please review manually.


# ============================================================================
# File: tests/test_poisson.py
# ============================================================================
import math
import numpy as np
from model_poisson import score_matrix, win_draw_probs, over_under_prob


def test_score_matrix_prob_mass():
    M = score_matrix(1.4, 1.1)
    assert np.isclose(M.sum(), 1.0, atol=1e-6)


def test_win_draw_probs_sum_to_one():
    ph, pd, pa = win_draw_probs(1.6, 1.2)
    assert math.isclose(ph + pd + pa, 1.0, rel_tol=1e-6)


def test_over_under_complements():
    over, under = over_under_prob(1.3, 0.9, line=2.5)
    assert math.isclose(over + under, 1.0, rel_tol=1e-6)


# ============================================================================
# File: tests/test_features_and_ev.py
# ============================================================================
import pandas as pd
from features import build_team_strengths, make_prediction_frame
from ev_scanner import scan_ev


def _toy_matches():
    return pd.DataFrame([
        {"match_id": 1, "league": "EPL", "season": 2024, "date": "2024-08-12",
         "home_team": "Team A", "away_team": "Team B", "home_goals": 2, "away_goals": 1,
         "book_home_odds": 2.0, "book_draw_odds": 3.5, "book_away_odds": 3.8},
        {"match_id": 2, "league": "EPL", "season": 2024, "date": "2024-08-19",
         "home_team": "Team B", "away_team": "Team A", "home_goals": 0, "away_goals": 1,
         "book_home_odds": 2.6, "book_draw_odds": 3.2, "book_away_odds": 2.8},
    ])


def test_strengths_positive():
    df = _toy_matches()
    strengths = build_team_strengths(df)
    assert set(["team", "atk", "def"]).issubset(strengths.columns)
    assert (strengths["atk"] > 0).all()
    assert (strengths["def"] > 0).all()


def test_make_prediction_and_ev_scan():
    df = _toy_matches()
    strengths = build_team_strengths(df)
    pred = make_prediction_frame(df, strengths)
    assert {"xg_home", "xg_away"}.issubset(pred.columns)
    ev = scan_ev(pred, min_edge=-1.0)  # no filtering for test
    # Should compute EV columns and pick a best side
    assert {"best_side", "best_ev", "chosen_edge"}.issubset(ev.columns)
    assert len(ev) >= 1


# ============================================================================
# File: tests/test_glm_ratings.py
# ============================================================================
import pandas as pd
from models.glm_ratings import make_prediction_frame_glm


def test_glm_returns_xg_columns():
    # Small synthetic dataset with two teams and 4 matches
    df = pd.DataFrame([
        {"match_id": 10, "league": "CHAMP", "season": 2024, "date": "2024-08-01",
         "home_team": "Club X", "away_team": "Club Y", "home_goals": 1, "away_goals": 0,
         "book_home_odds": 2.1, "book_draw_odds": 3.2, "book_away_odds": 3.9},
        {"match_id": 11, "league": "CHAMP", "season": 2024, "date": "2024-08-08",
         "home_team": "Club Y", "away_team": "Club X", "home_goals": 2, "away_goals": 2,
         "book_home_odds": 2.4, "book_draw_odds": 3.1, "book_away_odds": 2.9},
        {"match_id": 12, "league": "CHAMP", "season": 2024, "date": "2024-08-15",
         "home_team": "Club X", "away_team": "Club Y", "home_goals": 0, "away_goals": 1,
         "book_home_odds": 2.2, "book_draw_odds": 3.0, "book_away_odds": 3.6},
        {"match_id": 13, "league": "CHAMP", "season": 2024, "date": "2024-08-22",
         "home_team": "Club Y", "away_team": "Club X", "home_goals": 3, "away_goals": 1,
         "book_home_odds": 2.3, "book_draw_odds": 3.4, "book_away_odds": 3.4},
    ])
    pred = make_prediction_frame_glm(df)
    assert {"xg_home", "xg_away"}.issubset(pred.columns)
    assert pred["xg_home"].notna().all()
    assert pred["xg_away"].notna().all()


# ============================================================================
# File: pytest.ini
# ============================================================================
[pytest]
addopts = -q --cov=. --cov-report=term-missing
pythonpath = .


# ============================================================================
# File: .github/workflows/ci.yml
# ============================================================================
name: CI

on:
  push:
    branches: [ main, master ]
  pull_request:
    branches: [ "*" ]

permissions:
  contents: read

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install -r requirements.txt

      - name: Run tests
        env:
          PYTHONPATH: .
        run: |
          source .venv/bin/activate
          pytest

      - name: Upload coverage artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: ./.coverage*


# ============================================================================
# File: examples/quickstart.ipynb
# ============================================================================
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Euro Soccer Value — Quickstart Notebook (EPL + Championship)
",
    "
",
    "This notebook shows an end-to-end mini run:
",
    "
",
    "1. Load processed matches for a season (or create a tiny synthetic sample if none exist).
",
    "2. Generate predictions (baseline Poisson and GLM).
",
    "3. Scan for EV vs. bookmaker odds.
",
    "4. Preview top opportunities.
",
    "
",
    "> **Tip:** Run `scripts/test_ingest.py --season 2024 --leagues EPL CHAMP --with-odds` first to use real data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os, pandas as pd, numpy as np
",
    "from config import settings
",
    "from features import build_team_strengths, make_prediction_frame
",
    "from models.glm_ratings import make_prediction_frame_glm
",
    "from ev_scanner import scan_ev
",
    "
",
    "SEASON = 2024
",
    "LEAGUES = ['EPL','CHAMP']
",
    "paths = [os.path.join(settings.proc_dir, f'matches_{lg}_{SEASON}.csv') for lg in LEAGUES]
",
    "
",
    "dfs = []
",
    "for p in paths:
",
    "    if os.path.exists(p):
",
    "        dfs.append(pd.read_csv(p))
",
    "
",
    "if not dfs:
",
    "    # Fallback: tiny synthetic sample (structure matches pipeline expectations)
",
    "    print('No processed files found; using synthetic sample…')
",
    "    df_syn = pd.DataFrame([
",
    "        {'match_id': 1, 'league': 'CHAMP', 'season': SEASON, 'date': '2024-08-10', 'home_team': 'Club A', 'away_team': 'Club B', 'home_goals': 2, 'away_goals': 1, 'book_home_odds': 2.10, 'book_draw_odds': 3.30, 'book_away_odds': 3.60},
",
    "        {'match_id': 2, 'league': 'CHAMP', 'season': SEASON, 'date': '2024-08-17', 'home_team': 'Club B', 'away_team': 'Club A', 'home_goals': 0, 'away_goals': 1, 'book_home_odds': 2.60, 'book_draw_odds': 3.10, 'book_away_odds': 2.85},
",
    "        {'match_id': 3, 'league': 'EPL',   'season': SEASON, 'date': '2024-08-12', 'home_team': 'United', 'away_team': 'City',    'home_goals': 1, 'away_goals': 1, 'book_home_odds': 2.80, 'book_draw_odds': 3.25, 'book_away_odds': 2.55},
",
    "    ])
",
    "    dfs = [df_syn]
",
    "
",
    "df = pd.concat(dfs, ignore_index=True)
",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Baseline Poisson path
",
    "strengths = build_team_strengths(df)
",
    "pred_base = make_prediction_frame(df, strengths)
",
    "ev_base = scan_ev(pred_base, min_edge=0.03)
",
    "
",
    "# GLM path (better ratings)
",
    "pred_glm = make_prediction_frame_glm(df)
",
    "ev_glm = scan_ev(pred_glm, min_edge=0.03)
",
    "
",
    "print('Baseline EV picks:', len(ev_base))
",
    "print('GLM EV picks:', len(ev_glm))
",
    "ev_glm.sort_values('best_ev', ascending=False).head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Notes
",
    "- **Min Edge** is the probability gap between the model and market on the chosen side.
",
    "- Use **closing odds** for best validation where possible.
",
    "- For production feeds, run the nightly GitHub Action which ingests, predicts, and publishes to Pages."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}


# ============================================================================
# File: examples/quickstart.md
# ============================================================================
# Euro Soccer Value — Quickstart (Markdown)

This walkthrough gets you from clone → data → predictions → EV picks in ~10 minutes.

> Tip: If you prefer an interactive demo, open `examples/quickstart.ipynb` instead.

---

## 1) Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # paste your API_FOOTBALL_KEY
```

## 2) Ingest data (EPL + Championship)
Pull one season and include odds (Bet365 example via API‑Football):
```bash
python scripts/test_ingest.py --season 2024 --leagues EPL CHAMP --with-odds
```
This writes CSVs to `data/processed/matches_{LEAGUE}_{SEASON}.csv`.

## 3) Generate predictions & EV (two paths)
### A) Baseline Poisson
```python
from config import settings
import pandas as pd
from features import build_team_strengths, make_prediction_frame
from ev_scanner import scan_ev

df = pd.read_csv(f"{settings.proc_dir}/matches_CHAMP_2024.csv")
strengths = build_team_strengths(df)
pred = make_prediction_frame(df, strengths)
ev = scan_ev(pred, min_edge=0.03)
ev.sort_values('best_ev', ascending=False).head(10)
```

### B) GLM ratings (recommended)
```python
import pandas as pd
from models.glm_ratings import make_prediction_frame_glm
from ev_scanner import scan_ev

df = pd.read_csv('data/processed/matches_EPL_2024.csv')
pred = make_prediction_frame_glm(df)
ev = scan_ev(pred, min_edge=0.03)
ev[['date','home_team','away_team','best_side','chosen_edge','best_ev']].head(10)
```

## 4) Backtest
```python
from backtest import simple_backtest
print(simple_backtest('CHAMP', [2024], min_edge=0.03))
print(simple_backtest('EPL', [2024],   min_edge=0.03))
```
Outputs a summary CSV to `data/processed/bt_summary_{LEAGUE}.csv`.

## 5) Dashboard & API
- **Streamlit GLM dashboard**:
  ```bash
  streamlit run app_glm.py  # http://localhost:8501
  ```
- **FastAPI JSON**:
  ```bash
  uvicorn api:app --reload --port 8001  # http://localhost:8001/health
  ```
- **Docker Compose (both)**:
  ```bash
  docker compose up --build
  # dashboard: http://localhost:8501, api: http://localhost:8001
  ```

## 6) Public feeds (JSON/CSV) & site
Generate and preview locally:
```bash
python scripts/generate_public_feed.py
# opens: public/ev_picks.json, public/ev_picks.csv, public/meta.json
```
Open `public/index.html` to see the styled homepage and status badge.

## 7) CI/CD (optional but recommended)
- Nightly ingest & backtest: `.github/workflows/nightly_ingest.yml`
- Publish to Pages: `.github/workflows/publish_pages.yml`
- Daily email: `.github/workflows/daily_email.yml`
- CI tests: `.github/workflows/ci.yml`
- Dependabot + automerge: `.github/dependabot.yml`, `dependabot-automerge.yml`

## 8) Front‑end (React SPA)
```bash
cd web
npm install
VITE_FEED_URL="https://<your-domain>/ev_picks.json" \
VITE_META_URL="https://<your-domain>/meta.json" \
npm run build
```
Deploy `web/dist/` to Vercel/Netlify (see `DEPLOY.md`).

## 9) Troubleshooting
- **No picks shown**: Increase `min_edge` (lower threshold), ensure odds are present in CSVs.
- **API rate limits**: Add sleeps in `ingest_api.py` or upgrade plan with provider.
- **Mismatch team names**: Normalize team strings before computing strengths/ratings.
- **GLM convergence**: Reduce `alpha`, check dataset size, or cap maximum goals.

## 10) Next steps
- Add lineup/injury adjustments and referee effects.
- Track CLV vs closing lines and publish a public results page.
- Introduce subscription gating (Stripe) and alerts (email/Discord bots).


# ============================================================================
# File: auth/supabase_auth.py
# ============================================================================
from __future__ import annotations
import os, time
from typing import Optional, Dict, Any
import httpx
from jose import jwt
from cachetools import TTLCache

JWKS_CACHE = TTLCache(maxsize=1, ttl=3600)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
JWKS_URL = os.getenv("SUPABASE_JWKS_URL", f"{SUPABASE_URL}/auth/v1/keys")

class AuthError(Exception):
    pass

def _fetch_jwks() -> Dict[str, Any]:
    if 'jwks' in JWKS_CACHE:
        return JWKS_CACHE['jwks']
    with httpx.Client(timeout=10.0) as client:
        r = client.get(JWKS_URL)
        r.raise_for_status()
        data = r.json()
    JWKS_CACHE['jwks'] = data
    return data

def verify_bearer_token(auth_header: str) -> Dict[str, Any]:
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise AuthError("Missing bearer token")
    token = auth_header.split(" ", 1)[1].strip()
    jwks = _fetch_jwks()
    try:
        unverified = jwt.get_unverified_header(token)
        kid = unverified.get('kid')
        key = None
        for k in jwks.get('keys', []):
            if k.get('kid') == kid:
                key = k
                break
        if not key:
            raise AuthError("JWKS key not found")
        claims = jwt.decode(
            token,
            key,
            algorithms=[key.get('alg', 'RS256')],
            audience=["authenticated"],
            issuer=f"{SUPABASE_URL}/auth/v1",
            options={"verify_at_hash": False},
        )
        return claims
    except Exception as e:
        raise AuthError(f"JWT verification failed: {e}")

# ============================================================================
# File: db/supabase_client.py
# ============================================================================
from __future__ import annotations
import os
from supabase import create_client, Client

_url = os.getenv("SUPABASE_URL")
_key = os.getenv("SUPABASE_SERVICE_ROLE")  # server-only key
_sb: Client | None = None

def get_client() -> Client:
    global _sb
    if _sb is None:
        if not _url or not _key:
            raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE env")
        _sb = create_client(_url, _key)
    return _sb

# ============================================================================
# File: billing/stripe_webhooks.py
# ============================================================================
from __future__ import annotations
import os, json
from fastapi import APIRouter, Request, HTTPException
import stripe
from db.supabase_client import get_client

router = APIRouter()

stripe.api_key = os.getenv("STRIPE_API_KEY")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

ACTIVE_STATUSES = {"active", "trialing"}

def _upsert_subscription(email: str, status: str, customer_id: str | None, sub_id: str | None, current_period_end: int | None):
    sb = get_client()
    data = {
        "email": email.lower(),
        "status": status,
        "stripe_customer_id": customer_id,
        "stripe_subscription_id": sub_id,
    }
    if current_period_end:
        # Stripe gives seconds; Postgres expects ISO or timestamp
        from datetime import datetime, timezone
        data["current_period_end"] = datetime.fromtimestamp(current_period_end, tz=timezone.utc).isoformat()
    # upsert by email
    sb.table("user_subscriptions").upsert(data, on_conflict="email").execute()

@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    if not WEBHOOK_SECRET:
        raise HTTPException(500, "Webhook secret not configured")
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    except Exception as e:
        raise HTTPException(400, f"Webhook signature error: {e}")

    et = event.get("type")
    data = event.get("data", {}).get("object", {})

    # Try to resolve customer email
    email = (
        data.get("customer_details", {}).get("email")
        or data.get("receipt_email")
        or data.get("customer_email")
    )

    if et == "checkout.session.completed":
        # Initial purchase
        status = "active" if data.get("payment_status") == "paid" else "trialing"
        sub_id = data.get("subscription")
        cust_id = data.get("customer")
        _upsert_subscription(email, status, cust_id, sub_id, None)

    elif et in {"customer.subscription.updated", "customer.subscription.created"}:
        status = data.get("status")
        sub_id = data.get("id")
        cust_id = data.get("customer")
        cpe = data.get("current_period_end")
        # try expand customer to get email if missing
        if not email and cust_id:
            cust = stripe.Customer.retrieve(cust_id)
            email = cust.get("email")
        _upsert_subscription(email, status, cust_id, sub_id, cpe)

    elif et == "customer.subscription.deleted":
        status = "canceled"
        sub_id = data.get("id")
        cust_id = data.get("customer")
        if not email and cust_id:
            cust = stripe.Customer.retrieve(cust_id)
            email = cust.get("email")
        _upsert_subscription(email, status, cust_id, sub_id, None)

    return {"ok": True}

# ============================================================================
# File: routes/feed.py
# ============================================================================
from __future__ import annotations
import os, json, datetime as dt
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, Header, HTTPException
from auth.supabase_auth import verify_bearer_token, AuthError
from db.supabase_client import get_client

router = APIRouter(prefix="/feed", tags=["feed"])

DATA_PUBLIC = os.path.join("public", "ev_picks.json")

# --- Auth helpers ---

def _is_active_sub(email: str) -> bool:
    sb = get_client()
    res = sb.table("user_subscriptions").select("status,current_period_end").eq("email", email.lower()).order("updated_at", desc=True).limit(1).execute()
    rows = res.data or []
    if not rows:
        return False
    row = rows[0]
    status = (row.get("status") or "").lower()
    if status in ("active", "trialing"):
        # If a period end is set, ensure it's in the future
        cpe = row.get("current_period_end")
        if cpe:
            try:
                # Accept both ISO and timestamp
                if isinstance(cpe, (int, float)):
                    cpe_dt = dt.datetime.utcfromtimestamp(int(cpe))
                else:
                    cpe_dt = dt.datetime.fromisoformat(str(cpe).replace("Z","+00:00")).astimezone(dt.timezone.utc).replace(tzinfo=None)
                return cpe_dt > dt.datetime.utcnow()
            except Exception:
                return True
        return True
    return False

async def require_user(authorization: str = Header(None)) -> Dict[str, Any]:
    try:
        claims = verify_bearer_token(authorization)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    email = claims.get("email")
    if not email:
        raise HTTPException(401, "No email claim in token")
    return {"email": email, "sub": claims.get("sub")}

async def require_active_sub(user=Depends(require_user)) -> Dict[str, Any]:
    if not _is_active_sub(user["email"]):
        raise HTTPException(status_code=402, detail="Subscription required")
    return user

# --- Routes ---
@router.get("/public")
def public_feed(delay_days: int = 1) -> List[Dict[str, Any]]:
    """Return the public feed, optionally delaying by N days (default 1)."""
    if not os.path.exists(DATA_PUBLIC):
        return []
    with open(DATA_PUBLIC, "r", encoding="utf-8") as f:
        data = json.load(f)
    if delay_days <= 0:
        return data
    cutoff = dt.date.today() - dt.timedelta(days=delay_days)
    out = [r for r in data if (dt.date.fromisoformat(r.get("date", str(cutoff))) <= cutoff)]
    return out

@router.get("/pro")
def pro_feed(user=Depends(require_active_sub)) -> List[Dict[str, Any]]:
    """Return the pro (same-day) feed for active subscribers."""
    if not os.path.exists(DATA_PUBLIC):
        return []
    with open(DATA_PUBLIC, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# ============================================================================
# File: api.py (append router includes)
# ============================================================================
from fastapi import FastAPI
from routes.feed import router as feed_router
from billing.stripe_webhooks import router as stripe_router

app = FastAPI(title="FootyEV API", version="0.1.0")

@app.get("/health")
def health():
    return {"ok": True}

app.include_router(feed_router)
app.include_router(stripe_router, tags=["webhooks"])

# ============================================================================
# File: db/schema.sql
# ============================================================================
-- Run this in Supabase SQL editor
create table if not exists public.user_subscriptions (
  id uuid primary key default gen_random_uuid(),
  email text not null unique,
  status text not null,
  current_period_end timestamptz null,
  stripe_customer_id text null,
  stripe_subscription_id text null,
  updated_at timestamptz not null default now()
);
-- (Optional) RLS off for this table since service role will access; keep private
alter table public.user_subscriptions enable row level security;
create policy service_read on public.user_subscriptions for select using (auth.role() = 'service_role');
create policy service_write on public.user_subscriptions for insert with check (auth.role() = 'service_role');
create policy service_update on public.user_subscriptions for update using (auth.role() = 'service_role');
create index if not exists user_subscriptions_email_idx on public.user_subscriptions (email);

-- You may prefer to keep RLS disabled and only access with service role from server.

# ============================================================================
# File: DEPLOY.md (Subscription API section)
# ============================================================================
## Subscription API (Supabase + Stripe)

### 1) Provision Supabase
- Create a project → copy **SUPABASE_URL**, **anon** and **service role** keys.
- In the SQL editor, run `db/schema.sql` to create `user_subscriptions`.
- In **Project Settings → API**, copy the **JWKS URL** (`…/auth/v1/keys`).

### 2) Configure env
Add to repo **Secrets** (server):
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE`, `SUPABASE_JWKS_URL`
- `STRIPE_API_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID`

### 3) Deploy API (Render)
- Start cmd: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Env: same as above + `API_FOOTBALL_KEY` if needed elsewhere.

### 4) Stripe webhook
- In Stripe → Developers → Webhooks → Add endpoint: `https://<your-api-domain>/webhooks/stripe`
- Select events: `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated`, `customer.subscription.deleted`.
- Use the **signing secret** as `STRIPE_WEBHOOK_SECRET`.

### 5) Auth flow (frontend)
- Sign users in with **Supabase Auth** (email magic link or OAuth) on your SPA.
- Send API requests with `Authorization: Bearer <supabase_access_token>`.
- The API verifies JWT via JWKS and checks `user_subscriptions` by **email**.

### 6) Endpoints
- `GET /feed/public?delay_days=1` → delayed list (no auth)
- `GET /feed/pro` → **requires** active sub (Bearer token)
- `POST /webhooks/stripe` → Stripe sends updates; we upsert by email.

### 7) Notes
- To map Stripe → email, we read `customer_details.email` from events.
- If you run Checkout on your SPA, set `customer_email` or `client_reference_id` to ensure we can reconcile.
- For stricter mapping, add a `stripe_customers` table keyed by user id; extend the webhook to upsert that mapping.
