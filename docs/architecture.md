# Architecture

World Cup AI Predictor is organized as a modular prediction pipeline. Each module can run against the included sample data now and can be replaced with richer real data later.

## Performance Engine

The performance engine lives in `src/features/performance.py`.

It merges current tournament stats for both teams into each match row, including goals scored, goals conceded, xG, xGA, shots, possession, and rest days. It also creates delta features such as `xg_delta`, `shots_delta`, and `rest_days_delta`.

In real usage, this data must only include matches played before the target fixture.

## Tactical Engine

The tactical engine lives in `src/features/tactics.py`.

It merges team tactical profiles and creates matchup features such as pressing versus buildup, counterattack versus transition defense, set-piece edges, wide attack edges, aerial edge, and an overall `tactical_edge`.

The sample tactical values are manually scored from 1 to 10. They are placeholders for future event-data or scouting-based inputs.

## H2H Engine

The head-to-head engine lives in `src/features/h2h.py`.

It finds historical meetings between two teams in either direction, filters out future matches, and returns recent H2H counts, goals, goal difference, and a bounded weighted score. Recent matches and competitive tournaments are weighted slightly more than older friendlies.

H2H is intentionally small in the ensemble so it cannot dominate the prediction.

## Elo Engine

The Elo engine lives in `src/features/elo.py`.

It calculates chronological team ratings from historical scorelines. For each match, it records pre-match and post-match Elo. Feature generation uses pre-match Elo only, which prevents data leakage from the match being predicted.

Teams without prior history start at 1500.

## Result Model

The result model trainer lives in `src/models/train_result_model.py`.

It trains a three-class classifier:

- `0`: Team A win
- `1`: Draw
- `2`: Team B win

The trainer tries XGBoost first when available and falls back to scikit-learn RandomForest. It saves the model to `models/result_model.pkl` and metadata to `models/result_model_metadata.json`.

## Poisson Model

The score model lives in `src/models/poisson_model.py`.

It estimates team attack and defensive weakness from historical scorelines, then computes expected goals using attack strength, opponent defensive weakness, Elo difference, and tactical edge. It creates a Poisson scoreline grid and returns likely scores plus W/D/L probabilities.

Team strengths are saved to `models/team_strength.csv`.

## Ensemble Engine

The ensemble lives in `src/prediction/ensemble.py`.

It combines:

- Result ML model
- Poisson score model
- Elo probability model
- H2H and tactical adjustments

The default weighting is:

- 45% result model
- 25% Poisson model
- 20% Elo model
- 10% H2H/tactical adjustment

The output includes W/D/L probabilities, knockout advance probabilities, expected goals, likely scorelines, confidence, and explanation factors.

## Tournament Simulator

The knockout simulator lives in `src/prediction/simulate_tournament.py`.

It reads fixture CSVs with:

```text
match_id,round,team_a,team_b,next_match_id
```

It simulates each matchup through the ensemble engine, advances winners through `next_match_id`, and estimates each team's chance to reach later rounds and become champion.

Results are saved to `data/processed/simulation_results.csv`.
