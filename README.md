# World Cup AI Predictor

World Cup AI Predictor is a modular Python and React project for predicting football World Cup matches. It combines current tournament performance, tactical matchup profiles, head-to-head history, Elo strength, a result classifier, a Poisson-style score model, an ensemble prediction engine, a FastAPI backend, a knockout simulator, and a simple React dashboard.

The included data is realistic sample data for development and testing. It is not official data. The code is designed so real datasets and APIs can be plugged in later without replacing the whole architecture.

## Architecture

Core pipeline:

```text
sample/raw data
  -> data loaders
  -> feature engines
  -> training dataset
  -> result model + Poisson model
  -> ensemble predictor
  -> CLI / FastAPI / React dashboard / tournament simulator
```

Main modules:

- `src/data/load_data.py` - CSV loading and required-column validation
- `src/data/build_dataset.py` - model-ready training dataset builder
- `src/features/elo.py` - chronological Elo engine
- `src/features/performance.py` - current tournament performance features
- `src/features/tactics.py` - tactical matchup features
- `src/features/h2h.py` - recent weighted head-to-head features
- `src/models/train_result_model.py` - XGBoost or RandomForest result classifier
- `src/models/poisson_model.py` - expected goals and scoreline probabilities
- `src/prediction/ensemble.py` - final blended prediction engine
- `src/prediction/predict_match.py` - command-line match prediction
- `src/prediction/simulate_tournament.py` - knockout bracket simulator
- `api/main.py` - FastAPI backend
- `frontend/` - Vite React dashboard

More detail is in [docs/architecture.md](docs/architecture.md).

## Data Sources Expected

The sample files live in `data/sample/`:

- `matches.csv` - historical match results and knockout advancement
- `team_tactics.csv` - tactical team profiles scored from 1 to 10
- `current_worldcup_performance.csv` - current tournament team stats
- `h2h.csv` - head-to-head match history
- `knockout_fixtures.csv` - sample knockout bracket fixtures

For production use, replace these with larger and verified sources:

- Historical international results
- Current World Cup match and team feeds
- xG and shot data
- Player availability and rest data
- Tactical/event data
- Elo or other team-strength ratings
- Tournament bracket fixtures

Important: prediction features must only use information available before the match being predicted.

## Install

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

Frontend dependencies:

```bash
cd frontend
npm install
```

## Build Dataset

```bash
python -m src.data.build_dataset
```

Creates:

```text
data/processed/training_dataset.csv
```

## Train Model

```bash
python -m src.models.train_result_model
```

The trainer uses XGBoost when available and falls back to scikit-learn RandomForest.

Creates:

```text
models/result_model.pkl
models/result_model_metadata.json
```

Build Poisson team strengths:

```bash
python -m src.models.poisson_model
```

Creates:

```text
models/team_strength.csv
```

## Predict From CLI

```bash
python -m src.prediction.predict_match --team-a France --team-b Paraguay --stage R16 --knockout true
```

The CLI prints 90-minute probabilities, knockout advance probabilities, expected goals, likely scorelines, and explanation factors.

## Run API

```bash
uvicorn api.main:app --reload
```

Useful endpoints:

- `GET /` - project status
- `GET /health` - health check
- `GET /teams` - teams available in sample data
- `POST /predict-match` - match prediction
- `GET /docs` - Swagger UI

Example `POST /predict-match` body:

```json
{
  "team_a": "France",
  "team_b": "Paraguay",
  "stage": "R16",
  "is_knockout": true
}
```

## Run Frontend

Start the backend first:

```bash
uvicorn api.main:app --reload
```

Then start the dashboard:

```bash
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173/
```

The dashboard calls `POST http://127.0.0.1:8000/predict-match`. Set `VITE_API_BASE_URL` if the backend runs elsewhere.

## Simulate Tournament

```bash
python -m src.prediction.simulate_tournament --simulations 10000
```

Reads:

```text
data/sample/knockout_fixtures.csv
```

Writes:

```text
data/processed/simulation_results.csv
```

Fixture CSV format:

```text
match_id,round,team_a,team_b,next_match_id
```

## Run Tests

```bash
python -m pytest
```

Frontend build check:

```bash
cd frontend
npm run build
```

## Final Example Command Flow

```bash
pip install -r requirements.txt
python -m src.data.build_dataset
python -m src.models.train_result_model
python -m src.prediction.predict_match --team-a France --team-b Paraguay --stage R16 --knockout true
uvicorn api.main:app --reload
```

For the React dashboard:

```bash
cd frontend
npm install
npm run dev
```

## Model Limitations

Predictions are probabilities, not guarantees. Football is low scoring and highly variable. The sample data is small and can mislead. H2H is intentionally weighted lightly. Tactical values are manually scored until real event or scouting data is integrated.

Read [docs/model_limitations.md](docs/model_limitations.md) before interpreting outputs.

## Future Improvements

- Replace sample CSVs with official or licensed data feeds
- Add player-level injuries, suspensions, minutes, and fatigue
- Add market odds as calibration features
- Add proper time-aware backtesting
- Tune model hyperparameters on larger historical data
- Calibrate result probabilities
- Add model versioning and experiment tracking
- Add richer React views for tournament simulation and team comparison
- Add Docker Compose for backend/frontend startup
- Add deployment configuration for cloud hosting
