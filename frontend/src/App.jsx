import { useEffect, useMemo, useState } from 'react';
import { fetchTeams, predictMatch } from './api';
import './App.css';

const FALLBACK_TEAMS = [
  'Argentina',
  'Belgium',
  'Brazil',
  'Canada',
  'Colombia',
  'Egypt',
  'England',
  'France',
  'Mexico',
  'Morocco',
  'Norway',
  'Paraguay',
  'Portugal',
  'Spain',
  'Switzerland',
  'USA',
];

const STAGES = ['Group', 'R16', 'Quarterfinal', 'Semifinal', 'Final'];

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function App() {
  const [teams, setTeams] = useState(FALLBACK_TEAMS);
  const [teamA, setTeamA] = useState('France');
  const [teamB, setTeamB] = useState('Paraguay');
  const [stage, setStage] = useState('R16');
  const [isKnockout, setIsKnockout] = useState(true);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [teamsNotice, setTeamsNotice] = useState('');

  useEffect(() => {
    fetchTeams()
      .then((loadedTeams) => {
        const mergedTeams = [...new Set([...loadedTeams, ...FALLBACK_TEAMS])].sort();
        setTeams(mergedTeams);
        setTeamsNotice('');
      })
      .catch(() => {
        setTeamsNotice('Backend teams are unavailable. Showing local sample teams.');
      });
  }, []);

  const sameTeamSelected = teamA === teamB;

  const scoreRows = useMemo(() => {
    if (!prediction) {
      return [];
    }
    return prediction.most_likely_scores || [];
  }, [prediction]);

  async function handleSubmit(event) {
    event.preventDefault();

    if (sameTeamSelected) {
      setError('Choose two different teams.');
      return;
    }

    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const result = await predictMatch({ teamA, teamB, stage, isKnockout });
      setPrediction(result.prediction);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="app-header">
          <div>
            <p className="eyebrow">World Cup match intelligence</p>
            <h1>Atrox World Cup AI Predictor</h1>
          </div>
          <div className="status-pill">FastAPI backend</div>
        </header>

        <div className="layout">
          <form className="predictor-panel" onSubmit={handleSubmit}>
            <div className="field-grid">
              <label>
                <span>Team A</span>
                <select value={teamA} onChange={(event) => setTeamA(event.target.value)}>
                  {teams.map((team) => (
                    <option key={team} value={team}>
                      {team}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                <span>Team B</span>
                <select value={teamB} onChange={(event) => setTeamB(event.target.value)}>
                  {teams.map((team) => (
                    <option key={team} value={team}>
                      {team}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                <span>Stage</span>
                <select value={stage} onChange={(event) => setStage(event.target.value)}>
                  {STAGES.map((stageName) => (
                    <option key={stageName} value={stageName}>
                      {stageName}
                    </option>
                  ))}
                </select>
              </label>

              <label className="checkbox-row">
                <input
                  type="checkbox"
                  checked={isKnockout}
                  onChange={(event) => setIsKnockout(event.target.checked)}
                />
                <span>Knockout match</span>
              </label>
            </div>

            {teamsNotice && <p className="notice">{teamsNotice}</p>}
            {sameTeamSelected && <p className="notice">Select two different teams.</p>}

            <button type="submit" disabled={loading || sameTeamSelected}>
              {loading ? 'Predicting...' : 'Predict match'}
            </button>

            {error && <div className="error-box">{error}</div>}
          </form>

          <PredictionCard
            prediction={prediction}
            teamA={teamA}
            teamB={teamB}
            scoreRows={scoreRows}
            loading={loading}
          />
        </div>
      </section>
    </main>
  );
}

function PredictionCard({ prediction, teamA, teamB, scoreRows, loading }) {
  if (loading) {
    return (
      <section className="result-panel empty-state">
        <p>Running ensemble model...</p>
      </section>
    );
  }

  if (!prediction) {
    return (
      <section className="result-panel empty-state">
        <p>Select a matchup and run a prediction.</p>
      </section>
    );
  }

  return (
    <section className="result-panel">
      <div className="result-header">
        <div>
          <p className="eyebrow">Prediction</p>
          <h2>
            {teamA} vs {teamB}
          </h2>
        </div>
        <span className={`confidence ${prediction.confidence}`}>
          {prediction.confidence}
        </span>
      </div>

      <div className="probability-grid">
        <Metric label={`${teamA} 90-min win`} value={formatPercent(prediction.team_a_win)} />
        <Metric label="90-min draw" value={formatPercent(prediction.draw)} />
        <Metric label={`${teamB} 90-min win`} value={formatPercent(prediction.team_b_win)} />
        <Metric label={`${teamA} advance`} value={formatPercent(prediction.team_a_advance)} />
        <Metric label={`${teamB} advance`} value={formatPercent(prediction.team_b_advance)} />
      </div>

      <div className="detail-grid">
        <div>
          <h3>Expected goals</h3>
          <div className="xg-row">
            <span>{teamA}</span>
            <strong>{Number(prediction.expected_goals.team_a).toFixed(2)}</strong>
          </div>
          <div className="xg-row">
            <span>{teamB}</span>
            <strong>{Number(prediction.expected_goals.team_b).toFixed(2)}</strong>
          </div>
        </div>

        <div>
          <h3>Most likely exact scores</h3>
          <ol className="score-list">
            {scoreRows.slice(0, 5).map((scoreline) => (
              <li key={scoreline.score}>
                <span>
                  {teamA} {scoreline.score} {teamB}
                </span>
                <strong>{formatPercent(scoreline.probability)}</strong>
              </li>
            ))}
          </ol>
        </div>
      </div>

      <div className="explanations">
        <h3>Why this result</h3>
        <ul>
          {prediction.explanation_factors.map((factor) => (
            <li key={factor}>{factor}</li>
          ))}
        </ul>
      </div>

      {Boolean(prediction.data_notes?.length) && (
        <div className="data-notes">
          <h3>Data notes</h3>
          <ul>
            {prediction.data_notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export default App;
