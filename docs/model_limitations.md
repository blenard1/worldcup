# Model Limitations

World Cup AI Predictor is a probability system, not a certainty engine.

## Football Is Unpredictable

Football outcomes depend on injuries, red cards, tactical surprises, weather, refereeing decisions, penalties, player fatigue, and randomness. A strong model can improve probability estimates, but it cannot remove uncertainty from a low-scoring sport.

## Small Sample Data Can Mislead

The included CSV files are realistic sample data for development. They are not official datasets and are too small to produce reliable real-world forecasts. Small World Cup samples can exaggerate streaks, one-off performances, and matchup narratives.

For production use, replace the sample files with larger historical match data, reliable current tournament feeds, player availability, market odds, event data, and validated team-strength sources.

## H2H Should Not Dominate

Head-to-head history is noisy. Matches may be years apart, played under different coaches, and affected by very different squads. The H2H engine is intentionally weighted lightly in the ensemble so it can nudge predictions but not override stronger current and structural signals.

## Tactical Scores Are Currently Manual

The tactical engine uses numeric scores from 1 to 10. In this project version, those scores are manually supplied in `data/sample/team_tactics.csv`.

Until real event data or scouting feeds are integrated, tactical scores should be treated as approximate inputs. They are useful for pipeline design, not as authoritative measurements.

## Predictions Are Not Guarantees

The system outputs probabilities:

- A 60% win probability still loses 40% of the time across similar matches.
- A likely scoreline is the single most probable score, not a forecast that should be expected to happen.
- Knockout advance probabilities include uncertainty from draws, extra time, and penalties.

Use the model as decision support, not as a deterministic answer.
