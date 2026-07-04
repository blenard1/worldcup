const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

export async function fetchTeams() {
  const response = await fetch(`${API_BASE_URL}/teams`);

  if (!response.ok) {
    throw new Error('Unable to load teams from the backend.');
  }

  const payload = await response.json();
  return payload.teams || [];
}

export async function predictMatch({ teamA, teamB, stage, isKnockout }) {
  const response = await fetch(`${API_BASE_URL}/predict-match`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      team_a: teamA,
      team_b: teamB,
      stage,
      is_knockout: isKnockout,
    }),
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.detail || 'Prediction request failed.');
  }

  return payload;
}
