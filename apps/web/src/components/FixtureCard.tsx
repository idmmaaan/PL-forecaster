import { useMutation } from '@tanstack/react-query'

import { apiClient } from '../api/client'
import type { FixtureResponse, PredictionResponse } from '../api/types'
import { PredictionPanel } from './PredictionPanel'

interface FixtureCardProps {
  fixture: FixtureResponse
}

function formatKickoffTime(kickoffAt: string): string {
  return new Date(kickoffAt).toLocaleString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

export function FixtureCard({ fixture }: FixtureCardProps) {
  const prediction = useMutation<PredictionResponse, Error>({
    mutationFn: () => apiClient.predictFixture(fixture.id),
  })

  return (
    <article className="fixture-card">
      <header className="fixture-header">
        <span className="matchday">Matchday {fixture.matchday}</span>
        <span className="kickoff-time">{formatKickoffTime(fixture.kickoff_at)}</span>
      </header>

      <div className="teams-container">
        <span className="team home-team">{fixture.home_team.name}</span>
        <span className="vs">v</span>
        <span className="team away-team">{fixture.away_team.name}</span>
      </div>

      <button
        type="button"
        className="predict-button"
        onClick={() => prediction.mutate()}
        disabled={prediction.isPending}
      >
        {prediction.isPending ? 'Predicting…' : 'Predict'}
      </button>

      {prediction.isError && (
        <p className="prediction-error" role="alert">
          {prediction.error.message}
        </p>
      )}

      {prediction.data && <PredictionPanel prediction={prediction.data} />}
    </article>
  )
}

export default FixtureCard
