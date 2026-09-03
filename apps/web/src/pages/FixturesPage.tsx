import { useQuery } from '@tanstack/react-query'

import { apiClient } from '../api/client'
import type { FixtureResponse } from '../api/types'
import { FixtureCard } from '../components'

export function FixturesPage() {
  const {
    data: fixtures,
    isPending,
    isError,
    error,
  } = useQuery<FixtureResponse[], Error>({
    queryKey: ['fixtures'],
    queryFn: () => apiClient.getFixtures(),
  })

  return (
    <div className="fixtures-page">
      <h1>Upcoming Premier League fixtures</h1>

      {isPending && <p className="status-message">Loading fixtures…</p>}

      {isError && (
        <p className="error-message" role="alert">
          Failed to load fixtures: {error.message}
        </p>
      )}

      {fixtures && fixtures.length === 0 && (
        <p className="status-message">No fixtures available.</p>
      )}

      {fixtures && fixtures.length > 0 && (
        <div className="fixtures-container">
          {fixtures.map((fixture) => (
            <FixtureCard key={fixture.id} fixture={fixture} />
          ))}
        </div>
      )}
    </div>
  )
}

export default FixturesPage
