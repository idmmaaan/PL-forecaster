import { screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient } from '../api/client'
import type { FixtureResponse } from '../api/types'
import { renderWithQuery } from '../test/renderWithQuery'
import { FixturesPage } from './FixturesPage'

const fixtures: FixtureResponse[] = [
  {
    id: 14621,
    competition_code: 'PL',
    season_start_year: 2026,
    matchday: 4,
    kickoff_at: '2026-09-12T14:00:00Z',
    status: 'SCHEDULED',
    home_team: { id: 1, name: 'Arsenal', crest_url: null },
    away_team: { id: 2, name: 'Chelsea', crest_url: null },
  },
  {
    id: 14622,
    competition_code: 'PL',
    season_start_year: 2026,
    matchday: 4,
    kickoff_at: '2026-09-12T16:30:00Z',
    status: 'SCHEDULED',
    home_team: { id: 3, name: 'Liverpool', crest_url: null },
    away_team: { id: 4, name: 'Manchester City', crest_url: null },
  },
]

describe('FixturesPage', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('renders one card per fixture returned by the API', async () => {
    vi.spyOn(apiClient, 'getFixtures').mockResolvedValue(fixtures)
    renderWithQuery(<FixturesPage />)

    await waitFor(() => expect(screen.getByText('Arsenal')).toBeInTheDocument())
    expect(screen.getByText('Manchester City')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: 'Predict' })).toHaveLength(2)
  })

  it('reports a load failure rather than rendering an empty list silently', async () => {
    vi.spyOn(apiClient, 'getFixtures').mockRejectedValue(new Error('Service Unavailable'))
    renderWithQuery(<FixturesPage />)

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent('Failed to load fixtures'),
    )
  })

  it('shows an empty state when there are no fixtures', async () => {
    vi.spyOn(apiClient, 'getFixtures').mockResolvedValue([])
    renderWithQuery(<FixturesPage />)

    await waitFor(() => expect(screen.getByText('No fixtures available.')).toBeInTheDocument())
  })
})
