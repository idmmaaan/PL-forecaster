import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { apiClient, ApiError } from '../api/client'
import type { FixtureResponse, PredictionResponse } from '../api/types'
import { renderWithQuery } from '../test/renderWithQuery'
import { FixtureCard } from './FixtureCard'

const fixture: FixtureResponse = {
  id: 14621,
  competition_code: 'PL',
  season_start_year: 2026,
  matchday: 4,
  kickoff_at: '2026-09-12T14:00:00Z',
  status: 'SCHEDULED',
  home_team: { id: 1, name: 'Arsenal', crest_url: null },
  away_team: { id: 2, name: 'Chelsea', crest_url: null },
}

const prediction: PredictionResponse = {
  prediction_id: 1042,
  fixture_id: 14621,
  home_team: 'Arsenal',
  away_team: 'Chelsea',
  predicted_outcome: 'HOME_WIN',
  probabilities: { home_win: 0.51, draw: 0.27, away_win: 0.22 },
  model_name: 'dummy-epl',
  model_version: '0.1.0',
  feature_schema_version: '1.0.0',
  created_at: '2026-09-03T15:00:00Z',
}

describe('FixtureCard', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('renders both team names and a Predict button', () => {
    renderWithQuery(<FixtureCard fixture={fixture} />)

    expect(screen.getByText('Arsenal')).toBeInTheDocument()
    expect(screen.getByText('Chelsea')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Predict' })).toBeInTheDocument()
  })

  it('shows all three probabilities as percentages after predicting', async () => {
    vi.spyOn(apiClient, 'predictFixture').mockResolvedValue(prediction)
    renderWithQuery(<FixtureCard fixture={fixture} />)

    await userEvent.click(screen.getByRole('button', { name: 'Predict' }))

    await waitFor(() => expect(screen.getByText('51%')).toBeInTheDocument())
    expect(screen.getByText('27%')).toBeInTheDocument()
    expect(screen.getByText('22%')).toBeInTheDocument()
    expect(screen.getByText(/Most likely/)).toHaveTextContent('Home win')
    expect(apiClient.predictFixture).toHaveBeenCalledWith(14621)
  })

  it('surfaces the API error message instead of a prediction', async () => {
    vi.spyOn(apiClient, 'predictFixture').mockRejectedValue(
      new ApiError(503, 'Prediction model unavailable'),
    )
    renderWithQuery(<FixtureCard fixture={fixture} />)

    await userEvent.click(screen.getByRole('button', { name: 'Predict' }))

    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent('Prediction model unavailable'),
    )
    expect(screen.queryByText(/Most likely/)).not.toBeInTheDocument()
  })
})
