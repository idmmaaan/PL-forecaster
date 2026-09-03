import type { FixtureResponse, PredictionResponse } from './types'

/**
 * Relative by default so the Vite dev proxy handles it; override with
 * VITE_API_BASE_URL when the API is served from another origin.
 */
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

/** Error carrying the HTTP status and the API's `detail` message. */
export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function readError(response: Response): Promise<ApiError> {
  // FastAPI reports failures as {"detail": "..."}; fall back to the status text
  // when the body is empty or not JSON.
  let detail = response.statusText || `HTTP ${response.status}`
  try {
    const body = (await response.json()) as { detail?: unknown }
    if (typeof body.detail === 'string') {
      detail = body.detail
    }
  } catch {
    // Keep the status-derived message.
  }
  return new ApiError(response.status, detail)
}

export class EPLAPIClient {
  private readonly baseUrl: string

  constructor(baseUrl: string = BASE_URL) {
    this.baseUrl = baseUrl
  }

  /** GET /api/v1/fixtures */
  async getFixtures(): Promise<FixtureResponse[]> {
    const response = await fetch(`${this.baseUrl}/fixtures`)
    if (!response.ok) {
      throw await readError(response)
    }
    return (await response.json()) as FixtureResponse[]
  }

  /** GET /api/v1/fixtures/:id */
  async getFixture(fixtureId: number): Promise<FixtureResponse> {
    const response = await fetch(`${this.baseUrl}/fixtures/${fixtureId}`)
    if (!response.ok) {
      throw await readError(response)
    }
    return (await response.json()) as FixtureResponse
  }

  /** POST /api/v1/fixtures/:id/predict */
  async predictFixture(
    fixtureId: number,
    features: Record<string, unknown> = {},
  ): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseUrl}/fixtures/${fixtureId}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(features),
    })
    if (!response.ok) {
      throw await readError(response)
    }
    return (await response.json()) as PredictionResponse
  }

  /** GET /api/v1/predictions/:id */
  async getPrediction(predictionId: number): Promise<PredictionResponse> {
    const response = await fetch(`${this.baseUrl}/predictions/${predictionId}`)
    if (!response.ok) {
      throw await readError(response)
    }
    return (await response.json()) as PredictionResponse
  }
}

export const apiClient = new EPLAPIClient()
