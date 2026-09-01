// Team types
export interface TeamSummary {
  id: number;
  name: string;
  crest_url?: string | null;
}

// Fixture types
export interface FixtureResponse {
  id: number;
  competition_code: string;
  season_start_year: number;
  matchday: number;
  kickoff_at: string; // ISO format date string
  status: string;
  home_team: TeamSummary;
  away_team: TeamSummary;
}

// Prediction types
export type Outcome = 'HOME_WIN' | 'DRAW' | 'AWAY_WIN';

export interface Probabilities {
  home_win: number;
  draw: number;
  away_win: number;
}

export interface PredictionResponse {
  prediction_id: number;
  fixture_id: number;
  home_team: string;
  away_team: string;
  predicted_outcome: Outcome;
  probabilities: Probabilities;
  model_name: string;
  model_version: string;
  feature_schema_version: string;
  created_at: string; // ISO format date string
}