import { FixtureResponse, PredictionResponse } from './types';

// Base URL - in a real app this would come from environment variables
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export class EPLAPIClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = BASE_URL;
  }

  // Get all fixtures
  async getFixtures(): Promise<FixtureResponse[]> {
    const response = await fetch(`${this.baseUrl}/fixtures`);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }

  // Generate prediction for a fixture
  async predictFixture(fixtureId: number, features?: Record<string, any>): Promise<PredictionResponse> {
    const url = `${this.baseUrl}/fixtures/${fixtureId}/predict`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(features || {}),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.json();
  }
}

// Create a singleton instance
export const apiClient = new EPLAPIClient();