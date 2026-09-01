import React, { useState } from 'react';
import { FixtureResponse, PredictionResponse } from '../api/types';
import { apiClient } from '../api/client';

interface FixtureCardProps {
  fixture: FixtureResponse;
}

export const FixtureCard: React.FC<FixtureCardProps> = ({ fixture }) => {
  // Format kickoff time for display
  const formatKickoffTime = (kickoffAt: string) => {
    const date = new Date(kickoffAt);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Call the prediction endpoint with empty features (as no features are required yet)
      const result = await apiClient.predictFixture(fixture.id, {});
      setPrediction(result);
    } catch (err) {
      console.error('Failed to get prediction:', err);
      setError('Failed to generate prediction. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(0)}%`;
  };

  return (
    <div className="fixture-card">
      <div className="fixture-header">
        <span className="matchday">Matchday {fixture.matchday}</span>
      </div>
      
      <div className="teams-container">
        <div className="team home-team">
          {fixture.home_team.name}
        </div>
        
        <div className="vs">VS</div>
        
        <div className="team away-team">
          {fixture.away_team.name}
        </div>
      </div>
      
      <div className="kickoff-container">
        <span className="kickoff-time">{formatKickoffTime(fixture.kickoff_at)}</span>
      </div>
      
      <button 
        className="predict-button"
        onClick={handlePredict}
        disabled={loading}
      >
        {loading ? 'Predicting...' : 'Predict'}
      </button>
      
      {error && (
        <div className="prediction-error">
          {error}
        </div>
      )}
      
      {prediction && (
        <div className="prediction-result">
          <div className="probability-row">
            <span>Home win:</span>
            <span>{formatPercentage(prediction.probabilities.home_win)}</span>
          </div>
          <div className="probability-row">
            <span>Draw:</span>
            <span>{formatPercentage(prediction.probabilities.draw)}</span>
          </div>
          <div className="probability-row">
            <span>Away win:</span>
            <span>{formatPercentage(prediction.probabilities.away_win)}</span>
          </div>
          <div className="predicted-outcome">
            Predicted: {prediction.predicted_outcome}
          </div>
        </div>
      )}
    </div>
  );
};

export default FixtureCard;