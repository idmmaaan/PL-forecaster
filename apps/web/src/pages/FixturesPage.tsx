import React, { useState, useEffect } from 'react';
import { FixtureCard } from '../components';
import { FixtureResponse } from '../api/types';
import { apiClient } from '../api/client';

export const FixturesPage: React.FC = () => {
  const [fixtures, setFixtures] = useState<FixtureResponse[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchFixtures = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await apiClient.getFixtures();
        setFixtures(data);
      } catch (err) {
        console.error('Failed to fetch fixtures:', err);
        setError('Failed to load fixtures. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchFixtures();
  }, []);

  if (loading) {
    return (
      <div className="fixtures-page">
        <h1>Upcoming Fixtures</h1>
        <p>Loading fixtures...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fixtures-page">
        <h1>Upcoming Fixtures</h1>
        <div className="error-message">
          {error}
        </div>
      </div>
    );
  }

  if (fixtures.length === 0) {
    return (
      <div className="fixtures-page">
        <h1>Upcoming Fixtures</h1>
        <p>No fixtures available.</p>
      </div>
    );
  }

  return (
    <div className="fixtures-page">
      <h1>Upcoming Fixtures</h1>
      <div className="fixtures-container">
        {fixtures.map(fixture => (
          <FixtureCard 
            key={fixture.id} 
            fixture={fixture} 
          />
        ))}
      </div>
    </div>
  );
};

export default FixturesPage;