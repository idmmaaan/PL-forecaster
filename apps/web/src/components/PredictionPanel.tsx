import type { PredictionResponse } from '../api/types'

interface PredictionPanelProps {
  prediction: PredictionResponse
}

const OUTCOME_LABELS: Record<PredictionResponse['predicted_outcome'], string> = {
  HOME_WIN: 'Home win',
  DRAW: 'Draw',
  AWAY_WIN: 'Away win',
}

function formatPercentage(value: number): string {
  return `${Math.round(value * 100)}%`
}

/** Shows all three outcome probabilities plus the model that produced them. */
export function PredictionPanel({ prediction }: PredictionPanelProps) {
  const { probabilities, predicted_outcome } = prediction

  const rows = [
    { key: 'HOME_WIN' as const, label: 'Home win', value: probabilities.home_win },
    { key: 'DRAW' as const, label: 'Draw', value: probabilities.draw },
    { key: 'AWAY_WIN' as const, label: 'Away win', value: probabilities.away_win },
  ]

  return (
    <div className="prediction-result">
      <ul className="probability-list">
        {rows.map((row) => (
          <li
            key={row.key}
            className={
              row.key === predicted_outcome ? 'probability-row is-predicted' : 'probability-row'
            }
          >
            <span className="probability-label">{row.label}</span>
            <span className="probability-bar" aria-hidden="true">
              <span className="probability-bar-fill" style={{ width: formatPercentage(row.value) }} />
            </span>
            <span className="probability-value">{formatPercentage(row.value)}</span>
          </li>
        ))}
      </ul>

      <p className="predicted-outcome">
        Most likely: <strong>{OUTCOME_LABELS[predicted_outcome]}</strong>
      </p>

      <p className="prediction-meta">
        Model {prediction.model_name} {prediction.model_version} &middot; features{' '}
        {prediction.feature_schema_version}
      </p>
    </div>
  )
}

export default PredictionPanel
