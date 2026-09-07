# EPL AI Match Predictor

A local-first machine-learning project that predicts English Premier League match outcomes as three outcome probabilities:

- `HOME_WIN`
- `DRAW`
- `AWAY_WIN`

The first product milestone is a web page that lists upcoming Premier League fixtures and provides a **Predict** button for each fixture. The prediction is produced by a dedicated tabular machine-learning model trained on historical match data and pre-match statistics.

> **Want to run it?** See **[docs/local-development.md](docs/local-development.md)** for
> setup, the commands, and the current known gaps.
>
> **Want to understand it, or learn ML from it?** See
> **[docs/how-it-works.md](docs/how-it-works.md)** — how the pipeline works, what
> Hugging Face and transformers actually do here, and an eight-stage learning path
> through the codebase.
>
> The [Local development setup](#local-development-setup) section below describes
> building the project from scratch instead.

> Status: Milestones 1-6 implemented — repaired scaffold, full data model, real
> fixture import, historical ingestion, leakage-safe v1 features, evaluation
> harness with walk-forward backtesting, the four mandatory baselines with
> calibration comparison, a trained CatBoost model served through the API, and
> adapters plus a selection gate for the five Hugging Face candidates.
> Outstanding: the model registry and promotion CLI (Milestone 7), batch
> retraining (Milestone 8), and the complete five-way candidate benchmark.
>
> Last reviewed: 2026-09-03

## Table of contents

1. [Project goal](#project-goal)
2. [Critical architecture decision: Qwen is the implementation assistant](#critical-architecture-decision-qwen-is-the-implementation-assistant)
3. [First milestone](#first-milestone)
4. [System architecture](#system-architecture)
5. [Why the predictor is not an LLM](#why-the-predictor-is-not-an-llm)
6. [Prediction contract](#prediction-contract)
7. [Hugging Face model investigation](#hugging-face-model-investigation)
8. [Recommended model-selection strategy](#recommended-model-selection-strategy)
9. [Data strategy](#data-strategy)
10. [Feature engineering](#feature-engineering)
11. [Data-leakage rules](#data-leakage-rules)
12. [Training and evaluation](#training-and-evaluation)
13. [Model lifecycle and continual learning](#model-lifecycle-and-continual-learning)
14. [Application stack](#application-stack)
15. [Repository structure](#repository-structure)
16. [Local development setup](#local-development-setup)
17. [Database design](#database-design)
18. [API design](#api-design)
19. [Implementation milestones](#implementation-milestones)
20. [Qwen development workflow](#qwen-development-workflow)
21. [Acceptance criteria](#acceptance-criteria)
22. [Risks and limitations](#risks-and-limitations)
23. [Documentation plan](#documentation-plan)
24. [Next step](#next-step)
25. [References](#references)

---

## Project goal

Build a real local AI/ML application that:

1. Imports upcoming English Premier League fixtures.
2. Displays fixtures in a web interface.
3. Builds a pre-match feature snapshot for a selected fixture.
4. Loads an active, versioned prediction model.
5. Returns probabilities for home win, draw, and away win.
6. Stores the prediction, model version, and exact feature snapshot.
7. Retrains candidate models in a controlled batch pipeline as new results become available.
8. Promotes a new model only when it outperforms the active model on chronological evaluation data.

This project is designed as a learning and engineering project, not as betting advice. Predictions are probabilistic and can be wrong.

---

## Critical architecture decision: Qwen is the implementation assistant

Qwen is **not** part of the deployed football-prediction runtime.

Qwen is the local coding model that helps implement the repository. It may be asked to:

- create Python, SQL, YAML, Markdown, and TypeScript files;
- implement FastAPI endpoints;
- implement React components;
- create database migrations;
- build data-import scripts;
- implement feature engineering;
- add model adapters;
- run tests and fix failures;
- update documentation;
- review diffs and suggest refactoring.

Qwen must not:

- predict football outcomes;
- generate or modify match probabilities;
- run as an application worker;
- become a production API dependency;
- fabricate data, metrics, or test results;
- promote a model without evaluation evidence.

### Development-time architecture

```text
You
 |
 | implementation prompt
 v
Local Qwen coding model
 |
 +-- reads README.md and project instructions
 +-- creates or edits repository files
 +-- runs local commands and tests
 +-- reports changed files and validation results
 |
 v
Git repository + local development environment
```

### Runtime architecture

```text
React web application
        |
        | HTTP/JSON
        v
FastAPI application
        |
        +--> PostgreSQL
        |      +-- teams
        |      +-- fixtures
        |      +-- feature snapshots
        |      +-- model versions
        |      +-- predictions
        |
        +--> Feature builder
        |
        +--> Active predictor adapter
               |
               +-- HOME_WIN probability
               +-- DRAW probability
               +-- AWAY_WIN probability
```

There is intentionally no `qwen-worker`, no LLM explanation service, and no Ollama dependency in the deployed application.

---

## First milestone

The first milestone is a complete vertical slice:

```text
Upcoming fixtures
       |
       v
Fixture cards on website
       |
       v
[ Predict ] button
       |
       v
FastAPI prediction endpoint
       |
       v
Feature snapshot
       |
       v
Active predictor
       |
       v
Home win / draw / away win probabilities
```

Example UI result:

```text
Arsenal vs Chelsea

Home win: 51%
Draw:     27%
Away win: 22%

Model: tabicl-epl-0.1.0
```

The application must first work with a deterministic stub predictor. The real model is connected only after the frontend, API, database, and response contract are verified end to end.

---

## System architecture

### Online prediction flow

```text
1. User opens the fixtures page.
2. Web application calls GET /api/v1/fixtures.
3. User clicks Predict for one fixture.
4. Web application calls POST /api/v1/fixtures/{fixture_id}/predict.
5. FastAPI loads fixture and team data.
6. Feature service calculates values using data available before kickoff.
7. Feature snapshot is saved.
8. Active model version is loaded through a predictor adapter.
9. Predictor returns three probabilities.
10. Prediction is validated and saved.
11. API returns the prediction to the web application.
```

### Offline training flow

```text
Historical EPL files
        |
        v
Raw-data validation
        |
        v
Canonical match table
        |
        v
Chronological feature generation
        |
        v
Train / validation / final test periods
        |
        v
Baselines and Hugging Face candidates
        |
        v
Probability calibration and evaluation
        |
        v
Versioned candidate artifact
        |
        v
Promote or reject
```

### Important separation

Training and inference are separate concerns:

- **Training pipeline:** creates candidate artifacts and evaluation reports.
- **Inference application:** loads one approved artifact and predicts fixtures.
- **Qwen:** helps write and modify the code for both areas but is not executed by either pipeline.

---

## Why the predictor is not an LLM

The available football information is primarily structured tabular data:

```text
home_team
away_team
home_form_points
away_form_points
home_goals_for
away_goals_for
elo_difference
rest_days_difference
league_position_difference
...
```

The target is a three-class supervised-learning problem:

```text
features -> HOME_WIN | DRAW | AWAY_WIN
```

A text LLM is not the natural first choice for this task. A tabular classifier is more suitable because it provides:

- reproducible numerical output;
- direct multiclass probability estimation;
- controlled training and validation;
- standard metrics such as log loss and Brier score;
- simpler latency and resource requirements;
- clearer comparison against statistical baselines.

Hugging Face is still useful. It hosts pretrained tabular foundation models and can later serve as the registry for a custom PyTorch model or trained checkpoint.

---

## Prediction contract

### Labels

Historical source labels are commonly mapped as:

| Source label | Application label | Meaning |
|---|---|---|
| `H` | `HOME_WIN` | Home team wins |
| `D` | `DRAW` | Match ends in a draw |
| `A` | `AWAY_WIN` | Away team wins |

### Required output

```json
{
  "prediction_id": 1042,
  "fixture_id": 14621,
  "home_team": "Arsenal",
  "away_team": "Chelsea",
  "predicted_outcome": "HOME_WIN",
  "probabilities": {
    "home_win": 0.51,
    "draw": 0.27,
    "away_win": 0.22
  },
  "model_name": "tabicl-epl",
  "model_version": "0.1.0",
  "feature_schema_version": "1.0.0",
  "created_at": "2026-08-31T15:00:00Z"
}
```

### Output invariants

- Every probability must be between `0.0` and `1.0`.
- The three probabilities must sum to approximately `1.0`.
- `predicted_outcome` must be the class with the largest probability.
- Every prediction must reference a model version.
- Every prediction must reference the exact feature snapshot used.
- The API must never silently replace invalid probabilities with invented values.

---

## Hugging Face model investigation

The following list is a shortlist to benchmark, not a guarantee that the first model will win. All candidates must use the same data, feature schema, chronological splits, and metrics.

### Summary ranking

| Priority | Model | Hugging Face repository | Adaptation method | License summary | Proposed role |
|---:|---|---|---|---|---|
| 1 | TabICLv2 | `jingang/TabICL` | In-context use and real fine-tuning | BSD-3-Clause | Primary candidate |
| 2 | Mitra Classifier | `autogluon/mitra-classifier` | AutoGluon fine-tuning | Apache-2.0 | Main challenger |
| 3 | TabSTAR | `alana89/TabSTAR` | Dataset fitting / transfer learning | Model card: CC-BY-4.0; repository code: MIT | Semantic-feature challenger |
| 4 | TabPFNMix Classifier | `autogluon/tabpfn-mix-1.0-classifier` | AutoGluon training/fine-tuning | Apache-2.0 | Lightweight challenger |
| 5 | TabPFN-3 | `Prior-Labs/tabpfn_3` | In-context prediction and demanding fine-tuning | Non-commercial model license | Research benchmark only |

Licenses and model cards must be rechecked before any public or commercial deployment.

### 1. TabICLv2 - recommended primary candidate

Repository: `jingang/TabICL`

Why it is first:

- created specifically for tabular classification and regression;
- provides a scikit-learn-style API;
- includes `FinetunedTabICLClassifier` for changing pretrained weights on one dataset;
- supports probability prediction;
- creates a reusable fine-tuned checkpoint;
- uses a permissive BSD-3-Clause license;
- current implementation can automatically select CUDA, XPU, MPS, or CPU for normal use.

Cautions:

- a CUDA GPU is still recommended for larger fine-tuning jobs;
- Apple Silicon compatibility must be tested with the actual dataset and package version;
- the EPL dataset may be small enough that a simpler model still wins.

Provisional role:

```text
Primary pretrained candidate
        +
EPL fine-tuning experiment
        +
Chronological evaluation
```

### 2. Mitra Classifier - main challenger

Repository: `autogluon/mitra-classifier`

Why it is useful:

- pretrained tabular Transformer;
- supports multiclass classification;
- exposes `predict_proba()` through AutoGluon;
- documents both non-fine-tuned and fine-tuned modes;
- `fine_tune=True` and `fine_tune_steps` provide a practical adaptation path;
- approximately 72 million parameters;
- Apache-2.0 license.

Cautions:

- GPU use is strongly preferable for practical fine-tuning speed;
- AutoGluon adds a large dependency surface;
- its local Apple Silicon behavior must be measured rather than assumed.

Provisional role: benchmark directly against TabICLv2 using the same training and test tables.

### 3. TabSTAR - semantic-feature challenger

Repository: `alana89/TabSTAR`

Why it is useful:

- designed for tabular data with meaningful names, categories, and text fields;
- supports classification and regression;
- provides a simple `fit`, `predict`, and save/load workflow;
- approximately 47.3 million parameters;
- may become more valuable when the project adds manager names, venue data, status text, or other semantic columns.

Cautions:

- the first feature set will be mostly numerical, so TabSTAR's semantic advantage may be limited;
- the Hugging Face model card lists CC-BY-4.0, while the linked source repository describes its code as MIT; model-weight and code licenses must be treated separately and verified;
- hardware compatibility must be validated in a local spike.

Provisional role: third candidate, especially after richer categorical or textual features are introduced.

### 4. TabPFNMix Classifier - lightweight challenger

Repository: `autogluon/tabpfn-mix-1.0-classifier`

Why it is useful:

- tabular foundation model based on a 12-layer encoder-decoder Transformer;
- approximately 39 million parameters;
- official model card shows AutoGluon training with configurable epochs;
- supports multiclass probabilities through the AutoGluon predictor interface;
- Apache-2.0 license.

Cautions:

- older and less preferred than the first two candidates;
- AutoGluon dependency and runtime overhead still apply;
- must prove that it beats Logistic Regression and CatBoost.

Provisional role: smaller experimental challenger.

### 5. TabPFN-3 - research benchmark only

Repository: `Prior-Labs/tabpfn_3`

Why it is interesting:

- modern tabular foundation model;
- provides a multiclass-specialized checkpoint;
- supports structured classification and probability output;
- can serve as a strong research comparison.

Why it is not the first production choice:

- current fine-tuning example recommends a CUDA GPU with 80 GB VRAM;
- the supplied fine-tuning example rejects non-CUDA execution;
- model weights use a non-commercial license;
- the license restricts commercial and production use without a separate agreement.

Provisional role: research-only benchmark, preferably in normal inference mode rather than laptop fine-tuning.

### Mandatory non-foundation baselines

Every Hugging Face candidate must be compared against:

1. Majority-class dummy classifier.
2. Historical class-frequency predictor.
3. Multinomial Logistic Regression.
4. CatBoost multiclass classifier.

Optional additional baselines:

- LightGBM;
- XGBoost;
- Random Forest;
- a custom PyTorch multilayer perceptron with team embeddings.

A tabular foundation model is selected only when it provides a measurable benefit. Novelty is not an acceptance criterion.

---

## Recommended model-selection strategy

### Current provisional decision

```text
Primary candidate:       TabICLv2
Main challenger:         Mitra Classifier
Semantic challenger:     TabSTAR
Lightweight challenger:  TabPFNMix
Research benchmark:      TabPFN-3
Required baseline:       CatBoost
```

### Common predictor interface

The application must not depend directly on one model library. Qwen should implement a common interface and one adapter per candidate.

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Predictor(ABC):
    @abstractmethod
    def fit(self, train_data: Any, validation_data: Any) -> None:
        """Train or adapt a candidate model."""

    @abstractmethod
    def predict_proba(self, features: dict[str, object]) -> dict[str, float]:
        """Return home_win, draw, and away_win probabilities."""

    @abstractmethod
    def save(self, artifact_path: Path) -> None:
        """Save a complete, reloadable artifact."""

    @classmethod
    @abstractmethod
    def load(cls, artifact_path: Path) -> "Predictor":
        """Load a previously saved artifact."""
```

Suggested adapters:

```text
ml/src/epl_predictor/predictors/
├── base.py
├── dummy.py
├── logistic_regression.py
├── catboost.py
├── tabicl.py
├── mitra.py
├── tabstar.py
├── tabpfn_mix.py
└── tabpfn3.py
```

### Selection gate

A candidate may be promoted only when it:

- beats the active model on primary chronological test metrics;
- does not materially damage draw-class performance;
- produces valid and reasonably calibrated probabilities;
- passes feature-schema compatibility tests;
- passes artifact save/load tests;
- meets local inference-latency and memory requirements;
- has a license compatible with the intended use.

---

## Data strategy

Use separate sources for historical training and current fixtures.

### Historical match data

Initial source: Football-Data.co.uk Premier League CSV files.

Useful historical columns commonly include:

```text
Date
HomeTeam
AwayTeam
FTHG   full-time home goals
FTAG   full-time away goals
FTR    full-time result: H, D, or A
HS     home shots
AS     away shots
HST    home shots on target
AST    away shots on target
HC     home corners
AC     away corners
```

Not every season contains exactly the same schema. Import code must:

- validate expected columns per season;
- preserve raw files unchanged;
- normalize team names through a mapping table;
- record source URL, season, checksum, and import time;
- reject duplicate matches;
- report missing or malformed values;
- avoid silently filling critical fields.

### Current fixtures

Initial source: football-data.org API v4.

Premier League competition code:

```text
PL
```

Relevant endpoints include:

```text
GET /v4/competitions/PL
GET /v4/competitions/PL/matches
GET /v4/competitions/PL/standings
GET /v4/competitions/PL/teams
```

The fixture importer must respect the provider's current authentication, plan, terms, and rate limits.

### Initial chronological split

A strong first experiment is:

```text
Training:       2010/11 through 2023/24
Validation:     2024/25
Final test:     2025/26
Live shadow:    2026/27
```

This split may be adjusted after data quality is measured, but the final test season must remain untouched during feature selection and model tuning.

### Betting-odds policy

Football-Data.co.uk files may contain bookmaker odds. Do not include them in the primary model at first.

Reasons:

- odds already encode a strong external market prediction;
- they can hide whether the model learned football signals;
- availability and timing may vary;
- including them can make the project closer to market imitation than match modeling.

An odds-aware model may be created later as a separate benchmark with an explicit feature schema and clear labeling.

---

## Feature engineering

### Version 1 feature set

Use values that are known before kickoff:

```text
home_team
away_team
season_start_year
matchday

home_points_last_5
away_points_last_5

home_goals_for_last_5
away_goals_for_last_5

home_goals_against_last_5
away_goals_against_last_5

home_goal_difference_last_5
away_goal_difference_last_5

home_home_points_last_5
away_away_points_last_5

home_shots_on_target_last_5
away_shots_on_target_last_5

home_clean_sheet_rate_last_5
away_clean_sheet_rate_last_5

home_elo
away_elo
elo_difference

home_rest_days
away_rest_days
rest_days_difference

home_league_position_before_match
away_league_position_before_match
league_position_difference
```

### Later feature candidates

Add only after the first model is stable:

- expected goals;
- lineup strength;
- confirmed injuries and suspensions;
- manager changes;
- schedule congestion;
- European competition fatigue;
- promoted-team indicator;
- venue or travel variables;
- weather;
- market odds as a separate experiment.

### Feature schema versioning

Every feature definition must have a version, for example:

```text
feature_schema_version = 1.0.0
```

Changing a rolling window, formula, null policy, team encoder, or Elo implementation requires a new version.

---

## Data-leakage rules

Data leakage is the largest technical risk in this project.

For each match, features must be calculated before the current result updates team state.

Correct order:

```python
for match in matches_sorted_by_kickoff:
    features = build_features_from_prior_state(match)
    save_training_row(features=features, target=match.result)
    update_team_state_with_completed_match(match)
```

Incorrect order:

```python
for match in matches_sorted_by_kickoff:
    update_team_state_with_completed_match(match)
    features = build_features_from_prior_state(match)
```

The incorrect version uses information from the match being predicted.

### Forbidden current-match inputs

Never use these values for the fixture being predicted:

- final score;
- current-match goals;
- current-match shots;
- current-match shots on target;
- current-match corners;
- current-match cards;
- statistics recorded after kickoff;
- final season league position;
- any future match result.

### Split rules

Do not use a random row-level train/test split as the main evaluation method.

Use:

- complete later seasons as holdout periods;
- rolling-origin or walk-forward backtests;
- time-aware validation;
- feature generation that respects match chronology.

---

## Training and evaluation

### Primary metric

Use multiclass log loss as the primary model-selection metric.

It evaluates the quality of all three probabilities and penalizes confident wrong predictions more strongly than uncertain wrong predictions.

### Additional metrics

Track at least:

- multiclass log loss;
- multiclass Brier score;
- accuracy;
- per-class precision;
- per-class recall;
- per-class F1 score;
- confusion matrix;
- probability calibration error;
- draw recall and draw precision;
- inference latency;
- peak memory usage;
- model artifact size;
- training duration.

### Why draw performance is explicit

A model can obtain acceptable-looking accuracy while rarely identifying draws. Draw performance must therefore be separately reported and included in promotion rules.

### Evaluation sequence

```text
1. Build one immutable processed dataset.
2. Freeze one feature schema.
3. Freeze chronological train, validation, and test periods.
4. Train every baseline and candidate on the same training rows.
5. Tune only against the validation period.
6. Evaluate once against the final test period.
7. Generate a comparison report.
8. Select a model based on metrics, calibration, resources, and license.
```

### Calibration

Even a classifier with good ranking performance may produce poor probabilities. The project should evaluate calibration plots and optionally compare:

- uncalibrated output;
- temperature scaling;
- isotonic or sigmoid calibration where appropriate;
- calibration fitted only on chronological validation data.

The calibration method and its fitted parameters are part of the versioned model artifact.

---

## Model lifecycle and continual learning

Do not update the active model after every match.

Use controlled batch retraining:

```text
Completed matches imported
        |
        v
Data checks pass
        |
        v
Feature table rebuilt
        |
        v
Candidate trained
        |
        v
Chronological backtest
        |
        v
Candidate compared with active model
        |
        +--> reject
        |
        +--> promote
```

### Model statuses

```text
CANDIDATE
ACTIVE
REJECTED
ARCHIVED
```

### Version examples

```text
logreg-epl-0.1.0
catboost-epl-0.1.0
tabicl-epl-0.1.0
mitra-epl-0.1.0
tabstar-epl-0.1.0
```

### Artifact contents

A complete model artifact should contain:

```text
model weights or serialized estimator
model configuration
preprocessing pipeline
class-order mapping
feature list
feature schema version
training date range
validation date range
test date range
metrics
calibration object
library versions
random seeds
source commit hash
model card
```

Never overwrite the previous active artifact. Promotion updates an `ACTIVE` pointer, allowing rollback without retraining.

### Hugging Face Hub usage

Hugging Face can be used as a registry after a local model is validated. It supports arbitrary model files, not only Transformers models. A custom PyTorch `nn.Module` can use `PyTorchModelHubMixin` for `save_pretrained`, `from_pretrained`, and `push_to_hub` behavior.

Initial development should remain local. Uploading is a later milestone after data licensing, repository visibility, secrets, and model licensing are reviewed.

---

## Application stack

| Area | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Pydantic |
| Database | PostgreSQL |
| ORM and migrations | SQLAlchemy, Alembic |
| Frontend | React, TypeScript, Vite |
| Data processing | pandas, NumPy |
| Baselines | scikit-learn, CatBoost |
| Foundation-model candidates | TabICL, AutoGluon/Mitra, TabSTAR, TabPFNMix, TabPFN |
| Model artifacts | Local versioned directories first; Hugging Face Hub later |
| Python package manager | `uv` |
| Local infrastructure | Docker Compose |
| Tests | pytest, HTTPX; frontend test tools added when UI begins |
| Code quality | Ruff, mypy, TypeScript compiler |
| Implementation assistant | Local Qwen coding model, outside runtime |

Redis is not required for the first milestone.

---

## Repository structure

```text
epl-ai-predictor/
|
├── apps/
│   ├── api/
│   │   ├── app/
│   │   │   ├── api/
│   │   │   │   ├── fixtures.py
│   │   │   │   └── predictions.py
│   │   │   ├── core/
│   │   │   │   ├── config.py
│   │   │   │   └── database.py
│   │   │   ├── db/
│   │   │   ├── models/
│   │   │   ├── repositories/
│   │   │   ├── schemas/
│   │   │   ├── services/
│   │   │   │   ├── fixture_service.py
│   │   │   │   ├── feature_service.py
│   │   │   │   └── predictor_service.py
│   │   │   └── main.py
│   │   ├── migrations/
│   │   ├── tests/
│   │   └── pyproject.toml
│   |
│   └── web/
│       ├── src/
│       │   ├── api/
│       │   ├── components/
│       │   │   ├── FixtureCard.tsx
│       │   │   └── PredictionPanel.tsx
│       │   ├── pages/
│       │   └── App.tsx
│       └── package.json
|
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   ├── interim/
│   │   ├── processed/
│   │   └── README.md
│   ├── artifacts/
│   ├── reports/
│   ├── notebooks/
│   ├── src/epl_predictor/
│   │   ├── data/
│   │   │   ├── ingest.py
│   │   │   └── validation.py
│   │   ├── features/
│   │   │   ├── builder.py
│   │   │   ├── elo.py
│   │   │   └── state.py
│   │   ├── predictors/
│   │   │   ├── base.py
│   │   │   ├── dummy.py
│   │   │   ├── logistic_regression.py
│   │   │   ├── catboost.py
│   │   │   ├── tabicl.py
│   │   │   ├── mitra.py
│   │   │   ├── tabstar.py
│   │   │   ├── tabpfn_mix.py
│   │   │   └── tabpfn3.py
│   │   ├── evaluation/
│   │   │   ├── metrics.py
│   │   │   ├── backtest.py
│   │   │   └── report.py
│   │   ├── training/
│   │   │   ├── train.py
│   │   │   └── promote.py
│   │   └── registry/
│   │       └── local_registry.py
│   ├── tests/
│   └── pyproject.toml
|
├── infra/
│   └── compose.yaml
|
├── docs/
│   ├── architecture.md
│   ├── data-contract.md
│   ├── feature-catalog.md
│   ├── model-selection.md
│   ├── model-card-template.md
│   ├── qwen-development-workflow.md
│   ├── local-development.md
│   ├── retraining-runbook.md
│   └── adr/
│       └── 001-qwen-is-not-the-predictor.md
|
├── .env.example
├── .gitignore
├── AGENTS.md
├── Makefile
└── README.md
```

There is deliberately no `qwen-worker` directory.

---

## Local development setup

> The rest of this section describes creating the project from an empty
> directory, which is how it was originally built. To run the repository as it
> exists now, follow **[docs/local-development.md](docs/local-development.md)**
> instead: five commands, plus how the running system works and what is still
> missing.

### Prerequisites

Install locally:

- Git;
- Python 3.12;
- `uv`;
- Node.js compatible with the current Vite release;
- Docker Desktop or another Docker-compatible engine;
- Ollama or the chosen local runtime for Qwen;
- a local Qwen coding model that fits the available memory.

Qwen is required only for the assisted-development workflow. The API and web application must run without Ollama.

### Create the repository

```bash
mkdir epl-ai-predictor
cd epl-ai-predictor
git init

mkdir -p apps ml/data/raw ml/data/interim ml/data/processed
mkdir -p ml/artifacts ml/reports infra docs/adr
```

### Create the FastAPI application

```bash
uv init apps/api --python 3.12
cd apps/api

uv add \
  "fastapi[standard]" \
  sqlalchemy \
  alembic \
  "psycopg[binary]" \
  pydantic-settings \
  httpx

uv add --dev \
  pytest \
  pytest-asyncio \
  ruff \
  mypy

cd ../..
```

### Create the ML package

```bash
cd ml
uv init --python 3.12

uv add \
  pandas \
  numpy \
  pyarrow \
  scikit-learn \
  catboost \
  joblib \
  huggingface-hub

uv add --dev \
  pytest \
  ruff \
  mypy

cd ..
```

Install each foundation-model library in a separate experiment or optional dependency group rather than placing every large framework in the initial environment.

Examples to investigate when the corresponding adapter is implemented:

```bash
# Primary candidate
uv add tabicl

# Mitra and TabPFNMix candidates
uv add "autogluon.tabular[mitra]"

# TabSTAR candidate
uv add tabstar

# TabPFN-3 research candidate
uv add tabpfn
```

Package extras and names may change. Recheck the current model card before installation.

### Create the frontend

```bash
npm create vite@latest apps/web -- --template react-ts
cd apps/web
npm install
npm install @tanstack/react-query
cd ../..
```

### PostgreSQL with Docker Compose

Create `infra/compose.yaml`:

```yaml
name: epl-ai-predictor

services:
  postgres:
    image: postgres:17-alpine
    environment:
      POSTGRES_DB: epl_predictor
      POSTGRES_USER: epl
      POSTGRES_PASSWORD: epl
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U epl -d epl_predictor"]
      interval: 5s
      timeout: 5s
      retries: 10

volumes:
  postgres_data:
```

Start it:

```bash
docker compose -f infra/compose.yaml up -d
```

### Environment variables

Create `.env.example`:

```dotenv
APP_ENV=development
DATABASE_URL=postgresql+psycopg://epl:epl@localhost:5432/epl_predictor

FOOTBALL_DATA_API_TOKEN=replace_me
FOOTBALL_DATA_BASE_URL=https://api.football-data.org/v4

MODEL_REGISTRY_PATH=../../ml/artifacts
ACTIVE_MODEL_NAME=stub-epl
ACTIVE_MODEL_VERSION=0.0.1
```

Never commit the real `.env` or provider token.

### Local Qwen setup

Use the Qwen coding model already selected for the machine. With Ollama, the pattern is:

```bash
ollama pull <your-qwen-coding-model>
ollama run <your-qwen-coding-model>
```

A current example for capable hardware is Qwen3-Coder. Model size must be selected according to available RAM or unified memory.

Qwen should be launched from the repository root so it can be given the project README, `AGENTS.md`, and the exact task scope.

---

## Database design

### `teams`

```text
id
provider_id
canonical_name
short_name
code
crest_url
created_at
updated_at
```

### `team_aliases`

```text
id
team_id
source
alias
```

This table resolves naming differences between historical CSV files and fixture APIs.

### `fixtures`

```text
id
provider
provider_id
competition_code
season_start_year
matchday
kickoff_at
status
home_team_id
away_team_id
home_score
away_score
result
created_at
updated_at
```

### `data_imports`

```text
id
source
source_uri
season_start_year
checksum
rows_read
rows_accepted
rows_rejected
imported_at
report_json
```

### `feature_snapshots`

```text
id
fixture_id
feature_schema_version
calculated_at
features_json
source_cutoff_at
```

`source_cutoff_at` records the latest allowed information timestamp.

### `model_versions`

```text
id
model_name
version
adapter_type
artifact_path
feature_schema_version
trained_from
trained_until
validated_from
validated_until
tested_from
tested_until
metrics_json
license_summary
status
created_at
activated_at
```

### `predictions`

```text
id
fixture_id
model_version_id
feature_snapshot_id
home_win_probability
draw_probability
away_win_probability
predicted_outcome
created_at
```

Recommended constraints:

- unique provider and provider fixture ID;
- one canonical prediction per fixture and active model version;
- probabilities constrained to `[0, 1]`;
- result limited to `H`, `D`, `A`, or null;
- model status limited to the documented enum.

---

## API design

### Initial endpoints

```text
GET  /health
GET  /api/v1/fixtures
GET  /api/v1/fixtures/{fixture_id}
POST /api/v1/fixtures/{fixture_id}/predict
GET  /api/v1/predictions/{prediction_id}
```

Later administrative or CLI-driven operations:

```text
POST /api/v1/admin/import-fixtures
POST /api/v1/admin/import-results
POST /api/v1/admin/rebuild-features
POST /api/v1/admin/train-candidate
POST /api/v1/admin/promote-model/{model_version_id}
```

For local version 1, training and promotion are preferably CLI operations rather than public HTTP endpoints.

### Fixture response

```json
{
  "id": 14621,
  "competition_code": "PL",
  "season_start_year": 2026,
  "matchday": 4,
  "kickoff_at": "2026-09-12T14:00:00Z",
  "status": "SCHEDULED",
  "home_team": {
    "id": 1,
    "name": "Arsenal",
    "crest_url": null
  },
  "away_team": {
    "id": 2,
    "name": "Chelsea",
    "crest_url": null
  }
}
```

### Prediction endpoint behavior

`POST /api/v1/fixtures/{fixture_id}/predict` must:

1. reject unknown fixtures;
2. reject completed or invalid fixtures according to product rules;
3. load the active model metadata;
4. build or reuse a compatible feature snapshot;
5. validate all required features;
6. call the active predictor adapter;
7. validate the three probabilities;
8. save the result transactionally;
9. return the model and feature versions.

---

## Implementation milestones

### Milestone 0 - repository contract

- Create repository structure.
- Add this README.
- Add `AGENTS.md` for Qwen.
- Add formatting, linting, and test commands.
- Add `.env.example` and secret rules.

### Milestone 1 - vertical web slice with mock data

- Start PostgreSQL.
- Create teams and fixtures tables.
- Seed three fixtures.
- Implement `GET /fixtures`.
- Build React fixture cards.
- Implement a deterministic stub predictor.
- Add the Predict button and result panel.
- Add API and component tests.

### Milestone 2 - real current fixtures

- Integrate football-data.org.
- Add idempotent fixture synchronization.
- Add team alias normalization.
- Add importer tests with stored fixture responses.

### Milestone 3 - historical data pipeline

- Download selected EPL CSV seasons.
- Preserve raw files and checksums.
- Normalize schemas and team names.
- Validate rows and produce import reports.
- Build canonical Parquet data.

### Milestone 4 - leakage-safe features

- Implement chronological team state.
- Add rolling form features.
- Add Elo ratings.
- Add league-position snapshots.
- Add tests proving current-match information is excluded.
- Publish feature schema `1.0.0`.

### Milestone 5 - baseline models

- Train dummy baseline.
- Train Logistic Regression.
- Train CatBoost.
- Add chronological evaluation report.
- Connect the best baseline to FastAPI.

### Milestone 6 - Hugging Face candidates

- Implement TabICLv2 adapter and experiment.
- Implement Mitra adapter and experiment.
- Implement TabSTAR adapter if resources allow.
- Implement TabPFNMix adapter.
- Run TabPFN-3 only as a licensed research benchmark.
- Compare all candidates with the same test harness.

### Milestone 7 - model registry and promotion

- Save complete versioned artifacts.
- Register model metadata in PostgreSQL.
- Implement candidate and active statuses.
- Implement rollback.
- Add promotion-policy tests.

### Milestone 8 - continual batch retraining

- Import completed fixtures.
- Rebuild dataset deterministically.
- Train a new candidate on a schedule or explicit command.
- Generate comparison report.
- Require manual promotion initially.

### Milestone 9 - optional Hugging Face publication

- Review data and model licenses.
- Remove secrets and private metadata.
- Create model card.
- Upload approved model artifact.
- Verify local reload through the Hub interface.

---

## Qwen development workflow

Detailed prompts will be created after this README is accepted. The following rules define how those prompts should work.

### Context provided to Qwen

At the start of a coding session, provide:

1. this `README.md`;
2. `AGENTS.md`;
3. the current repository tree;
4. the exact milestone and task;
5. relevant existing files;
6. acceptance criteria;
7. allowed commands;
8. explicit non-goals.

### One prompt, one bounded change

Good scope:

```text
Create the initial FastAPI health endpoint, its test, and only the minimum package configuration required to run it.
```

Bad scope:

```text
Build the whole football AI platform.
```

### Required Qwen response behavior

For each implementation task, Qwen should:

1. restate the task and assumptions;
2. inspect existing files before editing;
3. give a short implementation plan;
4. modify only in-scope files;
5. avoid destructive commands;
6. run relevant formatting, linting, type checks, and tests;
7. report commands actually executed;
8. report changed files;
9. identify failures honestly;
10. avoid claiming success without command output.

### Safety and repository rules

Qwen must not:

- delete data directories without explicit approval;
- rewrite Git history;
- expose `.env` secrets;
- call production services;
- download unapproved datasets or model weights;
- alter model results manually;
- weaken failing tests simply to make them pass;
- bypass migration history;
- introduce a Qwen runtime dependency into the application.

### Suggested validation commands

The exact commands will evolve, but the repository should converge on commands similar to:

```bash
make format
make lint
make typecheck
make test
make test-api
make test-ml
make test-web
```

### Prompt sequence to create next

The next phase should create reusable Qwen prompts in this order:

1. repository bootstrap prompt;
2. project-instructions and `AGENTS.md` prompt;
3. PostgreSQL and Docker Compose prompt;
4. FastAPI skeleton prompt;
5. Alembic and initial schema prompt;
6. mock fixture endpoint prompt;
7. React fixture page prompt;
8. stub prediction endpoint prompt;
9. historical CSV importer prompt;
10. chronological feature-builder prompt;
11. baseline training prompt;
12. TabICLv2 experiment prompt.

---

## Acceptance criteria

The first product milestone is complete when:

- the website lists upcoming Premier League fixtures from PostgreSQL;
- fixture synchronization is idempotent;
- each fixture has a Predict button;
- prediction calls use a versioned active model;
- the response includes three valid probabilities;
- the probabilities sum to approximately `1.0`;
- every prediction stores its feature snapshot and model version;
- the UI shows home, draw, and away percentages;
- historical feature generation uses only pre-kickoff information;
- the final test season was not used for tuning;
- the model beats defined dummy baselines;
- draw-class results are explicitly reported;
- a failed or missing model produces a clear API error rather than fabricated output;
- a previous active model can be restored without retraining;
- the complete application runs without Qwen or Ollama.

---

## Risks and limitations

### Limited sample size

One Premier League season has only 380 matches. Even many seasons produce a relatively small ML dataset. Complex neural models can overfit and may lose to gradient-boosted trees.

Mitigation:

- keep strong simple baselines;
- use chronological validation;
- regularize models;
- keep the first feature set small;
- report uncertainty and calibration.

### Draw-class difficulty

Draws are difficult to predict and may be underrepresented relative to combined wins.

Mitigation:

- report per-class metrics;
- inspect class weighting carefully;
- avoid optimizing only accuracy;
- evaluate calibration by class.

### Concept drift

Teams, managers, playing styles, league strength, and promoted clubs change over time.

Mitigation:

- use recency-aware backtests;
- compare expanding and rolling training windows;
- monitor live shadow predictions;
- retrain in controlled batches.

### Team-name and provider mismatch

Data sources may use different names for the same club.

Mitigation:

- maintain canonical teams and source aliases;
- reject unresolved teams;
- test promoted and relegated club mappings.

### Data-provider changes

Schemas, API plans, endpoints, and rate limits may change.

Mitigation:

- wrap providers behind adapters;
- preserve raw responses for tests;
- validate schema changes;
- document provider terms and versions.

### Licensing

Model and data licenses differ. TabPFN-3 currently restricts commercial and production use. TabSTAR presents separate model-card and repository license information.

Mitigation:

- record a license summary with every model version;
- recheck licenses before promotion or publication;
- exclude incompatible candidates from production;
- obtain legal review for commercial use.

### False certainty

A probability is not a guarantee.

Mitigation:

- display all three probabilities;
- avoid language such as "certain" or "will win";
- show model version and limitations;
- do not present the project as financial advice.

---

## Documentation plan

The README is the project overview. Detailed topics should move into focused documents as implementation begins.

### `docs/architecture.md`

- system context;
- online prediction flow;
- training flow;
- component boundaries;
- failure handling;
- deployment considerations.

### `docs/data-contract.md`

For every field:

- source;
- type;
- null policy;
- availability time;
- whether it is permitted at prediction time;
- normalization rules.

### `docs/feature-catalog.md`

For every feature:

- exact formula;
- rolling window;
- state-update order;
- default/null behavior;
- feature version introduced;
- leakage tests.

### `docs/model-selection.md`

- candidate versions;
- hardware;
- dependencies;
- hyperparameters;
- chronological splits;
- metrics;
- license analysis;
- final decision.

### `docs/model-card-template.md`

- intended use;
- out-of-scope use;
- training data range;
- feature schema;
- target mapping;
- validation method;
- metrics;
- limitations;
- license;
- artifact checksum.

### `docs/retraining-runbook.md`

- import new results;
- validate data;
- rebuild features;
- train candidate;
- evaluate;
- register;
- promote;
- roll back.

### `docs/adr/001-qwen-is-not-the-predictor.md`

Decision:

```text
Qwen is a local implementation assistant only. It is not a runtime service and does not produce match probabilities.
```

---

## Next step

Create the local Qwen prompt pack, beginning with:

```text
Prompt 01: bootstrap the repository and create AGENTS.md
```

That prompt should make Qwen create only the initial directory structure, base configuration, developer commands, and validation tests. It should not yet implement data ingestion or machine learning.

---

## References

### Qwen and local coding

- Qwen3-Coder on Ollama: https://ollama.com/library/qwen3-coder

### Application stack

- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
- Vite guide: https://vite.dev/guide/

### Football data

- football-data.org API v4 overview: https://docs.football-data.org/general/v4/index.html
- football-data.org competition resources: https://docs.football-data.org/general/v4/competition.html
- Football-Data.co.uk England data: https://www.football-data.co.uk/englandm/index.php

### Hugging Face model candidates

- TabICLv2 model repository: https://huggingface.co/jingang/TabICL
- TabICLv2 implementation and fine-tuning documentation: https://github.com/soda-inria/tabicl
- Mitra Classifier: https://huggingface.co/autogluon/mitra-classifier
- TabSTAR: https://huggingface.co/alana89/TabSTAR
- TabSTAR implementation: https://github.com/alanarazi7/TabSTAR
- TabPFNMix Classifier: https://huggingface.co/autogluon/tabpfn-mix-1.0-classifier
- TabPFN-3: https://huggingface.co/Prior-Labs/tabpfn_3
- TabPFN fine-tuning example: https://github.com/PriorLabs/TabPFN/blob/main/examples/finetune_classifier.py

### Model registry

- Uploading custom models to Hugging Face Hub: https://huggingface.co/docs/hub/en/models-uploading

---

## Working decision summary

```text
Qwen role:
Local implementation and coding assistant only.

Runtime predictor:
A versioned tabular multiclass model.

Primary candidate:
TabICLv2.

Main challenger:
Mitra Classifier.

Required baseline:
CatBoost.

Initial product:
Premier League fixture list plus Predict button.

Prediction output:
Home win, draw, and away win probabilities.

Primary evaluation metric:
Multiclass log loss.

Evaluation method:
Chronological seasons and walk-forward backtesting.

Continual learning:
Controlled candidate retraining and explicit promotion, not immediate online updates.
```

## PostgreSQL management

To manage the PostgreSQL database for local development:

### Start PostgreSQL

```bash
# Start the database container
 docker compose -f infra/compose.yaml up -d
```

### Check container status

```bash
# View running containers
 docker compose -f infra/compose.yaml ps

# View container logs
 docker compose -f infra/compose.yaml logs postgres
```

### Stop PostgreSQL

```bash
# Stop the database container
 docker compose -f infra/compose.yaml stop
```

### Remove containers without deleting database volume

```bash
# Remove containers but keep data volume
 docker compose -f infra/compose.yaml rm

# Or remove and recreate containers (keeps volume)
 docker compose -f infra/compose.yaml down
```
