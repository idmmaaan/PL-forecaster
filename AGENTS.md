# Agent Guidelines

This document provides guidelines for all coding agents working on this project. These rules ensure consistency, maintainability, and adherence to the project's architecture and philosophy.

## Core Principles

- **Always read README.md and AGENTS.md before implementing anything**
- **Inspect existing implementation before changing files**
- **Implement only the task explicitly requested**
- **Avoid unrelated refactoring**
- **Keep changes small and reviewable**

## Architecture Compliance

- **Preserve the architecture described in README.md**
- **Use Qwen only as the development/coding assistant**
- **Never integrate Qwen into the football prediction runtime**
- **Keep predictor implementations behind a common interface**
- **Prevent football data leakage** (as specified in README.md)
- **Use chronological ML validation** (as specified in README.md)

## Implementation Standards

- **Add or update tests for implemented behavior**
- **Run relevant tests/linting after changes**
- **Never silently change API contracts**
- **Never add unnecessary dependencies**
- **Ask before making a significant architecture change**

## Predictor Interface

All predictor implementations must follow the common interface defined in the project:

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

This interface ensures that different predictor implementations can be used interchangeably without changing the calling code.