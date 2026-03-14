"""Tests for ModelTrainerAgent."""

import numpy as np
import pytest
from agents.model_trainer import ModelTrainerAgent, TrainingResult


class TestModelTrainer:
    def test_classification_training(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        trainer = ModelTrainerAgent()
        result = trainer.train(X, y, "classification")
        assert isinstance(result, TrainingResult)
        assert result.task_type == "classification"

    def test_regression_training(self):
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        trainer = ModelTrainerAgent()
        result = trainer.train(X, y, "regression")
        assert result.task_type == "regression"

    def test_four_models_compared(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        trainer = ModelTrainerAgent()
        result = trainer.train(X, y, "classification")
        assert len(result.model_scores) == 4

    def test_best_model_stored(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        trainer = ModelTrainerAgent()
        trainer.train(X, y, "classification")
        assert trainer.best_model is not None
        assert trainer.X_test is not None
        assert trainer.y_test is not None

    def test_best_score_matches(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        trainer = ModelTrainerAgent()
        result = trainer.train(X, y, "classification")
        scores = [m.score for m in result.model_scores]
        assert result.best_score == max(scores)

    def test_training_time_positive(self):
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        trainer = ModelTrainerAgent()
        result = trainer.train(X, y, "classification")
        assert result.training_time_seconds > 0
