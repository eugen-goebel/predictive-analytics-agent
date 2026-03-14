"""Tests for EvaluatorAgent."""

import os
import numpy as np
import pytest
from agents.model_trainer import ModelTrainerAgent
from agents.evaluator import EvaluatorAgent, EvaluationResult


@pytest.fixture
def trained_classifier():
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    trainer = ModelTrainerAgent()
    result = trainer.train(X, y, "classification")
    return trainer, result


class TestEvaluator:
    def test_classification_evaluation(self, trained_classifier, tmp_path):
        trainer, training_result = trained_classifier
        evaluator = EvaluatorAgent()
        result = evaluator.evaluate(
            model=trainer.best_model,
            X_test=trainer.X_test, y_test=trainer.y_test,
            X_train=trainer.X_train, y_train=trainer.y_train,
            training_result=training_result,
            feature_names=[f"f{i}" for i in range(5)],
            task_type="classification",
            output_dir=str(tmp_path),
        )
        assert isinstance(result, EvaluationResult)

    def test_scores_between_0_and_1(self, trained_classifier, tmp_path):
        trainer, training_result = trained_classifier
        result = EvaluatorAgent().evaluate(
            model=trainer.best_model,
            X_test=trainer.X_test, y_test=trainer.y_test,
            X_train=trainer.X_train, y_train=trainer.y_train,
            training_result=training_result,
            feature_names=[f"f{i}" for i in range(5)],
            task_type="classification",
            output_dir=str(tmp_path),
        )
        assert 0 <= result.test_score <= 1
        assert 0 <= result.train_score <= 1

    def test_charts_created(self, trained_classifier, tmp_path):
        trainer, training_result = trained_classifier
        result = EvaluatorAgent().evaluate(
            model=trainer.best_model,
            X_test=trainer.X_test, y_test=trainer.y_test,
            X_train=trainer.X_train, y_train=trainer.y_train,
            training_result=training_result,
            feature_names=[f"f{i}" for i in range(5)],
            task_type="classification",
            output_dir=str(tmp_path),
        )
        assert len(result.charts) >= 2
        for chart in result.charts:
            assert os.path.exists(os.path.join(str(tmp_path), chart.filename))

    def test_classification_report_present(self, trained_classifier, tmp_path):
        trainer, training_result = trained_classifier
        result = EvaluatorAgent().evaluate(
            model=trainer.best_model,
            X_test=trainer.X_test, y_test=trainer.y_test,
            X_train=trainer.X_train, y_train=trainer.y_train,
            training_result=training_result,
            feature_names=[f"f{i}" for i in range(5)],
            task_type="classification",
            output_dir=str(tmp_path),
        )
        assert result.classification_report is not None

    def test_metrics_populated(self, trained_classifier, tmp_path):
        trainer, training_result = trained_classifier
        result = EvaluatorAgent().evaluate(
            model=trainer.best_model,
            X_test=trainer.X_test, y_test=trainer.y_test,
            X_train=trainer.X_train, y_train=trainer.y_train,
            training_result=training_result,
            feature_names=[f"f{i}" for i in range(5)],
            task_type="classification",
            output_dir=str(tmp_path),
        )
        assert "accuracy" in result.metrics
        assert "precision" in result.metrics

    def test_regression_evaluation(self, tmp_path):
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        trainer = ModelTrainerAgent()
        training_result = trainer.train(X, y, "regression")
        result = EvaluatorAgent().evaluate(
            model=trainer.best_model,
            X_test=trainer.X_test, y_test=trainer.y_test,
            X_train=trainer.X_train, y_train=trainer.y_train,
            training_result=training_result,
            feature_names=[f"f{i}" for i in range(5)],
            task_type="regression",
            output_dir=str(tmp_path),
        )
        assert "rmse" in result.metrics
        assert result.classification_report is None
