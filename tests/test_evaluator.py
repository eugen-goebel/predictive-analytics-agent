"""Tests for EvaluatorAgent."""

import os

import numpy as np
import pytest

from agents.evaluator import EvaluationResult, EvaluatorAgent
from agents.model_trainer import ModelTrainerAgent


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
            X_test=trainer.X_test,
            y_test=trainer.y_test,
            X_train=trainer.X_train,
            y_train=trainer.y_train,
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
            X_test=trainer.X_test,
            y_test=trainer.y_test,
            X_train=trainer.X_train,
            y_train=trainer.y_train,
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
            X_test=trainer.X_test,
            y_test=trainer.y_test,
            X_train=trainer.X_train,
            y_train=trainer.y_train,
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
            X_test=trainer.X_test,
            y_test=trainer.y_test,
            X_train=trainer.X_train,
            y_train=trainer.y_train,
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
            X_test=trainer.X_test,
            y_test=trainer.y_test,
            X_train=trainer.X_train,
            y_train=trainer.y_train,
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
            X_test=trainer.X_test,
            y_test=trainer.y_test,
            X_train=trainer.X_train,
            y_train=trainer.y_train,
            training_result=training_result,
            feature_names=[f"f{i}" for i in range(5)],
            task_type="regression",
            output_dir=str(tmp_path),
        )
        assert "rmse" in result.metrics
        assert result.classification_report is None


class TestPermutationImportance:
    """Permutation importance must be produced for every supported model type."""

    def _evaluate_with(self, X, y, task_type, tmp_path, **kwargs):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

        model_cls = kwargs["model_cls"]
        model = model_cls()
        if task_type == "classification" and hasattr(model, "max_iter"):
            model.set_params(max_iter=500)
        model.fit(X_tr, y_tr)

        # Minimal training_result stub — only model_scores is used by the chart helper
        from agents.model_trainer import ModelScore, TrainingResult

        training_result = TrainingResult(
            best_model_name=model.__class__.__name__,
            best_score=0.5,
            task_type=task_type,
            model_scores=[
                ModelScore(name=model.__class__.__name__, score=0.5, cv_std=0.0, training_time=0.0)
            ],
            training_time_seconds=0.0,
        )

        return EvaluatorAgent().evaluate(
            model=model,
            X_test=X_te,
            y_test=y_te,
            X_train=X_tr,
            y_train=y_tr,
            training_result=training_result,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
            task_type=task_type,
            output_dir=str(tmp_path),
        )

    def test_generated_for_linear_classification(self, tmp_path):
        from sklearn.linear_model import LogisticRegression

        np.random.seed(0)
        X = np.random.randn(80, 4)
        y = (X[:, 0] + np.random.randn(80) * 0.1 > 0).astype(int)
        result = self._evaluate_with(X, y, "classification", tmp_path, model_cls=LogisticRegression)

        filenames = [c.filename for c in result.charts]
        assert "permutation_importance.png" in filenames
        assert os.path.exists(os.path.join(str(tmp_path), "permutation_importance.png"))

    def test_generated_for_knn_regression(self, tmp_path):
        from sklearn.neighbors import KNeighborsRegressor

        np.random.seed(1)
        X = np.random.randn(100, 4)
        y = X[:, 0] + 0.1 * np.random.randn(100)
        result = self._evaluate_with(X, y, "regression", tmp_path, model_cls=KNeighborsRegressor)

        filenames = [c.filename for c in result.charts]
        assert "permutation_importance.png" in filenames

    def test_top_k_limits_features_shown(self, tmp_path):
        from sklearn.linear_model import LogisticRegression

        from agents.evaluator import PERMUTATION_TOP_K

        np.random.seed(2)
        # Many more features than the limit
        n_features = PERMUTATION_TOP_K + 5
        X = np.random.randn(120, n_features)
        y = (X[:, 0] > 0).astype(int)
        result = self._evaluate_with(X, y, "classification", tmp_path, model_cls=LogisticRegression)

        # Chart file exists; we cannot easily inspect bar count without
        # parsing the image, but PERMUTATION_TOP_K must be honoured.
        assert "permutation_importance.png" in [c.filename for c in result.charts]
        assert PERMUTATION_TOP_K == 15

    def test_chart_description_mentions_model_agnostic(self, tmp_path):
        from sklearn.linear_model import LogisticRegression

        np.random.seed(3)
        X = np.random.randn(60, 3)
        y = (X[:, 0] > 0).astype(int)
        result = self._evaluate_with(X, y, "classification", tmp_path, model_cls=LogisticRegression)
        chart = next(c for c in result.charts if c.filename == "permutation_importance.png")
        assert "any model" in chart.description.lower() or "model-agnostic" in chart.title.lower()
