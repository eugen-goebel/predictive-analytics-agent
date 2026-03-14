"""Tests for FeatureEngineerAgent."""

import numpy as np
import pytest
from agents.feature_engineer import FeatureEngineerAgent, FeatureResult


@pytest.fixture
def engineer():
    return FeatureEngineerAgent()


class TestFeatureEngineer:
    def test_select_returns_tuple(self, engineer):
        X = np.random.randn(100, 8)
        y = np.random.randint(0, 2, 100)
        result, X_sel = engineer.select_features(X, y, [f"f{i}" for i in range(8)], "classification")
        assert isinstance(result, FeatureResult)

    def test_keeps_all_if_few_features(self, engineer):
        X = np.random.randn(50, 3)
        y = np.random.randint(0, 2, 50)
        result, X_sel = engineer.select_features(X, y, ["a", "b", "c"], "classification")
        assert result.selected_feature_count == 3

    def test_reduces_features(self, engineer):
        X = np.random.randn(100, 15)
        y = np.random.randint(0, 2, 100)
        names = [f"feature_{i}" for i in range(15)]
        result, X_sel = engineer.select_features(X, y, names, "classification")
        assert result.selected_feature_count <= 10

    def test_importances_sorted(self, engineer):
        X = np.random.randn(100, 8)
        y = np.random.randint(0, 2, 100)
        result, _ = engineer.select_features(X, y, [f"f{i}" for i in range(8)], "classification")
        if len(result.feature_importances) > 1:
            scores = [fi.importance for fi in result.feature_importances]
            assert scores == sorted(scores, reverse=True)

    def test_regression_mode(self, engineer):
        X = np.random.randn(100, 8)
        y = np.random.randn(100)
        result, _ = engineer.select_features(X, y, [f"f{i}" for i in range(8)], "regression")
        assert result.selected_feature_count > 0
