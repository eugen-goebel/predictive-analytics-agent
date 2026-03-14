"""Tests for PreprocessorAgent."""

import os
import numpy as np
import pytest
from agents.data_profiler import DataProfiler
from agents.preprocessor import PreprocessorAgent, PreprocessResult


@pytest.fixture
def sample_data():
    profiler = DataProfiler()
    sample = os.path.join(os.path.dirname(__file__), "..", "data", "sample_customers.csv")
    return profiler.profile(sample)


class TestPreprocessor:
    def test_returns_tuple(self, sample_data):
        profile, df = sample_data
        preprocessor = PreprocessorAgent()
        result, X, y = preprocessor.preprocess(df, profile)
        assert isinstance(result, PreprocessResult)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_shapes_match(self, sample_data):
        profile, df = sample_data
        result, X, y = PreprocessorAgent().preprocess(df, profile)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == result.n_features

    def test_no_nan_in_output(self, sample_data):
        profile, df = sample_data
        _, X, y = PreprocessorAgent().preprocess(df, profile)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))

    def test_steps_logged(self, sample_data):
        profile, df = sample_data
        result, _, _ = PreprocessorAgent().preprocess(df, profile)
        assert len(result.steps_applied) > 0

    def test_target_column_set(self, sample_data):
        profile, df = sample_data
        result, _, _ = PreprocessorAgent().preprocess(df, profile)
        assert result.target_column == profile.target_column

    def test_task_type_preserved(self, sample_data):
        profile, df = sample_data
        result, _, _ = PreprocessorAgent().preprocess(df, profile)
        assert result.task_type == profile.task_type
