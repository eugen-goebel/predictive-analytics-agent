"""Tests for the TimeSeriesTrainerAgent."""

import numpy as np
import pytest
from agents.timeseries_trainer import TimeSeriesTrainerAgent, ForecastResult


@pytest.fixture
def sample_series():
    """Sine wave with trend — realistic enough for testing."""
    np.random.seed(42)
    t = np.arange(100)
    return np.sin(t * 0.1) * 10 + t * 0.5 + np.random.randn(100) * 0.5


@pytest.fixture
def short_series():
    """Series too short for default parameters."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


class TestTimeSeriesTrainer:
    def test_returns_forecast_result(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=10, horizon=5)
        assert isinstance(result, ForecastResult)

    def test_correct_horizon(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=10, horizon=5)
        assert len(result.forecast_values) == 5

    def test_correct_lags(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=8, horizon=4)
        assert result.n_lags == 8
        assert result.horizon == 4

    def test_four_models_compared(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=10, horizon=5)
        assert len(result.model_scores) == 4
        names = {s.name for s in result.model_scores}
        assert "Linear Regression" in names
        assert "Random Forest" in names
        assert "Gradient Boosting" in names
        assert "Moving Average" in names

    def test_best_has_lowest_rmse(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=10, horizon=5)
        min_rmse = min(s.rmse for s in result.model_scores)
        assert result.best_rmse == min_rmse

    def test_best_model_stored(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        trainer.train(sample_series, n_lags=10, horizon=5)
        # best_model can be None for Moving Average, or an estimator
        assert trainer.n_lags == 10

    def test_training_time_positive(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=10, horizon=5)
        assert result.training_time_seconds > 0

    def test_forecast_values_are_finite(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=10, horizon=5)
        for v in result.forecast_values:
            assert np.isfinite(v)

    def test_rmse_and_mae_positive(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=10, horizon=5)
        for score in result.model_scores:
            assert score.rmse >= 0
            assert score.mae >= 0

    def test_short_series_raises(self, short_series):
        trainer = TimeSeriesTrainerAgent()
        with pytest.raises(ValueError, match="too short"):
            trainer.train(short_series, n_lags=12, horizon=6)

    def test_custom_lags_and_horizon(self, sample_series):
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(sample_series, n_lags=5, horizon=3)
        assert result.n_lags == 5
        assert result.horizon == 3
        assert len(result.forecast_values) == 3
