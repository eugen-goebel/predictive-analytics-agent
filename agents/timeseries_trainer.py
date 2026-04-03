"""
Time Series Trainer — Forecasting models for temporal data.

Supports multiple approaches for time series prediction:
  - Lag-based regression (using sklearn regressors on lag features)
  - Simple moving average baseline
  - Exponential smoothing

The agent auto-generates lag features from a time-ordered numeric series
and trains regression models for multi-step forecasting.
"""

import time
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TimeSeriesModelScore(BaseModel):
    """Performance metrics for a single forecasting model."""
    name: str
    rmse: float = Field(description="Root mean squared error on test set")
    mae: float = Field(description="Mean absolute error on test set")
    training_time: float = Field(description="Training time in seconds")


class ForecastResult(BaseModel):
    """Results of time series model training and selection."""
    best_model_name: str
    best_rmse: float
    n_lags: int = Field(description="Number of lag features used")
    horizon: int = Field(description="Forecast horizon (steps ahead)")
    model_scores: list[TimeSeriesModelScore]
    training_time_seconds: float
    forecast_values: list[float] = Field(description="Forecasted values for the next `horizon` steps")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TimeSeriesTrainerAgent:
    """
    Trains time series forecasting models using lag-based feature engineering.

    Converts a single time series into a supervised learning problem by
    creating lag features, then compares multiple regressors.

    Usage:
        trainer = TimeSeriesTrainerAgent()
        result = trainer.train(series, n_lags=12, horizon=6)
    """

    def __init__(self):
        self.best_model = None
        self.n_lags = None

    def train(
        self,
        series: np.ndarray,
        n_lags: int = 12,
        horizon: int = 6,
    ) -> ForecastResult:
        """
        Train forecasting models on a time series.

        Args:
            series:   1D array of time-ordered values
            n_lags:   Number of past observations to use as features
            horizon:  Number of future steps to forecast

        Returns:
            ForecastResult with model comparison and forecast
        """
        total_start = time.time()
        self.n_lags = n_lags

        if len(series) < n_lags + horizon + 10:
            raise ValueError(
                f"Series too short ({len(series)} points) for "
                f"n_lags={n_lags} and horizon={horizon}. "
                f"Need at least {n_lags + horizon + 10} data points."
            )

        # Build supervised dataset from lag features
        X, y = self._create_lag_features(series, n_lags)

        # Time-aware train/test split (last 20% for testing)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Define candidate models
        candidates = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Moving Average": None,  # handled separately
        }

        model_scores: list[TimeSeriesModelScore] = []
        best_name = ""
        best_rmse = np.inf
        best_estimator = None

        for name, model in candidates.items():
            start = time.time()

            if name == "Moving Average":
                # Simple moving average baseline
                y_pred = self._moving_average_predict(series, split_idx, n_lags, len(y_test))
                rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
                mae = float(np.mean(np.abs(y_test - y_pred)))
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
                mae = float(np.mean(np.abs(y_test - y_pred)))

            elapsed = time.time() - start

            model_scores.append(TimeSeriesModelScore(
                name=name,
                rmse=round(rmse, 4),
                mae=round(mae, 4),
                training_time=round(elapsed, 3),
            ))

            if rmse < best_rmse:
                best_rmse = rmse
                best_name = name
                best_estimator = model

        # Refit best model on full data
        if best_name != "Moving Average" and best_estimator is not None:
            best_estimator.fit(X, y)
        self.best_model = best_estimator

        # Generate forecast
        forecast = self._forecast(series, n_lags, horizon, best_name)

        total_time = time.time() - total_start

        return ForecastResult(
            best_model_name=best_name,
            best_rmse=round(best_rmse, 4),
            n_lags=n_lags,
            horizon=horizon,
            model_scores=model_scores,
            training_time_seconds=round(total_time, 3),
            forecast_values=[round(v, 4) for v in forecast],
        )

    def _create_lag_features(self, series: np.ndarray, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert a time series into a supervised learning dataset using lag features."""
        X, y = [], []
        for i in range(n_lags, len(series)):
            X.append(series[i - n_lags:i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def _moving_average_predict(
        self, series: np.ndarray, split_idx: int, window: int, n_predictions: int
    ) -> np.ndarray:
        """Generate moving average predictions for the test period."""
        predictions = []
        data = list(series[:split_idx + window])
        for i in range(n_predictions):
            idx = split_idx + window + i
            start = idx - window
            avg = np.mean(data[start:idx])
            predictions.append(avg)
            if idx < len(series):
                data.append(series[idx])
            else:
                data.append(avg)
        return np.array(predictions)

    def _forecast(
        self, series: np.ndarray, n_lags: int, horizon: int, model_name: str
    ) -> list[float]:
        """Generate future forecast values."""
        values = list(series[-n_lags:])

        for _ in range(horizon):
            if model_name == "Moving Average":
                pred = float(np.mean(values[-n_lags:]))
            else:
                features = np.array(values[-n_lags:]).reshape(1, -1)
                pred = float(self.best_model.predict(features)[0])
            values.append(pred)

        return values[n_lags:]
