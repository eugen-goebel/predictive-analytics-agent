"""
Feature Engineer — Selects the most important features for the ML model.

Not all columns are equally useful for predictions. Some features are
highly predictive, others add noise. This agent:

  1. Measures how important each feature is (using statistical tests)
  2. Removes features with near-zero variance (they carry no information)
  3. Keeps only the top features

WHY FEATURE SELECTION?
  - Fewer features = faster training
  - Removes noise = better accuracy
  - Simpler model = easier to explain to stakeholders
"""

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class FeatureImportance(BaseModel):
    """Importance score for a single feature."""
    name: str
    importance: float = Field(description="Higher = more important for predictions")


class FeatureResult(BaseModel):
    """Results of the feature selection process."""
    original_feature_count: int
    selected_feature_count: int
    selected_features: list[str]
    dropped_features: list[str]
    feature_importances: list[FeatureImportance]
    method: str = Field(description="Selection method used")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FeatureEngineerAgent:
    """
    Selects the most important features using statistical tests.

    For classification: uses ANOVA F-test (do groups differ significantly?)
    For regression: uses F-regression (is there a linear relationship?)

    Usage:
        engineer = FeatureEngineerAgent()
        result, X_selected = engineer.select_features(X, y, feature_names, "classification")
    """

    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        task_type: str,
    ) -> tuple[FeatureResult, np.ndarray]:
        """
        Select the best features from the dataset.

        Args:
            X:             Feature matrix (numpy array)
            y:             Target array
            feature_names: Names of the features (columns)
            task_type:     "classification" or "regression"

        Returns:
            Tuple of (FeatureResult metadata, X with only selected features)
        """
        original_count = X.shape[1]

        # --- Step 1: Remove near-zero variance features ---
        vt = VarianceThreshold(threshold=0.01)
        try:
            X_var = vt.fit_transform(X)
            var_mask = vt.get_support()
            remaining_names = [n for n, keep in zip(feature_names, var_mask) if keep]
            dropped_by_var = [n for n, keep in zip(feature_names, var_mask) if not keep]
        except ValueError:
            X_var = X
            remaining_names = list(feature_names)
            dropped_by_var = []

        # --- Step 2: If few features, keep all ---
        if len(remaining_names) <= 5:
            importances = self._compute_importances(X_var, y, remaining_names, task_type)
            return FeatureResult(
                original_feature_count=original_count,
                selected_feature_count=len(remaining_names),
                selected_features=remaining_names,
                dropped_features=dropped_by_var,
                feature_importances=importances,
                method="all features kept (<=5 features)",
            ), X_var

        # --- Step 3: Statistical feature selection ---
        k = min(10, len(remaining_names))
        score_func = f_classif if task_type == "classification" else f_regression

        try:
            selector = SelectKBest(score_func=score_func, k=k)
            X_selected = selector.fit_transform(X_var, y)
            select_mask = selector.get_support()

            selected = [n for n, keep in zip(remaining_names, select_mask) if keep]
            dropped_by_select = [n for n, keep in zip(remaining_names, select_mask) if not keep]

            # Build importances from scores
            scores = selector.scores_
            importances = []
            for name, score, keep in zip(remaining_names, scores, select_mask):
                if keep and np.isfinite(score):
                    importances.append(FeatureImportance(name=name, importance=round(float(score), 4)))
            importances.sort(key=lambda x: x.importance, reverse=True)

            method = f"SelectKBest ({score_func.__name__}, k={k})"
        except Exception:
            # Fallback: keep all
            X_selected = X_var
            selected = remaining_names
            dropped_by_select = []
            importances = self._compute_importances(X_var, y, remaining_names, task_type)
            method = "fallback: all features kept"

        all_dropped = dropped_by_var + dropped_by_select

        return FeatureResult(
            original_feature_count=original_count,
            selected_feature_count=len(selected),
            selected_features=selected,
            dropped_features=all_dropped,
            feature_importances=importances,
            method=method,
        ), X_selected

    def _compute_importances(
        self, X: np.ndarray, y: np.ndarray, names: list[str], task_type: str
    ) -> list[FeatureImportance]:
        """Compute importance scores for all features."""
        score_func = f_classif if task_type == "classification" else f_regression
        try:
            scores, _ = score_func(X, y)
            importances = [
                FeatureImportance(name=n, importance=round(float(s), 4))
                for n, s in zip(names, scores)
                if np.isfinite(s)
            ]
            importances.sort(key=lambda x: x.importance, reverse=True)
            return importances
        except Exception:
            return [FeatureImportance(name=n, importance=0.0) for n in names]
