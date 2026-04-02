"""
Model Trainer — Trains multiple ML models and picks the best one.

Instead of guessing which algorithm works best, this agent trains
FOUR different models and compares them automatically:

  Classification:                    Regression:
  - Logistic Regression              - Linear Regression
  - Random Forest                    - Random Forest
  - Gradient Boosting                - Gradient Boosting
  - K-Nearest Neighbors              - K-Nearest Neighbors

WHAT IS CROSS-VALIDATION?
  Instead of testing once, we split the training data 5 times (5-fold CV).
  Each time, a different 20% is used for testing. The average score across
  all 5 folds gives a more reliable estimate of real performance.
"""

import time
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ModelScore(BaseModel):
    """Performance metrics for a single model."""
    name: str
    score: float = Field(description="Accuracy (classification) or R² (regression)")
    cv_std: float = Field(description="Standard deviation across CV folds")
    training_time: float = Field(description="Training time in seconds")


class TrainingResult(BaseModel):
    """Results of the model comparison."""
    best_model_name: str
    best_score: float
    task_type: Literal["classification", "regression"]
    model_scores: list[ModelScore]
    training_time_seconds: float = Field(description="Total time for all models")
    tuned: bool = Field(default=False, description="Whether hyperparameter tuning was applied")
    best_params: dict | None = Field(default=None, description="Best hyperparameters found by tuning")


# ---------------------------------------------------------------------------
# Hyperparameter grids for GridSearchCV
# ---------------------------------------------------------------------------

CLASSIFICATION_PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [1000],
    },
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
}

REGRESSION_PARAM_GRIDS = {
    "Linear Regression": {},
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "Gradient Boosting": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "K-Nearest Neighbors": {
        "n_neighbors": [3, 5, 7, 11],
        "weights": ["uniform", "distance"],
    },
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ModelTrainerAgent:
    """
    Trains multiple ML models, compares them with cross-validation,
    and picks the best one.

    After training, these attributes are available:
      - self.best_model:  The fitted best model (sklearn estimator)
      - self.X_train:     Training features
      - self.y_train:     Training target
      - self.X_test:      Test features (20% holdout)
      - self.y_test:      Test target

    Usage:
        trainer = ModelTrainerAgent()
        result = trainer.train(X, y, "classification")
        print(result.best_model_name)  # e.g., "Random Forest"
        print(trainer.best_model)      # the fitted sklearn model
    """

    def __init__(self):
        self.best_model = None
        self.scaler = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def train(self, X: np.ndarray, y: np.ndarray, task_type: str, tune: bool = False) -> TrainingResult:
        """
        Train multiple models and select the best one.

        Steps:
          1. Split data 80/20 (train/test)
          2. Train 4 models with 5-fold cross-validation
          3. Pick the best model by mean CV score
          4. Refit the best model on full training set

        If tune=True, each model is first optimized via GridSearchCV
        before the comparison step.

        Args:
            X:         Feature matrix
            y:         Target array
            task_type: "classification" or "regression"
            tune:      Run hyperparameter tuning with GridSearchCV

        Returns:
            TrainingResult with comparison metrics
        """
        total_start = time.time()

        # --- Step 1: Split data ---
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # --- Step 1b: Scale features (fit on train only to prevent data leakage) ---
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # --- Step 2: Define candidate models ---
        if task_type == "classification":
            candidates = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "K-Nearest Neighbors": KNeighborsClassifier(),
            }
            scoring = "accuracy"
            param_grids = CLASSIFICATION_PARAM_GRIDS
        else:
            candidates = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                "K-Nearest Neighbors": KNeighborsRegressor(),
            }
            scoring = "r2"
            param_grids = REGRESSION_PARAM_GRIDS

        # --- Step 2b: Hyperparameter tuning (optional) ---
        tuned_params = {}
        if tune:
            candidates, tuned_params = self._tune_candidates(
                candidates, param_grids, scoring,
            )

        # --- Step 3: Train and evaluate each model ---
        model_scores: list[ModelScore] = []
        best_name = ""
        best_mean_score = -np.inf
        best_estimator = None

        for name, model in candidates.items():
            start = time.time()

            cv_scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=min(5, len(self.X_train)), scoring=scoring,
            )
            elapsed = time.time() - start

            mean_score = float(np.mean(cv_scores))
            std_score = float(np.std(cv_scores))

            model_scores.append(ModelScore(
                name=name,
                score=round(mean_score, 4),
                cv_std=round(std_score, 4),
                training_time=round(elapsed, 3),
            ))

            if mean_score > best_mean_score:
                best_mean_score = mean_score
                best_name = name
                best_estimator = model

        # --- Step 4: Refit best model on full training set ---
        best_estimator.fit(self.X_train, self.y_train)
        self.best_model = best_estimator

        total_time = time.time() - total_start

        return TrainingResult(
            best_model_name=best_name,
            best_score=round(best_mean_score, 4),
            task_type=task_type,
            model_scores=model_scores,
            training_time_seconds=round(total_time, 3),
            tuned=tune,
            best_params=tuned_params.get(best_name) if tune else None,
        )

    def _tune_candidates(
        self,
        candidates: dict,
        param_grids: dict,
        scoring: str,
    ) -> tuple[dict, dict]:
        """Run GridSearchCV on each candidate and return tuned models with best params."""
        tuned = {}
        best_params = {}
        cv_folds = min(5, len(self.X_train))

        for name, model in candidates.items():
            grid = param_grids.get(name, {})
            if not grid:
                tuned[name] = model
                best_params[name] = {}
                continue

            search = GridSearchCV(
                model, grid, scoring=scoring,
                cv=cv_folds, n_jobs=-1, error_score="raise",
            )
            search.fit(self.X_train, self.y_train)
            tuned[name] = search.best_estimator_
            best_params[name] = search.best_params_

        return tuned, best_params
