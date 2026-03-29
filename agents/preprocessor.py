"""
Preprocessor Agent — Cleans and prepares data for machine learning.

Raw data is messy: missing values, text categories, different scales.
ML models need clean numbers. This agent handles all of that:

  1. Separate features (X) from the target (y)
  2. Drop columns with too many missing values
  3. Fill remaining missing values (median for numbers, mode for categories)
  4. Encode categories as numbers (e.g., "male"=0, "female"=1)
  5. Scale all numbers to similar ranges (StandardScaler)

WHY SCALING?
  Without scaling, a column like "income" (30000-120000) would dominate
  over "age" (20-65) just because the numbers are bigger. Scaling puts
  all features on the same playing field.
"""

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from sklearn.preprocessing import LabelEncoder

from .data_profiler import DataProfile


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class PreprocessResult(BaseModel):
    """Metadata about what preprocessing was done."""
    feature_names: list[str] = Field(description="Names of features after preprocessing")
    target_column: str
    task_type: Literal["classification", "regression"]
    n_samples: int
    n_features: int
    steps_applied: list[str] = Field(description="Log of preprocessing steps")
    label_mapping: dict[str, int] | None = Field(
        default=None, description="Target label mapping (classification only)"
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PreprocessorAgent:
    """
    Cleans and prepares data for ML models. Handles missing values,
    categorical encoding, and feature scaling.

    Usage:
        preprocessor = PreprocessorAgent()
        result, X, y = preprocessor.preprocess(df, profile)
        # X = feature matrix (numpy array)
        # y = target array (numpy array)
    """

    def preprocess(
        self, df: pd.DataFrame, profile: DataProfile
    ) -> tuple[PreprocessResult, np.ndarray, np.ndarray]:
        """
        Clean and prepare data for ML.

        Args:
            df:      Raw pandas DataFrame
            profile: DataProfile from the DataProfiler agent

        Returns:
            Tuple of (PreprocessResult metadata, X features, y target)
        """
        steps: list[str] = []
        df = df.copy()

        # --- Step 1: Separate target from features ---
        target_col = profile.target_column
        y_series = df[target_col]
        X_df = df.drop(columns=[target_col])
        steps.append(f"Separated target column '{target_col}' from features")

        # --- Step 2: Drop columns with >50% missing ---
        threshold = len(df) * 0.5
        cols_before = list(X_df.columns)
        X_df = X_df.dropna(axis=1, thresh=int(len(df) - threshold))
        dropped = set(cols_before) - set(X_df.columns)
        if dropped:
            steps.append(f"Dropped {len(dropped)} columns with >50% missing: {', '.join(dropped)}")

        # --- Step 3: Fill missing values ---
        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in numeric_cols:
            n_missing = int(X_df[col].isna().sum())
            if n_missing > 0:
                median_val = X_df[col].median()
                X_df[col] = X_df[col].fillna(median_val)
                steps.append(f"Filled {n_missing} missing values in '{col}' with median ({median_val:.2f})")

        for col in cat_cols:
            n_missing = int(X_df[col].isna().sum())
            if n_missing > 0:
                mode_val = X_df[col].mode().iloc[0] if len(X_df[col].mode()) > 0 else "unknown"
                X_df[col] = X_df[col].fillna(mode_val)
                steps.append(f"Filled {n_missing} missing values in '{col}' with mode ('{mode_val}')")

        # --- Step 4: Encode categorical features ---
        cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))
            steps.append(f"Encoded categorical column '{col}' ({len(le.classes_)} categories)")

        # --- Step 5: Encode target if classification ---
        label_mapping = None
        if profile.task_type == "classification":
            if not pd.api.types.is_numeric_dtype(y_series):
                le_target = LabelEncoder()
                y_array = le_target.fit_transform(y_series.astype(str))
                label_mapping = {str(cls): int(i) for i, cls in enumerate(le_target.classes_)}
                steps.append(f"Encoded target '{target_col}' ({len(le_target.classes_)} classes)")
            else:
                y_array = y_series.values.astype(int)
        else:
            y_array = y_series.values.astype(float)

        # --- Step 6: Convert to numpy (scaling deferred to ModelTrainer to prevent data leakage) ---
        feature_names = list(X_df.columns)
        X_array = X_df.values.astype(float)

        result = PreprocessResult(
            feature_names=feature_names,
            target_column=target_col,
            task_type=profile.task_type,
            n_samples=len(X_array),
            n_features=len(feature_names),
            steps_applied=steps,
            label_mapping=label_mapping,
        )

        return result, X_array, y_array
