"""
Data Profiler — Loads and analyzes datasets for machine learning.

This is the first agent in the pipeline. It reads a CSV or Excel file and
produces a detailed profile: column types, missing values, statistics, and
most importantly, it auto-detects which column is the TARGET (what we want
to predict) and whether the task is CLASSIFICATION or REGRESSION.

CLASSIFICATION vs REGRESSION (for beginners):
  - Classification: Predict a CATEGORY (e.g., "Will the customer leave? Yes/No")
  - Regression: Predict a NUMBER (e.g., "What will the house price be?")
"""

import os
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from typing import Literal


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ColumnProfile(BaseModel):
    """Profile of a single column."""
    name: str
    dtype: Literal["numeric", "categorical", "datetime"]
    missing_count: int
    unique_count: int
    sample_values: list[str] = Field(description="Up to 5 sample values")


class NumericStat(BaseModel):
    """Descriptive statistics for a numeric column."""
    column: str
    mean: float
    median: float
    std: float
    min: float
    max: float
    skewness: float = Field(description="Positive = right-skewed, negative = left-skewed")


class DataProfile(BaseModel):
    """Complete profile of a dataset."""
    filename: str
    row_count: int
    column_count: int
    target_column: str = Field(description="Auto-detected target column")
    task_type: Literal["classification", "regression"]
    column_profiles: list[ColumnProfile]
    numeric_stats: list[NumericStat]
    class_distribution: dict[str, int] | None = Field(
        default=None, description="Value counts of target (classification only)"
    )
    data_quality_score: float = Field(description="0-100 quality score")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

SUPPORTED_FORMATS = {".csv", ".xlsx", ".xls"}


class DataProfiler:
    """
    Loads a dataset and produces a detailed profile for ML pipeline planning.
    Auto-detects the target column and task type.

    Usage:
        profiler = DataProfiler()
        profile, df = profiler.profile("data/customers.csv")
        print(profile.task_type)      # "classification" or "regression"
        print(profile.target_column)  # e.g., "churn"
    """

    def profile(self, filepath: str) -> tuple[DataProfile, pd.DataFrame]:
        """
        Load and profile a dataset.

        Args:
            filepath: Path to CSV or Excel file

        Returns:
            Tuple of (DataProfile, pandas DataFrame)
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: '{ext}'. Use CSV or Excel.")

        # Read file
        df = pd.read_csv(filepath) if ext == ".csv" else pd.read_excel(filepath)

        if df.empty or len(df.columns) < 2:
            raise ValueError("Dataset must have at least 2 columns and 1 row.")

        # Auto-detect target and task type
        target_column, task_type = self._detect_target(df)

        # Build column profiles
        column_profiles = self._build_column_profiles(df)

        # Numeric statistics
        numeric_stats = self._build_numeric_stats(df)

        # Class distribution (classification only)
        class_dist = None
        if task_type == "classification":
            class_dist = df[target_column].value_counts().to_dict()
            class_dist = {str(k): int(v) for k, v in class_dist.items()}

        # Data quality score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = int(df.isnull().sum().sum())
        completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
        data_quality_score = round(min(completeness, 100), 1)

        profile = DataProfile(
            filename=os.path.basename(filepath),
            row_count=len(df),
            column_count=len(df.columns),
            target_column=target_column,
            task_type=task_type,
            column_profiles=column_profiles,
            numeric_stats=numeric_stats,
            class_distribution=class_dist,
            data_quality_score=data_quality_score,
        )

        return profile, df

    def _detect_target(self, df: pd.DataFrame) -> tuple[str, str]:
        """
        Auto-detect which column is the target and what task type it is.

        Logic:
          1. Check the LAST column first (convention in many ML datasets)
          2. If it has <= 10 unique values → classification
          3. If it's numeric with many unique values → regression
        """
        last_col = df.columns[-1]
        n_unique = df[last_col].nunique()

        if n_unique <= 10:
            return last_col, "classification"
        elif pd.api.types.is_numeric_dtype(df[last_col]):
            return last_col, "regression"
        else:
            # Fallback: look for common target column names
            for candidate in ["target", "label", "class", "y", "outcome"]:
                if candidate in df.columns:
                    return candidate, "classification"
            return last_col, "regression"

    def _build_column_profiles(self, df: pd.DataFrame) -> list[ColumnProfile]:
        """Build a profile for each column."""
        profiles = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                dtype = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                dtype = "datetime"
            else:
                dtype = "categorical"

            samples = df[col].dropna().head(5).astype(str).tolist()
            profiles.append(ColumnProfile(
                name=col,
                dtype=dtype,
                missing_count=int(df[col].isna().sum()),
                unique_count=int(df[col].nunique()),
                sample_values=samples,
            ))
        return profiles

    def _build_numeric_stats(self, df: pd.DataFrame) -> list[NumericStat]:
        """Compute descriptive statistics for all numeric columns."""
        stats = []
        for col in df.select_dtypes(include=[np.number]).columns:
            desc = df[col].describe()
            stats.append(NumericStat(
                column=col,
                mean=round(float(desc["mean"]), 2),
                median=round(float(df[col].median()), 2),
                std=round(float(desc["std"]), 2),
                min=round(float(desc["min"]), 2),
                max=round(float(desc["max"]), 2),
                skewness=round(float(df[col].skew()), 2),
            ))
        return stats
