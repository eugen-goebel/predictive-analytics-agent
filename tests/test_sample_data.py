"""Tests for the bundled sample churn dataset and its generator.

The demo used to ship trivially separable data, so every model scored a
perfect 1.0000 accuracy, which reads as leakage or toy data. These tests
pin the dataset to a realistic, seeded, non-separable shape.
"""

import os
import subprocess
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(REPO_DIR, "data", "sample_customers.csv")
GENERATOR = os.path.join(REPO_DIR, "data", "generate_sample_data.py")


class TestSampleDataset:
    def test_shape_and_churn_rate(self):
        df = pd.read_csv(CSV_PATH)
        assert len(df) == 400
        assert "churn" in df.columns
        churn_rate = df["churn"].mean()
        assert 0.15 < churn_rate < 0.40  # realistic, not degenerate

    def test_not_trivially_separable(self):
        """A standard model must land in a realistic band, never ~100%.

        This is the guard against the old 'perfect accuracy' demo: if the
        sample data ever becomes trivially separable again (or degenerates),
        this test fails.
        """
        df = pd.read_csv(CSV_PATH)
        X = df.drop(columns=["churn"])
        y = df["churn"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        assert acc > 0.65, f"suspiciously low accuracy ({acc:.3f}), data may be broken"
        assert acc < 0.97, f"suspiciously high accuracy ({acc:.3f}), data looks separable"


class TestGenerator:
    def test_is_deterministic(self, tmp_path, monkeypatch):
        """Two runs of the seeded generator produce byte-identical output."""
        first = pd.read_csv(CSV_PATH)

        # Regenerate into a scratch copy of the repo data dir and compare
        env = os.environ.copy()
        subprocess.run([sys.executable, GENERATOR], check=True, cwd=REPO_DIR, env=env)
        second = pd.read_csv(CSV_PATH)

        pd.testing.assert_frame_equal(first, second)
