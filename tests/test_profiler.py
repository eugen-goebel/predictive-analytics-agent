"""Tests for DataProfiler."""

import os
import pytest
from agents.data_profiler import DataProfiler, DataProfile


@pytest.fixture
def profiler():
    return DataProfiler()


@pytest.fixture
def sample_path():
    return os.path.join(os.path.dirname(__file__), "..", "data", "sample_customers.csv")


class TestDataProfiler:
    def test_profile_returns_tuple(self, profiler, sample_path):
        profile, df = profiler.profile(sample_path)
        assert isinstance(profile, DataProfile)

    def test_detects_classification(self, profiler, sample_path):
        profile, _ = profiler.profile(sample_path)
        assert profile.task_type == "classification"
        assert profile.target_column == "churn"

    def test_row_count(self, profiler, sample_path):
        profile, _ = profiler.profile(sample_path)
        assert profile.row_count == 80

    def test_column_profiles(self, profiler, sample_path):
        profile, _ = profiler.profile(sample_path)
        assert len(profile.column_profiles) == profile.column_count

    def test_numeric_stats(self, profiler, sample_path):
        profile, _ = profiler.profile(sample_path)
        assert len(profile.numeric_stats) > 0
        assert all(s.std >= 0 for s in profile.numeric_stats)

    def test_class_distribution(self, profiler, sample_path):
        profile, _ = profiler.profile(sample_path)
        assert profile.class_distribution is not None
        assert sum(profile.class_distribution.values()) == profile.row_count

    def test_quality_score(self, profiler, sample_path):
        profile, _ = profiler.profile(sample_path)
        assert 0 <= profile.data_quality_score <= 100

    def test_file_not_found(self, profiler):
        with pytest.raises(FileNotFoundError):
            profiler.profile("/nonexistent.csv")

    def test_unsupported_format(self, profiler, tmp_path):
        p = tmp_path / "test.json"
        p.write_text("{}")
        with pytest.raises(ValueError):
            profiler.profile(str(p))

    def test_model_json_roundtrip(self, profiler, sample_path):
        profile, _ = profiler.profile(sample_path)
        restored = DataProfile.model_validate_json(profile.model_dump_json())
        assert restored.filename == profile.filename
