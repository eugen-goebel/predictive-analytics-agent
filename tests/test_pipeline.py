"""End-to-end pipeline test."""

import os
import pytest
from agents.orchestrator import MLPipelineOrchestrator


class TestPipeline:
    def test_full_pipeline_runs(self, tmp_path):
        sample = os.path.join(os.path.dirname(__file__), "..", "data", "sample_customers.csv")
        orch = MLPipelineOrchestrator(output_dir=str(tmp_path))
        report_path = orch.run(sample)
        assert os.path.exists(report_path)
        assert report_path.endswith(".docx")

    def test_report_not_empty(self, tmp_path):
        sample = os.path.join(os.path.dirname(__file__), "..", "data", "sample_customers.csv")
        orch = MLPipelineOrchestrator(output_dir=str(tmp_path))
        report_path = orch.run(sample)
        assert os.path.getsize(report_path) > 1000
