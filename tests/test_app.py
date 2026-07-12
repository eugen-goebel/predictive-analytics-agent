"""Tests for the Streamlit app layer (app.py)."""

import os

from streamlit.testing.v1 import AppTest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(REPO_DIR, "app.py")


class TestApp:
    def test_boots_without_error(self):
        """The app starts and shows the upload prompt before any dataset."""
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()

        assert not at.exception
        assert any("Predictive Analytics Agent" in t.value for t in at.title)

    def test_sample_run_shows_data_quality_and_written_out_task(self):
        """Running the sample renders the cleaned-up profile header.

        Guards the header fix: the metric is labelled 'Data Quality' (not a
        bare 'Quality' next to model accuracy), and the task type is written
        out in full in the subheader instead of being truncated in a metric.
        """
        at = AppTest.from_file(APP_PATH, default_timeout=300)
        at.run()

        sample_btn = next(b for b in at.button if "sample dataset" in b.label.lower())
        sample_btn.click().run()

        assert not at.exception
        assert "Data Quality" in {m.label for m in at.metric}
        assert any("Classification" in s.value for s in at.subheader)
