"""Tests for the Streamlit app layer (app.py)."""

import os

from streamlit.testing.v1 import AppTest

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(REPO_DIR, "app.py")


class TestApp:
    def test_boots_and_autoloads_the_sample(self):
        """A first visit runs the sample pipeline with no clicks.

        Regression: the app used to open on a bare 'upload a file' prompt and
        hid its results behind a sidebar button a first visitor never found.
        Now the bundled sample runs on load, so the profile header is present
        and the demo banner explains it.
        """
        at = AppTest.from_file(APP_PATH, default_timeout=300)
        at.run()

        assert not at.exception
        assert any("Predictive Analytics Agent" in t.value for t in at.title)
        assert "Data Quality" in {m.label for m in at.metric}
        assert any("Demo mode" in i.value for i in at.info)

    def test_sample_run_shows_data_quality_and_written_out_task(self):
        """The autoloaded sample renders the cleaned-up profile header.

        Guards the header fix: the metric is labelled 'Data Quality' (not a
        bare 'Quality' next to model accuracy), and the task type is written
        out in full in the subheader instead of being truncated in a metric.
        """
        at = AppTest.from_file(APP_PATH, default_timeout=300)
        at.run()

        assert not at.exception
        assert "Data Quality" in {m.label for m in at.metric}
        assert any("Classification" in s.value for s in at.subheader)
