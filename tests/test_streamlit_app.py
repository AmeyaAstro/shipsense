"""The demo UI must render, and must show a friendly error (not crash) when the API is down."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def test_app_renders_and_handles_api_down(monkeypatch):
    monkeypatch.setenv("SHIPSENSE_API_URL", "http://127.0.0.1:9")  # nothing listens on port 9
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Risk prediction", "Ask the copilot", "Drift"]
    assert "Can't reach the API" in at.sidebar.error[0].value
    at.button[0].click().run()  # "Predict risk" with the API down
    assert not at.exception and "Can't reach the API" in at.error[0].value
