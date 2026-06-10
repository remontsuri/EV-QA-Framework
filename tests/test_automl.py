"""Tests for AutoML pipeline."""

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.automl import AutoMLSOH, AutoMLAnomaly
from ev_qa_framework.config import FrameworkConfig


def make_telemetry_df(n_rows: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "voltage": rng.normal(400, 10, n_rows),
        "current": rng.normal(50, 5, n_rows),
        "temperature": rng.normal(35, 2, n_rows),
        "soh": np.linspace(100, 90, n_rows),
    })


class TestAutoMLSOH:
    def test_fit_returns_dict(self):
        automl = AutoMLSOH()
        df = make_telemetry_df()
        result = automl.fit(df)
        assert isinstance(result, dict)

    def test_fit_finds_best_model(self):
        automl = AutoMLSOH()
        df = make_telemetry_df()
        result = automl.fit(df)
        assert result["best_model"] is not None
        assert "best_r2" in result

    def test_predict_after_fit(self):
        automl = AutoMLSOH()
        df = make_telemetry_df()
        automl.fit(df)
        predictions = automl.predict(df)
        assert len(predictions) == len(df)

    def test_predict_before_fit_raises(self):
        automl = AutoMLSOH()
        with pytest.raises(ValueError, match="No model trained"):
            automl.predict(make_telemetry_df())

    def test_evaluate_after_fit(self):
        automl = AutoMLSOH()
        df = make_telemetry_df()
        automl.fit(df)
        metrics = automl.evaluate(df)
        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_evaluate_before_fit_raises(self):
        automl = AutoMLSOH()
        with pytest.raises(ValueError, match="No model trained"):
            automl.evaluate(make_telemetry_df())

    def test_missing_columns(self):
        automl = AutoMLSOH()
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = automl.fit(df)
        assert "error" in result

    def test_custom_config(self):
        config = FrameworkConfig()
        automl = AutoMLSOH(config=config)
        assert automl.config is config


class TestAutoMLAnomaly:
    def test_fit_returns_dict(self):
        automl = AutoMLAnomaly()
        df = make_telemetry_df()
        result = automl.fit(df)
        assert isinstance(result, dict)

    def test_fit_finds_best_model(self):
        automl = AutoMLAnomaly()
        df = make_telemetry_df()
        result = automl.fit(df)
        assert result["best_model"] is not None

    def test_predict_after_fit(self):
        automl = AutoMLAnomaly()
        df = make_telemetry_df()
        automl.fit(df)
        predictions = automl.predict(df)
        assert len(predictions) == len(df)
        # Predictions should be 1 (normal) or -1 (anomaly)
        assert all(p in [1, -1] for p in predictions)

    def test_predict_before_fit_raises(self):
        automl = AutoMLAnomaly()
        with pytest.raises(ValueError, match="No model trained"):
            automl.predict(make_telemetry_df())

    def test_score_after_fit(self):
        automl = AutoMLAnomaly()
        df = make_telemetry_df()
        automl.fit(df)
        scores = automl.score(df)
        assert len(scores) == len(df)

    def test_missing_columns(self):
        automl = AutoMLAnomaly()
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = automl.fit(df)
        assert "error" in result

    def test_custom_config(self):
        config = FrameworkConfig()
        automl = AutoMLAnomaly(config=config)
        assert automl.config is config
