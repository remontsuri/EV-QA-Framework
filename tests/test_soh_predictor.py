"""
Comprehensive test module for SOH Predictor.

Tests cover:
- Initialization (default and custom params)
- Data preparation
- Training (success, empty data, invalid data)
- Prediction (success, before training, various inputs)
- Save/load model persistence
- Edge cases: NaN, single row, large dataset
- Graceful handling when TensorFlow is not installed (mocked)
"""

import importlib
import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.soh_predictor import SOHPredictor, _import_tensorflow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_dataframe(n_rows: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic telemetry DataFrame for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "voltage": rng.normal(400, 10, n_rows),
            "current": rng.normal(100, 5, n_rows),
            "temperature": rng.normal(35, 2, n_rows),
            "soh": np.linspace(100, 95, n_rows),
        }
    )


def _make_mock_tensorflow():
    """Create a fully mocked tensorflow module with keras Sequential support."""
    mock_tf = types.ModuleType("tensorflow")

    # --- mock keras layers ---
    mock_layer = MagicMock()
    mock_layer.LSTM = MagicMock()
    mock_layer.Dense = MagicMock()
    mock_layer.Dropout = MagicMock()

    # --- mock keras models ---
    mock_sequential_instance = MagicMock()
    mock_sequential_instance.compile = MagicMock()
    mock_sequential_instance.fit = MagicMock()
    mock_sequential_instance.save = MagicMock()
    mock_sequential_instance.predict = MagicMock(return_value=np.array([[0.5]]))

    mock_sequential_class = MagicMock(return_value=mock_sequential_instance)

    mock_models = MagicMock()
    mock_models.Sequential = mock_sequential_class
    mock_models.load_model = MagicMock(
        return_value=MagicMock(predict=MagicMock(return_value=np.array([[0.5]])))
    )

    # --- wire keras ---
    mock_keras = MagicMock()
    mock_keras.layers = mock_layer
    mock_keras.models = mock_models

    mock_tf.keras = mock_keras

    return mock_tf, mock_sequential_instance


# ---------------------------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_default_params(self):
        predictor = SOHPredictor()
        assert predictor.sequence_length == 10
        assert predictor.model is None
        assert predictor.is_trained is False
        assert predictor.scaler is not None

    def test_custom_sequence_length(self):
        predictor = SOHPredictor(sequence_length=5)
        assert predictor.sequence_length == 5
        assert predictor.model is None
        assert predictor.is_trained is False

    def test_custom_sequence_length_large(self):
        predictor = SOHPredictor(sequence_length=100)
        assert predictor.sequence_length == 100

    def test_scaler_is_minmax(self):
        from sklearn.preprocessing import MinMaxScaler

        predictor = SOHPredictor()
        assert isinstance(predictor.scaler, MinMaxScaler)


# ---------------------------------------------------------------------------
# 2. prepare_data tests
# ---------------------------------------------------------------------------


class TestPrepareData:
    def test_output_shapes(self):
        predictor = SOHPredictor(sequence_length=5)
        df = make_dataframe(20)
        x, y = predictor.prepare_data(df)
        # With 20 rows and seq_len=5, we get 20-5=15 sequences
        assert x.shape == (15, 5, 3)
        assert y.shape == (15,)

    def test_output_types(self):
        predictor = SOHPredictor(sequence_length=5)
        df = make_dataframe(20)
        x, y = predictor.prepare_data(df)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_custom_target_col(self):
        predictor = SOHPredictor(sequence_length=5)
        df = make_dataframe(20)
        df = df.rename(columns={"soh": "health"})
        x, y = predictor.prepare_data(df, target_col="health")
        assert x.shape == (15, 5, 3)
        assert y.shape == (15,)

    def test_prepare_data_too_few_rows(self):
        """With fewer rows than sequence_length, prepare_data returns empty arrays."""
        predictor = SOHPredictor(sequence_length=10)
        df = make_dataframe(5)
        x, y = predictor.prepare_data(df)
        assert len(x) == 0
        assert len(y) == 0

    def test_prepare_data_exact_sequence_length(self):
        """Exactly sequence_length rows → 0 sequences (need seq_len+1)."""
        predictor = SOHPredictor(sequence_length=10)
        df = make_dataframe(10)
        x, y = predictor.prepare_data(df)
        assert len(x) == 0

    def test_prepare_data_one_more_than_sequence_length(self):
        """sequence_length+1 rows → exactly 1 sequence."""
        predictor = SOHPredictor(sequence_length=10)
        df = make_dataframe(11)
        x, y = predictor.prepare_data(df)
        assert len(x) == 1
        assert x.shape == (1, 10, 3)

    def test_prepare_data_with_nan(self):
        """NaN in input data propagates into scaled output."""
        predictor = SOHPredictor(sequence_length=5)
        df = make_dataframe(20)
        df.loc[0, "voltage"] = np.nan
        x, y = predictor.prepare_data(df)
        assert np.isnan(x).any() or np.isnan(y).any()

    def test_prepare_data_deterministic(self):
        """Same input → same output."""
        predictor = SOHPredictor(sequence_length=5)
        df = make_dataframe(20)
        x1, y1 = predictor.prepare_data(df.copy())
        x2, y2 = predictor.prepare_data(df.copy())
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)


# ---------------------------------------------------------------------------
# 3. Training tests (mocked TensorFlow)
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_success(self):
        """Successful training sets is_trained and creates a model."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=2, batch_size=10)

                        assert predictor.is_trained is True
                        assert predictor.model is not None
                        mock_model_instance.compile.assert_called_once()
                        mock_model_instance.fit.assert_called_once()

    def test_train_custom_epochs_and_batch(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=5, batch_size=16)

                        # Verify fit was called with the right params
                        call_kwargs = mock_model_instance.fit.call_args
                        assert call_kwargs[1]["epochs"] == 5
                        assert call_kwargs[1]["batch_size"] == 16

    def test_train_empty_dataframe_raises(self):
        """Training on a DataFrame with too few rows raises ValueError."""
        predictor = SOHPredictor(sequence_length=10)
        df = make_dataframe(5)  # Not enough for seq_len=10
        with pytest.raises(ValueError, match="Not enough data"):
            predictor.train(df)

    def test_train_missing_columns_raises(self):
        """Training on DataFrame missing required columns raises KeyError."""
        predictor = SOHPredictor(sequence_length=5)
        df = pd.DataFrame({"voltage": [1, 2, 3], "current": [4, 5, 6]})
        with pytest.raises(KeyError):
            predictor.train(df)

    def test_train_single_row_above_sequence_length(self):
        """Minimum viable training data: sequence_length + 1 rows."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(6)  # Exactly seq_len + 1
                        predictor.train(df, epochs=1, batch_size=1)
                        assert predictor.is_trained is True


# ---------------------------------------------------------------------------
# 4. Prediction tests
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_before_training_raises(self):
        """Calling predict before training raises ValueError."""
        predictor = SOHPredictor(sequence_length=5)
        df = make_dataframe(10)
        with pytest.raises(ValueError, match="not trained"):
            predictor.predict_next(df)

    def test_predict_insufficient_data_raises(self):
        """Calling predict with fewer rows than sequence_length raises ValueError."""
        predictor = SOHPredictor(sequence_length=10)
        predictor.is_trained = True
        predictor.model = MagicMock()
        df = make_dataframe(5)
        with pytest.raises(ValueError, match="Need at least"):
            predictor.predict_next(df)

    def test_predict_returns_float(self):
        """predict_next returns a float value."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)

                        recent = df.tail(5)
                        result = predictor.predict_next(recent)
                        assert isinstance(result, float)

    def test_predict_with_exact_sequence_length(self):
        """Predict works when input has exactly sequence_length rows."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)

                        recent = df.tail(5)  # Exactly sequence_length
                        result = predictor.predict_next(recent)
                        assert isinstance(result, float)

    def test_predict_with_more_than_sequence_length(self):
        """Predict works when input has more than sequence_length rows (uses last N)."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)

                        recent = df.tail(15)  # More than sequence_length
                        result = predictor.predict_next(recent)
                        assert isinstance(result, float)

    def test_predict_missing_columns_raises(self):
        """Predict with missing feature columns raises KeyError."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)

                        bad_input = pd.DataFrame({"voltage": [1, 2, 3, 4, 5]})
                        with pytest.raises(KeyError):
                            predictor.predict_next(bad_input)


# ---------------------------------------------------------------------------
# 5. Save / Load tests
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_files(self, tmp_path):
        """Save creates the model and scaler files."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)

                        model_dir = str(tmp_path / "model")
                        predictor.save(model_dir)

                        # Verify model.save() was called
                        mock_model_instance.save.assert_called_once()
                        # Verify directory was created
                        assert os.path.isdir(model_dir)

    def test_save_untrained_model_no_op(self, tmp_path):
        """Saving an untrained model does nothing (no model files created)."""
        predictor = SOHPredictor(sequence_length=5)
        model_dir = str(tmp_path / "untrained")
        predictor.save(model_dir)
        assert not os.path.exists(os.path.join(model_dir, "soh_lstm.keras"))
        assert not os.path.exists(os.path.join(model_dir, "feature_scaler.joblib"))

    def test_save_creates_directory(self, tmp_path):
        """Save creates the directory if it doesn't exist."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)

                        nested_dir = str(tmp_path / "deep" / "nested" / "dir")
                        predictor.save(nested_dir)
                        assert os.path.isdir(nested_dir)

    def test_load_restores_state(self, tmp_path):
        """Loading a saved model restores is_trained and model."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)

                        model_dir = str(tmp_path / "model")
                        predictor.save(model_dir)

                        # Mock load_model and joblib.load for the load call
                        loaded_model = MagicMock()
                        loaded_model.predict = MagicMock(return_value=np.array([[0.5]]))
                        mock_tf.keras.models.load_model = MagicMock(return_value=loaded_model)

                        with patch("ev_qa_framework.soh_predictor.joblib.load") as mock_joblib_load:
                            from sklearn.preprocessing import MinMaxScaler

                            fitted_scaler = MinMaxScaler()
                            fitted_scaler.fit([[400, 100, 35], [300, 50, 20]])
                            mock_joblib_load.side_effect = [fitted_scaler, fitted_scaler]

                            new_predictor = SOHPredictor(sequence_length=5)
                            new_predictor.load(model_dir)

                            assert new_predictor.is_trained is True
                            assert new_predictor.model is not None

    def test_save_load_predict_roundtrip(self, tmp_path):
        """Full roundtrip: train → save → load → predict."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)

                        model_dir = str(tmp_path / "roundtrip")
                        predictor.save(model_dir)

                        # Now load into a new predictor
                        loaded_model = MagicMock()
                        loaded_model.predict = MagicMock(return_value=np.array([[0.5]]))
                        mock_tf.keras.models.load_model = MagicMock(return_value=loaded_model)

                        with patch("ev_qa_framework.soh_predictor.joblib.load") as mock_joblib_load:
                            from sklearn.preprocessing import MinMaxScaler

                            # Full scaler (4 cols: 3 features + target) and feature scaler (3 cols)
                            full_scaler = MinMaxScaler()
                            full_scaler.fit([[400, 100, 35, 95], [300, 50, 20, 80]])
                            feat_scaler = MinMaxScaler()
                            feat_scaler.fit([[400, 100, 35], [300, 50, 20]])
                            mock_joblib_load.side_effect = [full_scaler, feat_scaler]

                            new_predictor = SOHPredictor(sequence_length=5)
                            new_predictor.load(model_dir)

                            recent = df.tail(5)
                            result = new_predictor.predict_next(recent)
                            assert isinstance(result, float)


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row_dataframe(self):
        """A single-row DataFrame cannot produce sequences."""
        predictor = SOHPredictor(sequence_length=5)
        df = make_dataframe(1)
        x, y = predictor.prepare_data(df)
        assert len(x) == 0
        assert len(y) == 0

    def test_two_row_dataframe(self):
        """Two rows with seq_len=1 produces 1 sequence."""
        predictor = SOHPredictor(sequence_length=1)
        df = make_dataframe(2)
        x, y = predictor.prepare_data(df)
        assert len(x) == 1
        assert x.shape == (1, 1, 3)

    def test_large_dataset(self):
        """Large dataset (10000 rows) produces correct shapes."""
        predictor = SOHPredictor(sequence_length=10)
        df = make_dataframe(10000)
        x, y = predictor.prepare_data(df)
        assert x.shape == (9990, 10, 3)
        assert y.shape == (9990,)

    def test_constant_values(self):
        """All-constant feature values (no variance) — scaler handles it."""
        predictor = SOHPredictor(sequence_length=5)
        n = 20
        df = pd.DataFrame(
            {
                "voltage": np.full(n, 400.0),
                "current": np.full(n, 100.0),
                "temperature": np.full(n, 35.0),
                "soh": np.linspace(100, 95, n),
            }
        )
        x, y = predictor.prepare_data(df)
        assert x.shape == (15, 5, 3)
        # With constant features, MinMaxScaler produces 0s
        assert not np.isnan(x).any()  # Should not crash

    def test_negative_values(self):
        """Negative values in telemetry are handled."""
        predictor = SOHPredictor(sequence_length=5)
        rng = np.random.default_rng(99)
        n = 30
        df = pd.DataFrame(
            {
                "voltage": rng.normal(-10, 5, n),
                "current": rng.normal(-50, 10, n),
                "temperature": rng.normal(-5, 3, n),
                "soh": np.linspace(100, 90, n),
            }
        )
        x, y = predictor.prepare_data(df)
        assert x.shape == (25, 5, 3)

    def test_predict_with_empty_dataframe_raises(self):
        """Predict with empty DataFrame raises ValueError (not enough data)."""
        predictor = SOHPredictor(sequence_length=5)
        predictor.is_trained = True
        predictor.model = MagicMock()
        empty_df = pd.DataFrame(columns=["voltage", "current", "temperature"])
        with pytest.raises(ValueError):
            predictor.predict_next(empty_df)

    def test_sequence_length_one(self):
        """sequence_length=1 is a valid edge case."""
        predictor = SOHPredictor(sequence_length=1)
        df = make_dataframe(10)
        x, y = predictor.prepare_data(df)
        assert x.shape == (9, 1, 3)
        assert y.shape == (9,)

    def test_train_after_already_trained(self):
        """Training again overwrites the previous model."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        mock_tf.keras.models.Sequential.return_value = mock_model_instance

        with patch.dict(sys.modules, {"tensorflow": mock_tf}):
            with patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}):
                with patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}):
                    with patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}):
                        predictor = SOHPredictor(sequence_length=5)
                        df = make_dataframe(50)
                        predictor.train(df, epochs=1, batch_size=10)
                        first_model = predictor.model

                        # Reset mock for second call
                        mock_model_instance2 = MagicMock()
                        mock_tf.keras.models.Sequential.return_value = mock_model_instance2

                        predictor.train(df, epochs=1, batch_size=10)
                        second_model = predictor.model
                        # Model should be replaced
                        assert first_model is not second_model


# ---------------------------------------------------------------------------
# 7. TensorFlow import error handling
# ---------------------------------------------------------------------------


class TestTensorFlowImport:
    def test_import_error_raised_on_build_model(self):
        """_build_model raises ImportError when TF not installed."""
        predictor = SOHPredictor(sequence_length=5)
        # Remove tensorflow from sys.modules to simulate it not being installed
        saved = sys.modules.pop("tensorflow", None)
        saved_keras = sys.modules.pop("tensorflow.keras", None)
        saved_layers = sys.modules.pop("tensorflow.keras.layers", None)
        saved_models = sys.modules.pop("tensorflow.keras.models", None)
        try:
            with pytest.raises(ImportError, match="TensorFlow is required for SOHPredictor"):
                predictor._build_model((5, 3))
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved
            if saved_keras is not None:
                sys.modules["tensorflow.keras"] = saved_keras
            if saved_layers is not None:
                sys.modules["tensorflow.keras.layers"] = saved_layers
            if saved_models is not None:
                sys.modules["tensorflow.keras.models"] = saved_models

    def test_lazy_import_at_module_level(self):
        """Module-level import should NOT raise even without TF."""
        importlib.reload(sys.modules["ev_qa_framework.soh_predictor"])
        # If we get here, no ImportError at module level


# ---------------------------------------------------------------------------
# 8. _import_tensorflow function tests
# ---------------------------------------------------------------------------


    def test_returns_module_when_available(self):
        """When TF is available, _import_tensorflow returns the module."""
        mock_tf = MagicMock()
        saved = sys.modules.get("tensorflow")
        sys.modules["tensorflow"] = mock_tf
        try:
            result = _import_tensorflow()
            assert result is mock_tf
        finally:
            if saved is not None:
                sys.modules["tensorflow"] = saved
            else:
                sys.modules.pop("tensorflow", None)

    def test_raises_when_not_available(self):
        """When TF is not available, _import_tensorflow raises ImportError."""
        saved = sys.modules.pop("tensorflow", None)
        saved_keras = sys.modules.pop("tensorflow.keras", None)
        saved_layers = sys.modules.pop("tensorflow.keras.layers", None)
        saved_models = sys.modules.pop("tensorflow.keras.models", None)
        old_env = os.environ.pop("_EV_SIMULATE_MISSING_TF", None)
        os.environ["_EV_SIMULATE_MISSING_TF"] = "1"
        try:
            with pytest.raises(ImportError, match="TensorFlow is required for SOHPredictor"):
                _import_tensorflow()
        finally:
            if old_env is not None:
                os.environ["_EV_SIMULATE_MISSING_TF"] = old_env
            elif "_EV_SIMULATE_MISSING_TF" in os.environ:
                del os.environ["_EV_SIMULATE_MISSING_TF"]
            if saved is not None:
                sys.modules["tensorflow"] = saved
            if saved_keras is not None:
                sys.modules["tensorflow.keras"] = saved_keras
            if saved_layers is not None:
                sys.modules["tensorflow.keras.layers"] = saved_layers
            if saved_models is not None:
                sys.modules["tensorflow.keras.models"] = saved_models

