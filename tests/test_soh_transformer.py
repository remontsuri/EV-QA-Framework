"""
Comprehensive test module for SOH Transformer (LSTM-Transformer hybrid).

Tests cover:
- Initialization (default and custom params)
- Model building
- Data preparation
- Training (success, empty data, invalid data)
- Prediction (success, before training, various inputs)
- Save/load model persistence
- Edge cases: NaN, single row, large dataset
- Graceful handling when TensorFlow is not installed (mocked)
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ev_qa_framework.soh_transformer import SOHTransformer

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
    """
    Create a fully mocked tensorflow module that supports the
    LSTM-Transformer hybrid architecture (Functional API).
    """
    mock_tf = types.ModuleType("tensorflow")

    # --- mock keras layers ---
    mock_layer = MagicMock()
    mock_layer.LSTM = MagicMock(return_value=MagicMock())
    mock_layer.MultiHeadAttention = MagicMock(return_value=MagicMock())
    mock_layer.Dense = MagicMock(return_value=MagicMock())
    mock_layer.Dropout = MagicMock(return_value=MagicMock())
    mock_layer.LayerNormalization = MagicMock(return_value=MagicMock())
    mock_layer.Input = MagicMock(return_value=MagicMock())
    mock_layer.GlobalAveragePooling1D = MagicMock(return_value=MagicMock())
    mock_layer.Concatenate = MagicMock(return_value=MagicMock())

    # --- mock keras Model (Functional API) ---
    mock_model_instance = MagicMock()
    mock_model_instance.compile = MagicMock()
    mock_model_instance.fit = MagicMock(return_value=MagicMock())
    mock_model_instance.save = MagicMock()
    mock_model_instance.predict = MagicMock(side_effect=lambda x, **kw: np.random.rand(len(x), 1))

    def mock_model_constructor(inputs=None, outputs=None):
        mock_model_instance.inputs = inputs
        mock_model_instance.outputs = outputs
        return mock_model_instance

    mock_Model = MagicMock(side_effect=mock_model_constructor)

    # --- mock keras models ---
    mock_models = MagicMock()
    mock_models.Model = mock_Model
    mock_models.load_model = MagicMock(
        return_value=MagicMock(predict=MagicMock(return_value=np.array([[0.5], [0.6]])))
    )

    # --- wire keras ---
    mock_keras = MagicMock()
    mock_keras.layers = mock_layer
    mock_keras.models = mock_models

    mock_tf.keras = mock_keras

    return mock_tf, mock_model_instance


def _patch_mock_tf(mock_tf):
    """Return a list of patch.dict context managers for the mock TF."""
    return [
        patch.dict(sys.modules, {"tensorflow": mock_tf}),
        patch.dict(sys.modules, {"tensorflow.keras": mock_tf.keras}),
        patch.dict(sys.modules, {"tensorflow.keras.layers": mock_tf.keras.layers}),
        patch.dict(sys.modules, {"tensorflow.keras.models": mock_tf.keras.models}),
    ]


# ---------------------------------------------------------------------------
# 1. Initialization tests
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_default_params(self):
        transformer = SOHTransformer()
        assert transformer.sequence_length == 10
        assert transformer.n_features == 3
        assert transformer.model is None
        assert transformer.is_trained is False
        assert transformer.scaler is not None

    def test_custom_sequence_length(self):
        transformer = SOHTransformer(sequence_length=5)
        assert transformer.sequence_length == 5
        assert transformer.model is None
        assert transformer.is_trained is False

    def test_custom_n_features(self):
        transformer = SOHTransformer(n_features=5)
        assert transformer.n_features == 5

    def test_custom_both_params(self):
        transformer = SOHTransformer(sequence_length=20, n_features=7)
        assert transformer.sequence_length == 20
        assert transformer.n_features == 7

    def test_scaler_is_minmax(self):
        from sklearn.preprocessing import MinMaxScaler

        transformer = SOHTransformer()
        assert isinstance(transformer.scaler, MinMaxScaler)
        assert isinstance(transformer._feature_scaler, MinMaxScaler)

    def test_large_sequence_length(self):
        transformer = SOHTransformer(sequence_length=500)
        assert transformer.sequence_length == 500


# ---------------------------------------------------------------------------
# 2. build_model tests (mocked TF)
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_build_model_returns_model(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=10, n_features=3)
            model = transformer.build_model()
            assert model is not None

    def test_build_model_compiles_with_mae(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=10, n_features=3)
            transformer.build_model()
            mock_model_instance.compile.assert_called_once()
            call_kwargs = mock_model_instance.compile.call_args
            assert call_kwargs[1]["loss"] == "mae"

    def test_build_model_uses_adam(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=10, n_features=3)
            transformer.build_model()
            call_kwargs = mock_model_instance.compile.call_args
            # optimizer="adam" — check it's passed
            assert "adam" in str(call_kwargs[1]["optimizer"])


# ---------------------------------------------------------------------------
# 3. prepare_data tests
# ---------------------------------------------------------------------------


class TestPrepareData:
    def test_output_shapes(self):
        transformer = SOHTransformer(sequence_length=5)
        df = make_dataframe(20)
        x, y = transformer.prepare_data(df)
        assert x.shape == (15, 5, 3)
        assert y.shape == (15,)

    def test_output_types(self):
        transformer = SOHTransformer(sequence_length=5)
        df = make_dataframe(20)
        x, y = transformer.prepare_data(df)
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_custom_target_col(self):
        transformer = SOHTransformer(sequence_length=5)
        df = make_dataframe(20)
        df = df.rename(columns={"soh": "health"})
        x, y = transformer.prepare_data(df, target_col="health")
        assert x.shape == (15, 5, 3)
        assert y.shape == (15,)

    def test_prepare_data_too_few_rows(self):
        transformer = SOHTransformer(sequence_length=10)
        df = make_dataframe(5)
        x, y = transformer.prepare_data(df)
        assert len(x) == 0
        assert len(y) == 0

    def test_prepare_data_exact_sequence_length(self):
        transformer = SOHTransformer(sequence_length=10)
        df = make_dataframe(10)
        x, y = transformer.prepare_data(df)
        assert len(x) == 0

    def test_prepare_data_one_more_than_sequence_length(self):
        transformer = SOHTransformer(sequence_length=10)
        df = make_dataframe(11)
        x, y = transformer.prepare_data(df)
        assert len(x) == 1
        assert x.shape == (1, 10, 3)

    def test_prepare_data_deterministic(self):
        transformer = SOHTransformer(sequence_length=5)
        df = make_dataframe(20)
        x1, y1 = transformer.prepare_data(df.copy())
        x2, y2 = transformer.prepare_data(df.copy())
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(y1, y2)

    def test_prepare_data_with_nan(self):
        transformer = SOHTransformer(sequence_length=5)
        df = make_dataframe(20)
        df.loc[0, "voltage"] = np.nan
        x, y = transformer.prepare_data(df)
        assert np.isnan(x).any() or np.isnan(y).any()


# ---------------------------------------------------------------------------
# 4. Training tests (mocked TF)
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_success(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df, epochs=2, batch_size=10)

            assert transformer.is_trained is True
            assert transformer.model is not None
            mock_model_instance.compile.assert_called_once()
            mock_model_instance.fit.assert_called_once()

    def test_train_returns_history(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            history = transformer.train(df, epochs=2, batch_size=10)
            assert history is not None

    def test_train_custom_epochs_and_batch(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df, epochs=5, batch_size=16)

            call_kwargs = mock_model_instance.fit.call_args
            assert call_kwargs[1]["epochs"] == 5
            assert call_kwargs[1]["batch_size"] == 16

    def test_train_default_epochs(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df)

            call_kwargs = mock_model_instance.fit.call_args
            assert call_kwargs[1]["epochs"] == 20

    def test_train_empty_dataframe_raises(self):
        transformer = SOHTransformer(sequence_length=10)
        df = make_dataframe(5)
        with pytest.raises(ValueError, match="Not enough data"):
            transformer.train(df)

    def test_train_missing_columns_raises(self):
        transformer = SOHTransformer(sequence_length=5)
        df = pd.DataFrame({"voltage": [1, 2, 3], "current": [4, 5, 6]})
        with pytest.raises(KeyError):
            transformer.train(df)

    def test_train_minimum_viable_data(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(6)
            transformer.train(df, epochs=1, batch_size=1)
            assert transformer.is_trained is True


# ---------------------------------------------------------------------------
# 5. Prediction tests
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_before_training_raises(self):
        transformer = SOHTransformer(sequence_length=5)
        df = make_dataframe(10)
        with pytest.raises(ValueError, match="not trained"):
            transformer.predict(df)

    def test_predict_insufficient_data_raises(self):
        transformer = SOHTransformer(sequence_length=10)
        transformer.is_trained = True
        transformer.model = MagicMock()
        df = make_dataframe(5)
        with pytest.raises(ValueError, match="Need at least"):
            transformer.predict(df)

    def test_predict_returns_array(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df, epochs=1, batch_size=10)

            result = transformer.predict(df)
            assert isinstance(result, np.ndarray)

    def test_predict_output_length(self):
        """With N rows and seq_len=L, predict returns N-L+1 values."""
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(20)
            transformer.train(df, epochs=1, batch_size=10)

            result = transformer.predict(df)
            # 20 rows, seq_len=5 → 20-5+1 = 16 predictions
            assert len(result) == 16

    def test_predict_exact_sequence_length(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df_train = make_dataframe(50)
            transformer.train(df_train, epochs=1, batch_size=10)

            # Predict on exactly sequence_length rows
            result = transformer.predict(df_train.tail(5))
            assert isinstance(result, np.ndarray)
            assert len(result) == 1

    def test_predict_missing_columns_raises(self):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df, epochs=1, batch_size=10)

            bad_input = pd.DataFrame({"voltage": [1, 2, 3, 4, 5]})
            with pytest.raises(KeyError):
                transformer.predict(bad_input)


# ---------------------------------------------------------------------------
# 6. Save / Load tests
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_creates_files(self, tmp_path):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df, epochs=1, batch_size=10)

            model_dir = str(tmp_path / "model")
            transformer.save(model_dir)

            mock_model_instance.save.assert_called_once()
            assert os.path.isdir(model_dir)

    def test_save_untrained_model_no_op(self, tmp_path):
        transformer = SOHTransformer(sequence_length=5)
        model_dir = str(tmp_path / "untrained")
        transformer.save(model_dir)
        assert not os.path.exists(os.path.join(model_dir, "soh_transformer.keras"))

    def test_save_creates_directory(self, tmp_path):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df, epochs=1, batch_size=10)

            nested_dir = str(tmp_path / "deep" / "nested" / "dir")
            transformer.save(nested_dir)
            assert os.path.isdir(nested_dir)

    def test_load_restores_state(self, tmp_path):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df, epochs=1, batch_size=10)

            model_dir = str(tmp_path / "model")
            transformer.save(model_dir)

            loaded_model = MagicMock()
            loaded_model.predict = MagicMock(return_value=np.array([[0.5], [0.6]]))
            mock_tf.keras.models.load_model = MagicMock(return_value=loaded_model)

            with patch("ev_qa_framework.soh_transformer.joblib.load") as mock_joblib_load:
                from sklearn.preprocessing import MinMaxScaler

                fitted_scaler = MinMaxScaler()
                fitted_scaler.fit([[400, 100, 35], [300, 50, 20]])
                mock_joblib_load.side_effect = [fitted_scaler, fitted_scaler]

                new_transformer = SOHTransformer(sequence_length=5)
                new_transformer.load(model_dir)

                assert new_transformer.is_trained is True
                assert new_transformer.model is not None

    def test_save_load_predict_roundtrip(self, tmp_path):
        mock_tf, mock_model_instance = _make_mock_tensorflow()
        patches = _patch_mock_tf(mock_tf)

        with patches[0], patches[1], patches[2], patches[3]:
            transformer = SOHTransformer(sequence_length=5)
            df = make_dataframe(50)
            transformer.train(df, epochs=1, batch_size=10)

            model_dir = str(tmp_path / "roundtrip")
            transformer.save(model_dir)

            loaded_model = MagicMock()
            loaded_model.predict = MagicMock(return_value=np.array([[0.5], [0.6], [0.7]]))
            mock_tf.keras.models.load_model = MagicMock(return_value=loaded_model)

            with patch("ev_qa_framework.soh_transformer.joblib.load") as mock_joblib_load:
                from sklearn.preprocessing import MinMaxScaler

                full_scaler = MinMaxScaler()
                full_scaler.fit([[400, 100, 35, 95], [300, 50, 20, 80]])
                feat_scaler = MinMaxScaler()
                feat_scaler.fit([[400, 100, 35], [300, 50, 20]])
                mock_joblib_load.side_effect = [full_scaler, feat_scaler]

                new_transformer = SOHTransformer(sequence_length=5)
                new_transformer.load(model_dir)

                result = new_transformer.predict(df)
                assert isinstance(result, np.ndarray)
                assert len(result) > 0


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_row_dataframe(self):
        transformer = SOHTransformer(sequence_length=5)
        df = make_dataframe(1)
        x, y = transformer.prepare_data(df)
        assert len(x) == 0
        assert len(y) == 0

    def test_two_row_dataframe(self):
        transformer = SOHTransformer(sequence_length=1)
        df = make_dataframe(2)
        x, y = transformer.prepare_data(df)
        assert len(x) == 1
        assert x.shape == (1, 1, 3)

    def test_large_dataset(self):
        transformer = SOHTransformer(sequence_length=10)
        df = make_dataframe(10000)
        x, y = transformer.prepare_data(df)
        assert x.shape == (9990, 10, 3)
        assert y.shape == (9990,)

    def test_constant_values(self):
        transformer = SOHTransformer(sequence_length=5)
        n = 20
        df = pd.DataFrame(
            {
                "voltage": np.full(n, 400.0),
                "current": np.full(n, 100.0),
                "temperature": np.full(n, 35.0),
                "soh": np.linspace(100, 95, n),
            }
        )
        x, y = transformer.prepare_data(df)
        assert x.shape == (15, 5, 3)
        assert not np.isnan(x).any()

    def test_negative_values(self):
        transformer = SOHTransformer(sequence_length=5)
        rng = np.random.default_rng(99)
        n = 30
        df = pd.DataFrame(
            {
                "voltage": rng.normal(400, 10, n),
                "current": rng.normal(-50, 5, n),  # negative current
                "temperature": rng.normal(35, 2, n),
                "soh": np.linspace(100, 95, n),
            }
        )
        x, y = transformer.prepare_data(df)
        assert x.shape == (25, 5, 3)
        assert not np.isnan(x).any()
