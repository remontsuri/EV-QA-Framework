"""
SOH Transformer Module: LSTM-Transformer hybrid for battery health prediction.

Architecture:
  - LSTM layer (64 units) captures temporal dependencies
  - Multi-Head Self-Attention (4 heads) captures long-range dependencies
  - Feed-Forward Network (Dense 16) with LayerNorm and Dropout
  - Loss: MAE (more robust to outliers than MSE)

Note: TensorFlow is an optional dependency. Without it, SOHTransformer
will raise an ImportError only when model methods are called, not at import time.
"""
from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def _import_tensorflow():
    """Lazy-import tensorflow to avoid hard dependency at module level."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for SOHTransformer. "
            "Install it via: pip install tensorflow>=2.15"
        )


class SOHTransformer:
    """
    LSTM-Transformer hybrid for Battery State of Health (SOH) prediction.

    The model first processes input sequences through an LSTM layer to capture
    local temporal patterns, then applies multi-head self-attention to model
    long-range dependencies across the sequence.

    Parameters
    ----------
    sequence_length : int
        Number of time steps in each input sequence.
    n_features : int
        Number of input features (default 3: voltage, current, temperature).
    """

    def __init__(self, sequence_length: int = 10, n_features: int = 3):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model: Any | None = None
        self.scaler = MinMaxScaler()
        self._feature_scaler = MinMaxScaler()
        self.is_trained = False

    def build_model(self):
        """
        Build the LSTM-Transformer hybrid Keras model.

        Architecture:
            Input → LSTM(64, return_sequences=True)
                  → MultiHeadAttention(4 heads)
                  → LayerNorm
                  → Dense(16, relu)
                  → Dropout(0.2)
                  → GlobalAveragePooling1D
                  → Dense(1)

        Returns
        -------
        tensorflow.keras.Model
            Compiled Keras model with MAE loss.
        """
        tf = _import_tensorflow()
        from tensorflow.keras.layers import (
            LSTM,
            MultiHeadAttention,
            Dense,
            Dropout,
            LayerNormalization,
            Input,
            GlobalAveragePooling1D,
        )
        from tensorflow.keras.models import Model

        # Input: (batch, sequence_length, n_features)
        inputs = Input(shape=(self.sequence_length, self.n_features))

        # LSTM branch — captures temporal dependencies
        x = LSTM(64, return_sequences=True)(inputs)

        # Multi-Head Self-Attention — captures long-range dependencies
        attn_output = MultiHeadAttention(
            num_heads=4, key_dim=16
        )(x, x)
        # Residual connection + LayerNorm
        x = LayerNormalization()(x + attn_output)

        # Feed-Forward Network
        x = Dense(16, activation="relu")(x)
        x = Dropout(0.2)(x)

        # Pool across time dimension
        x = GlobalAveragePooling1D()(x)

        # Output: single SOH value
        outputs = Dense(1)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mae")
        return model

    def prepare_data(
        self, df: pd.DataFrame, target_col: str = "soh"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare time-series sequences for the LSTM-Transformer model.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['voltage', 'current', 'temperature', target_col].
        target_col : str
            Name of the target SOH column.

        Returns
        -------
        X : np.ndarray of shape (n_samples, sequence_length, n_features)
        y : np.ndarray of shape (n_samples,)
        """
        features = ["voltage", "current", "temperature"]
        data = df[features + [target_col]].values

        # Scale features and target together for inverse transform later
        self.scaler.fit(data)
        self._feature_scaler.fit(data[:, :-1])
        scaled_data = self.scaler.transform(data)

        x_seq, y_seq = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            x_seq.append(scaled_data[i : i + self.sequence_length, :-1])
            y_seq.append(scaled_data[i + self.sequence_length, -1])

        return np.array(x_seq), np.array(y_seq)

    def train(
        self, df: pd.DataFrame, epochs: int = 20, batch_size: int = 32
    ):
        """
        Train the LSTM-Transformer model on historical data.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with telemetry and SOH columns.
        epochs : int
            Number of training epochs.
        batch_size : int
            Training batch size.

        Returns
        -------
        history : Keras History object
        """
        x_train, y_train = self.prepare_data(df)
        if len(x_train) == 0:
            raise ValueError("Not enough data to create sequences for training")

        self.model = self.build_model()
        history = self.model.fit(
            x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0
        )
        self.is_trained = True
        return history

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict SOH values from a telemetry DataFrame.

        Each prediction uses a sliding window of `sequence_length` rows.
        Returns one prediction per valid window position.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns ['voltage', 'current', 'temperature'].

        Returns
        -------
        np.ndarray of shape (n_predictions,)
            Predicted SOH values (inverse-scaled to original units).
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")

        features = ["voltage", "current", "temperature"]
        data = df[features].values

        if len(data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} data points, "
                f"got {len(data)}"
            )

        scaled_data = self._feature_scaler.transform(data)

        # Build sliding windows
        windows = []
        for i in range(len(scaled_data) - self.sequence_length + 1):
            windows.append(scaled_data[i : i + self.sequence_length])

        x_input = np.array(windows)
        predictions_scaled = self.model.predict(x_input, verbose=0).flatten()

        # Inverse scale predictions
        dummy = np.zeros((len(predictions_scaled), 4))
        dummy[:, -1] = predictions_scaled
        predictions = self.scaler.inverse_transform(dummy)[:, -1]

        return predictions

    def save(self, path: str):
        """
        Save the model and scalers to a directory.

        Parameters
        ----------
        path : str
            Directory path to save into.
        """
        if not self.is_trained or self.model is None:
            return
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "soh_transformer.keras"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))
        joblib.dump(
            self._feature_scaler, os.path.join(path, "feature_scaler.joblib")
        )

    def load(self, path: str):
        """
        Load the model and scalers from a directory.

        Parameters
        ----------
        path : str
            Directory path to load from.
        """
        tf = _import_tensorflow()
        from tensorflow.keras.models import load_model

        self.model = load_model(
            os.path.join(path, "soh_transformer.keras")
        )
        self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
        self._feature_scaler = joblib.load(
            os.path.join(path, "feature_scaler.joblib")
        )
        self.is_trained = True
