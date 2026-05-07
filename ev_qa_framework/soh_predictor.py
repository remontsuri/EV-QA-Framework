"""
SOH Predictor Module: LSTM-based battery health degradation forecasting.

Note: TensorFlow is an optional dependency. Without it, SOHPredictor
will raise an ImportError only when instantiated, not at import time.
"""
from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib


def _import_tensorflow():
    """Lazy-import tensorflow to avoid hard dependency at module level."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ImportError(
            "TensorFlow is required for SOHPredictor. "
            "Install it via: pip install tensorflow>=2.15"
        )


class SOHPredictor:
    """
    LSTM-based predictor for Battery State of Health (SOH) degradation.

    Uses historical time-series data (voltage, current, temperature)
    to predict future SOH.
    """

    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.model: Optional[Sequential] = None
        self.scaler = MinMaxScaler()
        self.is_trained = False

    def _build_model(self, input_shape: Tuple[int, int]):
        tf = _import_tensorflow()
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        model = Sequential([
            LSTM(64, activation='relu', input_shape=input_shape,
                 return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, df: pd.DataFrame,
                     target_col: str = 'soh') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time-series sequences for LSTM.
        """
        features = ['voltage', 'current', 'temperature']
        data = df[features + [target_col]].values

        # Scale data
        scaled_data = self.scaler.fit_transform(data)

        x_seq, y_seq = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            x_seq.append(scaled_data[i:i + self.sequence_length, :-1])
            y_seq.append(scaled_data[i + self.sequence_length, -1])

        return np.array(x_seq), np.array(y_seq)

    def train(self, df: pd.DataFrame, epochs: int = 10, batch_size: int = 32):
        """
        Train the LSTM model on historical data.
        """
        x_train, y_train = self.prepare_data(df)
        if len(x_train) == 0:
            raise ValueError("Not enough data to create sequences for LSTM")

        self.model = self._build_model((x_train.shape[1], x_train.shape[2]))
        self.model.fit(x_train, y_train, epochs=epochs,
                       batch_size=batch_size, verbose=0)
        self.is_trained = True

    def predict_next(self, recent_telemetry: pd.DataFrame) -> float:
        """
        Predict the next SOH value based on recent telemetry.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")

        features = ['voltage', 'current', 'temperature']

        if len(recent_telemetry) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} "
                             "data points")

        data = recent_telemetry[features].values[-self.sequence_length:]

        # Let's fix the scaler logic in a real implementation
        scaled_feat = self.scaler.transform(
            np.hstack([data, np.zeros((len(data), 1))])
        )[:, :-1]

        x_input = np.expand_dims(scaled_feat, axis=0)
        prediction_scaled = self.model.predict(x_input, verbose=0)

        # Inverse scale prediction
        # Create a dummy array for inverse transform
        dummy = np.zeros((1, 4))
        dummy[0, -1] = prediction_scaled[0, 0]
        prediction = self.scaler.inverse_transform(dummy)[0, -1]

        return float(prediction)

    def save(self, path: str):
        """Save the model and scaler"""
        if not self.is_trained or self.model is None:
            return
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "soh_lstm.keras"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))

    def load(self, path: str):
        """Load the model and scaler"""
        from tensorflow.keras.models import load_model
        self.model = load_model(os.path.join(path, "soh_lstm.keras"))
        self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
        self.is_trained = True
