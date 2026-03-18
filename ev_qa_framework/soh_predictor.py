from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
import os

# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib

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

    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        model = Sequential([
            LSTM(64, activation='relu', input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'soh') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare time-series sequences for LSTM.
        """
        features = ['voltage', 'current', 'temperature']
        data = df[features + [target_col]].values

        # Scale data
        scaled_data = self.scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length, :-1])
            y.append(scaled_data[i + self.sequence_length, -1])

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, epochs: int = 10, batch_size: int = 32):
        """
        Train the LSTM model on historical data.
        """
        X, y = self.prepare_data(df)
        if len(X) == 0:
            raise ValueError("Not enough data to create sequences for LSTM")

        self.model = self._build_model((X.shape[1], X.shape[2]))
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.is_trained = True

    def predict_next(self, recent_telemetry: pd.DataFrame) -> float:
        """
        Predict the next SOH value based on recent telemetry.
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet")

        features = ['voltage', 'current', 'temperature']
        # We need a dummy SOH column to use the scaler properly if we scaled everything
        # Actually, it's better to have separate scalers for features and target
        # but for simplicity in this demo we'll just use the last few rows

        if len(recent_telemetry) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")

        data = recent_telemetry[features].values[-self.sequence_length:]
        # Use a temporary scaler for features only for prediction if we don't want to mess up
        # or just use the fitted scaler's feature part.

        # Re-fitting a simplified version for inference in this demo
        feat_scaler = MinMaxScaler()
        # In a real scenario, you'd save and load the exact scaler used in training
        # For the sake of the demo, we'll assume the inputs are already somewhat normalized
        # or we use the scaler from training.

        # Scaling features part (first 3 columns of our original scaler)
        # This is a bit hacky for a demo, ideally we have self.feature_scaler and self.target_scaler

        # Let's fix the scaler logic in a real implementation
        scaled_feat = self.scaler.transform(
            np.hstack([data, np.zeros((len(data), 1))])
        )[:, :-1]

        X = np.expand_dims(scaled_feat, axis=0)
        prediction_scaled = self.model.predict(X, verbose=0)

        # Inverse scale prediction
        # Create a dummy array for inverse transform
        dummy = np.zeros((1, 4))
        dummy[0, -1] = prediction_scaled[0, 0]
        prediction = self.scaler.inverse_transform(dummy)[0, -1]

        return float(prediction)

    def save(self, path: str):
        if not self.is_trained or self.model is None:
            return
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "soh_lstm.keras"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))

    def load(self, path: str):
        self.model = load_model(os.path.join(path, "soh_lstm.keras"))
        self.scaler = joblib.load(os.path.join(path, "scaler.joblib"))
        self.is_trained = True
