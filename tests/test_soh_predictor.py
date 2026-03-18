import pandas as pd
import numpy as np
from ev_qa_framework.soh_predictor import SOHPredictor


def test_soh_predictor_training():
    """Test LSTM training"""
    predictor = SOHPredictor(sequence_length=5)

    # Generate mock data
    data = {
        'voltage': np.random.normal(400, 10, 100),
        'current': np.random.normal(100, 5, 100),
        'temperature': np.random.normal(35, 2, 100),
        'soh': np.linspace(100, 95, 100)
    }
    df = pd.DataFrame(data)

    predictor.train(df, epochs=2, batch_size=10)
    assert predictor.is_trained
    assert predictor.model is not None


def test_soh_predictor_inference():
    """Test LSTM inference"""
    predictor = SOHPredictor(sequence_length=5)
    data = {
        'voltage': np.random.normal(400, 10, 50),
        'current': np.random.normal(100, 5, 50),
        'temperature': np.random.normal(35, 2, 50),
        'soh': np.linspace(100, 98, 50)
    }
    df = pd.DataFrame(data)
    predictor.train(df, epochs=2, batch_size=10)

    recent_data = df.tail(5)
    prediction = predictor.predict_next(recent_data)

    assert isinstance(prediction, float)
    assert 0 <= prediction <= 100


def test_soh_predictor_persistence(tmp_path):
    """Test saving and loading the model"""
    predictor = SOHPredictor(sequence_length=5)
    data = {
        'voltage': np.random.normal(400, 10, 50),
        'current': np.random.normal(100, 5, 50),
        'temperature': np.random.normal(35, 2, 50),
        'soh': np.linspace(100, 98, 50)
    }
    df = pd.DataFrame(data)
    predictor.train(df, epochs=1, batch_size=10)

    model_dir = tmp_path / "model"
    predictor.save(str(model_dir))

    new_predictor = SOHPredictor(sequence_length=5)
    new_predictor.load(str(model_dir))

    assert new_predictor.is_trained
    assert new_predictor.model is not None

    recent_data = df.tail(5)
    prediction = new_predictor.predict_next(recent_data)
    assert isinstance(prediction, float)
