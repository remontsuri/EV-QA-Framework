import sys
import asyncio
import pandas as pd
# ensure project root is in path for imports
sys.path.append(r"c:\Users\vladc\Desktop\code\EV-QA-Framework")
from ev_qa_framework.config import FrameworkConfig
from ev_qa_framework.framework import EVQAFramework

config = FrameworkConfig()
config.safety_thresholds.max_temperature = 45.0
config.default_vin = "INTEGRATION_TEST_VIN"
qa = EVQAFramework(name="Integration-QA", config=config)

# build test data
test_data = [
    {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 80, 'soh': 98},
    {'voltage': 401.0, 'current': 51, 'temperature': 31, 'soc': 79, 'soh': 98},
    {'voltage': 402.0, 'current': 52, 'temperature': 32, 'soc': 78, 'soh': 98},
    {'voltage': 400.0, 'current': 50, 'temperature': 50, 'soc': 77, 'soh': 98},
    {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 76, 'soh': 98},
    {'voltage': 400.0, 'current': 50, 'temperature': 40, 'soc': 75, 'soh': 98},
]

# inspect telemetries objects and anomalies
from ev_qa_framework.models import BatteryTelemetryModel
# replicate run_test_suite VIN injection
telemetries = []
for d in test_data:
    data = d.copy()
    if 'vin' not in data:
        data['vin'] = qa.config.default_vin
    telemetries.append(BatteryTelemetryModel(**data))
anom = qa.detect_anomalies(telemetries)
print('anomalies list:', anom)
print('### run_test_suite result ###')
result = asyncio.run(qa.run_test_suite(test_data))
print(result)

# simulate persistence using integration-test pattern
from ev_qa_framework.analysis import EVBatteryAnalyzer
import tempfile

# training data identical points (as in integration test)
train_data = [
    {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 80, 'soh': 95}
] * 20
qa_train = EVQAFramework("Trainer")
asyncio.run(qa_train.run_test_suite(train_data))
with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
    model_path = f.name
qa_train.ml_analyzer.save_model(model_path, metadata={'task': 'debug'})
print('saved model to', model_path)
loaded = EVBatteryAnalyzer.load_model(model_path)
qa_prod = EVQAFramework("Prod")
qa_prod.ml_analyzer = loaded

anomaly_data = [{'voltage':800.0,'current':300,'temperature':55,'soc':20,'soh':90}]
df_anom = pd.DataFrame(anomaly_data)
ml_results = qa_prod.ml_analyzer.analyze_telemetry(df_anom)
print('ml_results on anomaly_data', ml_results)
print('predictions via loaded model:', qa_prod.ml_analyzer.model.predict(qa_prod.ml_analyzer.scaler.transform(df_anom[['voltage','current','temp']])) if hasattr(qa_prod.ml_analyzer.model, 'predict') else 'no predict')
