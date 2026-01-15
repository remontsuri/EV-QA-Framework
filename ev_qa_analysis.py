"""EV QA Analysis: ML-based battery telemetry and quality assurance."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class EVBatteryAnalyzer:
    """
    Machine Learning-based EV battery telemetry analyzer
    for quality assurance testing and anomaly detection.
    """
    
    def __init__(self, contamination=0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.anomalies = None
        
    def analyze_telemetry(self, df_telemetry):
        """
        Analyze battery telemetry data for anomalies.
        
        Args:
            df_telemetry: DataFrame with columns [voltage, current, temp, soc]
        
        Returns:
            dict with anomaly scores and alerts
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(df_telemetry[['voltage', 'current', 'temp']])
        
        # Train and predict
        predictions = self.model.fit_predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        # Identify anomalies
        self.anomalies = df_telemetry[predictions == -1]
        
        return {
            'total_samples': len(df_telemetry),
            'anomalies_detected': len(self.anomalies),
            'anomaly_percentage': (len(self.anomalies) / len(df_telemetry)) * 100,
            'severity': self._assess_severity(anomaly_scores)
        }
    
    def _assess_severity(self, scores):
        """
        Assess severity of detected anomalies (Critical/Warning/Info).
        """
        if np.min(scores) < -0.8:
            return 'CRITICAL'
        elif np.min(scores) < -0.5:
            return 'WARNING'
        return 'INFO'

if __name__ == '__main__':
    # Example usage
    analyzer = EVBatteryAnalyzer()
    
    # Generate sample telemetry
    np.random.seed(42)
    data = {
        'voltage': np.random.normal(48, 2, 1000),
        'current': np.random.normal(100, 15, 1000),
        'temp': np.random.normal(35, 5, 1000),
        'soc': np.random.normal(85, 10, 1000)
    }
    df = pd.DataFrame(data)
    
    # Analyze
    results = analyzer.analyze_telemetry(df)
    print(f"Analysis Complete: {results}")
    print(f"Anomalies: {results['anomalies_detected']}/{results['total_samples']}")
    print(f"Severity: {results['severity']}")
