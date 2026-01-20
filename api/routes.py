from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
from pydantic import BaseModel
from ev_qa_framework import EVQAFramework, EVBatteryAnalyzer
from ev_qa_models import BatteryTelemetryModel

router = APIRouter()

class AnalysisRequest(BaseModel):
    telemetry: List[dict]

class AnalysisResponse(BaseModel):
    total_samples: int
    anomalies_detected: int
    anomaly_percentage: float
    severity: str
    rule_based_anomalies: List[str]

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_telemetry(request: AnalysisRequest):
    """Analyze battery telemetry using ML and rule-based methods"""
    try:
        # Initialize framework
        qa = EVQAFramework("API-Analyzer")
        
        # Run analysis
        results = await qa.run_test_suite(request.telemetry)
        
        return AnalysisResponse(
            total_samples=results['ml_analysis']['total_samples'],
            anomalies_detected=results['ml_analysis']['anomalies_detected'],
            anomaly_percentage=results['ml_analysis']['anomaly_percentage'],
            severity=results['ml_analysis']['severity'],
            rule_based_anomalies=results['anomalies']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_single_telemetry(telemetry: BatteryTelemetryModel):
    """Validate single telemetry point"""
    qa = EVQAFramework()
    
    # Convert Pydantic model to BatteryTelemetry
    from ev_qa_framework.framework import BatteryTelemetry
    bt = BatteryTelemetry(
        voltage=telemetry.voltage,
        current=telemetry.current,
        temperature=telemetry.temperature,
        soc=telemetry.soc,
        soh=telemetry.soh
    )
    
    is_valid = qa.validate_telemetry(bt)
    
    return {
        "valid": is_valid,
        "telemetry": telemetry.model_dump()
    }