#!/usr/bin/env python3
"""
Tesla Model S Battery QA Test
Реальный тест батареи Tesla Model S с поиском дефектов
"""

import pandas as pd
import json
from datetime import datetime
import sys
import os
sys.path.append('.')

from ev_qa_framework.framework import EVQAFramework, BatteryTelemetry
from ev_qa_framework.analysis import EVBatteryAnalyzer
from ev_qa_models import validate_telemetry

def analyze_tesla_battery():
    print("Tesla Model S Battery QA Analysis")
    print("=" * 50)
    
    # Загружаем данные
    df = pd.read_csv('examples/tesla_model_s_defective.csv')
    print("Loaded {} telemetry points".format(len(df)))
    print("VIN: {}".format(df['vin'].iloc[0]))
    
    # Инициализируем фреймворк
    qa = EVQAFramework("Tesla-Model-S-QA")
    
    # Конвертируем в нужный формат
    telemetry_data = []
    critical_issues = []
    
    print("\nAnalyzing telemetry points...")
    
    for idx, row in df.iterrows():
        data = {
            'voltage': row['voltage'],
            'current': row['current'],
            'temperature': row['temp'],
            'soc': row['soc'],
            'soh': row['soh']
        }
        
        # Проверяем каждую точку через Pydantic
        try:
            validated = validate_telemetry({
                'vin': row['vin'],
                'voltage': row['voltage'],
                'current': row['current'],
                'temperature': row['temp'],
                'soc': row['soc'],
                'soh': row['soh']
            })
        except Exception as e:
            critical_issues.append("Point {}: Validation failed - {}".format(idx, e))
        
        telemetry_data.append(data)
        
        # Проверяем критические значения
        if row['voltage'] > 450:
            critical_issues.append("CRITICAL: Overvoltage {}V at point {}".format(row['voltage'], idx))
        if row['voltage'] < 50:
            critical_issues.append("CRITICAL: Undervoltage {}V at point {}".format(row['voltage'], idx))
        if row['temp'] > 70:
            critical_issues.append("CRITICAL: Overheating {}C at point {}".format(row['temp'], idx))
    
    # Запускаем полный анализ
    print("\nRunning ML-powered analysis...")
    import asyncio
    results = asyncio.run(qa.run_test_suite(telemetry_data))
    
    # Выводим результаты
    print("\n" + "="*50)
    print("TESLA MODEL S BATTERY TEST RESULTS")
    print("="*50)
    
    print("Total tests: {}".format(results['total_tests']))
    print("Passed: {}".format(results['passed']))
    print("Failed: {}".format(results['failed']))
    print("Success rate: {:.1f}%".format(results['passed']/results['total_tests']*100))
    
    if results['ml_analysis']:
        ml = results['ml_analysis']
        print("\nML ANALYSIS:")
        print("   Anomalies detected: {}".format(ml['anomalies_detected']))
        print("   Anomaly rate: {:.2f}%".format(ml['anomaly_percentage']))
        print("   Severity: {}".format(ml['severity']))
    
    print("\nRULE-BASED ANOMALIES:")
    for anomaly in results['anomalies']:
        print("   - {}".format(anomaly))
    
    print("\nCRITICAL ISSUES FOUND:")
    for issue in critical_issues:
        print("   - {}".format(issue))
    
    # Диагноз батареи
    print("\nBATTERY DIAGNOSIS:")
    
    failure_rate = results['failed'] / results['total_tests']
    anomaly_rate = results['ml_analysis']['anomaly_percentage'] if results['ml_analysis'] else 0
    
    if failure_rate > 0.2 or anomaly_rate > 15:
        print("   STATUS: BATTERY REJECTED - Multiple critical issues detected")
        print("   ACTION: Replace battery pack")
    elif failure_rate > 0.1 or anomaly_rate > 8:
        print("   STATUS: BATTERY WARNING - Issues detected, monitoring required")
        print("   ACTION: Schedule maintenance")
    else:
        print("   STATUS: BATTERY APPROVED - Within acceptable parameters")
        print("   ACTION: Continue normal operation")
    
    # Сохраняем отчет
    report = {
        'timestamp': datetime.now().isoformat(),
        'vehicle': 'Tesla Model S',
        'vin': df['vin'].iloc[0],
        'test_results': results,
        'critical_issues': critical_issues,
        'diagnosis': 'REJECTED' if failure_rate > 0.2 else 'WARNING' if failure_rate > 0.1 else 'APPROVED'
    }
    
    with open('tesla_battery_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nFull report saved to: tesla_battery_report.json")
    
    return results

if __name__ == '__main__':
    try:
        results = analyze_tesla_battery()
    except Exception as e:
        print("Test failed: {}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)