"""Специальные тесты для "Tesla Potesti" сценария"""

import pytest
import asyncio
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.config import TESLA_CONFIG


def test_tesla_configuration_valid():
    """Проверяем что профиль Tesla корректно создаётся"""
    assert TESLA_CONFIG.safety_thresholds.max_voltage == 450.0
    assert TESLA_CONFIG.fail_on_anomaly is True
    assert len(TESLA_CONFIG.default_vin) == 17


@pytest.mark.asyncio
async def test_tesla_normal_and_spike():
    """Один нормальный цикл и один скачок температуры"""
    qa = EVQAFramework(name="Tesla-Potesti-QA", config=TESLA_CONFIG)
    normal = {'voltage': 360.0, 'current': 120.0, 'temperature': 35.0, 'soc': 80.0, 'soh': 95.0}
    spike = {'voltage': 400.0, 'current': 200.0, 'temperature': 65.0, 'soc': 60.0, 'soh': 90.0}
    results = await qa.run_test_suite([normal, spike])
    # первый элемент должен пройти, второй засчитать аномалией и провалом
    assert results['passed'] == 1
    assert results['failed'] == 1
    assert any('Температуры' in msg for msg in results['anomalies'])


@pytest.mark.asyncio
async def test_tesla_vin_format():
    """VIN в профиле должен проходить валидацию"""
    qa = EVQAFramework(config=TESLA_CONFIG)
    data = {'voltage': 300.0, 'current': 50.0, 'temperature': 25.0, 'soc': 90.0, 'soh': 98.0}
    result = await qa.run_test_suite([data])
    assert result['passed'] == 1
