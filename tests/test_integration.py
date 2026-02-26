"""
Интеграционные тесты для EV-QA-Framework.
Проверка взаимодействия всех компонентов: Config -> Framework -> Model -> Results.
"""

import pytest
import asyncio
import os
import tempfile
from ev_qa_framework.config import FrameworkConfig, SafetyThresholds
from ev_qa_framework.framework import EVQAFramework
from ev_qa_framework.analysis import EVBatteryAnalyzer


class TestIntegrationFlow:
    """Интеграционные тесты полного цикла"""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_custom_config(self):
        """
        Тестирование полного цикла:
        1. Создание кастомного конфига
        2. Инициализация фреймворка
        3. Запуск теста на смешанных данных (норма + аномалии)
        4. Проверка результатов
        """
        # 1. Настройка строгого конфига
        config = FrameworkConfig()
        config.safety_thresholds.max_temperature = 45.0  # Очень строго
        config.default_vin = "INTEGRATION_TEST_VIN"
        # в интеграционном сценарии мы хотим, чтобы rule-based аномалии
        # тоже считались провалами
        config.fail_on_anomaly = True
        
        # 2. Инициализация
        qa = EVQAFramework(name="Integration-QA", config=config)
        
        # 3. Подготовка данных
        test_data = [
            # Нормальные данные (для обучения и проверки)
            {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 80, 'soh': 98},
            {'voltage': 401.0, 'current': 51, 'temperature': 31, 'soc': 79, 'soh': 98},
            {'voltage': 402.0, 'current': 52, 'temperature': 32, 'soc': 78, 'soh': 98},
            # Rule-based аномалия (температура > 45)
            {'voltage': 400.0, 'current': 50, 'temperature': 50, 'soc': 77, 'soh': 98},
            # Rule-based аномалия (скачок температуры > 5)
            {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 76, 'soh': 98},
            {'voltage': 400.0, 'current': 50, 'temperature': 40, 'soc': 75, 'soh': 98}, # Скачок +10
        ]
        
        # 4. Запуск
        results = await qa.run_test_suite(test_data)
        
        # 5. Проверка
        assert results['total_tests'] == 6
        # Первые 3 прошли, 4-й завален по температуре, 5-й прошел (база для скачка), 6-й завален по скачку
        # Но подождите: 4-й завален (50 > 45). 6-й завален (скачок 40-30=10 > 5).
        # Итого 2 завалено, 4 прошло.
        assert results['passed'] == 4
        assert results['failed'] == 2
        
        # Проверка сообщений об аномалиях
        anomaly_list = results['anomalies']
        assert any("Температуры: 50.0" in msg for msg in anomaly_list)
        assert any("Резкий скачок температуры: 10.0" in msg for msg in anomaly_list)

    @pytest.mark.asyncio
    async def test_ml_persistence_integration(self):
        """
        Интеграция ML персистентности:
        1. Обучение модели на нормальных данных
        2. Сохранение модели
        3. Загрузка во фреймворк
        4. Проверка детекции на новых данных без переобучения
        """
        # Данные для обучения
        train_data = [
            {'voltage': 400.0, 'current': 50, 'temperature': 30, 'soc': 80, 'soh': 95}
        ] * 20 # 20 одинаковых точек для стабильности
        
        # 1. Обучаем
        qa_train = EVQAFramework("Trainer")
        await qa_train.run_test_suite(train_data)
        
        # 2. Сохраняем модель
        with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as f:
            model_path = f.name
            
        try:
            qa_train.ml_analyzer.save_model(model_path, metadata={'task': 'integration_test'})
            
            # 3. Создаем новый фреймворк и загружаем модель
            loaded_analyzer = EVBatteryAnalyzer.load_model(model_path)
            qa_prod = EVQAFramework("Production")
            qa_prod.ml_analyzer = loaded_analyzer
            
            # 4. Проверяем на аномалии (без дополнительного обучения)
            anomaly_data = [
                {'voltage': 800.0, 'current': 300, 'temperature': 55, 'soc': 20, 'soh': 90} # Сильное отклонение
            ]
            
            # Мы используем напрямую ml_analyzer, так как run_test_suite вызывает analyze_telemetry
            # который в текущей реализации AnomalyDetector делает fit если данных много или если вызывается метод analyze_telemetry.
            # Но если мы загрузили модель, она уже fitted.
            
            import pandas as pd
            df_anomaly = pd.DataFrame(anomaly_data)
            ml_results = qa_prod.ml_analyzer.analyze_telemetry(df_anomaly)
            
            # Так как мы обучили на 400V/50A, 800V/300A точно будет аномалией
            assert ml_results['anomalies_detected'] > 0
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    @pytest.mark.asyncio
    async def test_config_hot_reload_simulation(self):
        """
        Симуляция смены конфигурации "на лету"
        """
        qa = EVQAFramework("Hot-Reload")
        
        # Сначала дефолтные пороги (60 градусов)
        telemetry = {'voltage': 400.0, 'current': 50, 'temperature': 55, 'soc': 80, 'soh': 98}
        results_1 = await qa.run_test_suite([telemetry])
        assert results_1['passed'] == 1
        
        # Меняем конфиг на более строгий (50 градусов)
        new_config = FrameworkConfig()
        new_config.safety_thresholds.max_temperature = 50.0
        qa.config = new_config
        
        results_2 = await qa.run_test_suite([telemetry])
        assert results_2['failed'] == 1
        assert "Температуры: 55.0" in results_2['anomalies'][0]

    def test_error_handling_invalid_telemetry_format(self):
        """
        Тест обработки ошибок при некорректном формате данных
        """
        qa = EVQAFramework("Error-Handler")
        
        # Данные с пропущенным обязательным полем 'voltage'
        invalid_data = [
            {'current': 50, 'temperature': 30, 'soc': 80, 'soh': 98}
        ]
        
        # run_test_suite возвращает результаты даже если были ошибки валидации,
        # логируя их как проваленные тесты.
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(qa.run_test_suite(invalid_data))
        
        assert results['total_tests'] == 1
        assert results['failed'] == 1
        assert any("Validation failed" in msg for msg in results['critical_issues'])

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
