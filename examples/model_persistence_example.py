"""
Пример использования персистентности ML-моделей
Демонстрирует train-once, deploy-many паттерн для production
"""

import numpy as np
import pandas as pd
from ev_qa_framework.analysis import EVBatteryAnalyzer, AnomalyDetector


def generate_normal_telemetry(n_samples=1000):
    """Генерация нормальных данных телеметрии"""
    np.random.seed(42)
    return pd.DataFrame({
        'voltage': np.random.normal(48, 1, n_samples),
        'current': np.random.normal(100, 5, n_samples),
        'temp': np.random.normal(35, 2, n_samples),
        'soc': np.random.normal(85, 5, n_samples)
    })


def generate_test_telemetry_with_anomalies():
    """Генерация тестовых данных с аномалиями"""
    normal_data = pd.DataFrame({
        'voltage': [48, 48, 48, 48, 48],
        'current': [100, 100, 100, 100, 100],
        'temp': [35, 35, 35, 35, 35],
        'soc': [85, 85, 85, 85, 85]
    })
    
    # Добавляем аномалии
    anomalies = pd.DataFrame({
        'voltage': [200, 10],  # Экстремальные значения
        'current': [500, 600],
        'temp': [90, -20],
        'soc': [5, 100]
    })
    
    return pd.concat([normal_data, anomalies], ignore_index=True)


def example_1_train_and_save():
    """
    Пример 1: Обучение модели и сохранение
    
    Типичный сценарий: обучаем модель на исторических данных
    и сохраняем для использования в production.
    """
    print("=" * 70)
    print("Пример 1: Обучение и сохранение модели")
    print("=" * 70)
    
    # Генерируем тренировочные данные (исторические "нормальные" данные)
    print("📊 Генерация тренировочных данных (1000 нормальных точек)...")
    train_data = generate_normal_telemetry(1000)
    
    # Создаем и обучаем анализатор
    print("🧠 Обучение ML-модели (Isolation Forest)...")
    analyzer = EVBatteryAnalyzer(
        contamination=0.05,  # Ожидаем 5% аномалий
        n_estimators=200,
        critical_threshold=-0.9,
        warning_threshold=-0.6
    )
    
    results = analyzer.analyze_telemetry(train_data)
    print(f"   Обучено на {results['total_samples']} точках")
    print(f"   Обнаружено аномалий: {results['anomalies_detected']}")
    print(f"   Серьезность: {results['severity']}")
    
    # Сохраняем модель с метаданными
    print("\n💾 Сохранение модели...")
    metadata = {
        'version': '1.0',
        'dataset': 'historical_battery_data_2024',
        'contamination': 0.05,
        'date': '2024-01-28',
        'description': 'Baseline model trained on 1000 normal samples'
    }
    
    analyzer.save_model('models/battery_analyzer_baseline', metadata=metadata)
    print()


def example_2_load_and_infer():
    """
    Пример 2: Загрузка сохраненной модели и inference
    
    Production сценарий: загружаем предобученную модель
    и используем для детекции аномалий в новых данных.
    """
    print("=" * 70)
    print("Пример 2: Загрузка модели и inference")
    print("=" * 70)
    
    # Загружаем сохраненную модель
    print("📥 Загрузка сохраненной модели...")
    analyzer = EVBatteryAnalyzer.load_model('models/battery_analyzer_baseline')
    
    # Получаем информацию о модели
    print("\n📋 Информация о загруженной модели:")
    info = analyzer.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Тестируем на новых данных
    print("\n🔍 Тестирование на новых данных с аномалиями...")
    test_data = generate_test_telemetry_with_anomalies()
    
    results = analyzer.analyze_telemetry(test_data)
    print(f"   Всего точек: {results['total_samples']}")
    print(f"   Аномалий: {results['anomalies_detected']}")
    print(f"   Процент аномалий: {results['anomaly_percentage']:.2f}%")
    print(f"   Серьезность: {results['severity']}")
    print()


def example_3_model_versioning():
    """
    Пример 3: Версионирование моделей
    
    Сценарий: создаем несколько версий модели с разными параметрами
    для A/B тестирования.
    """
    print("=" * 70)
    print("Пример 3: Версионирование моделей (A/B тестирование)")
    print("=" * 70)
    
    train_data = generate_normal_telemetry(500)
    test_data = generate_test_telemetry_with_anomalies()
    
    # Модель A: консервативная (низкий contamination)
    print("\n🅰️  Создание модели A (консервативная, contamination=0.01)...")
    model_a = EVBatteryAnalyzer(contamination=0.01, n_estimators=200)
    model_a.analyze_telemetry(train_data)
    model_a.save_model('models/model_a_conservative', 
                       metadata={'version': 'A', 'type': 'conservative'})
    
    # Модель B: терпимая (высокий contamination)
    print("🅱️  Создание модели B (терпимая, contamination=0.15)...")
    model_b = EVBatteryAnalyzer(contamination=0.15, n_estimators=200)
    model_b.analyze_telemetry(train_data)
    model_b.save_model('models/model_b_tolerant', 
                       metadata={'version': 'B', 'type': 'tolerant'})
    
    # Сравнение моделей на тестовых данных
    print("\n📊 Сравнение моделей на тестовых данных:")
    
    loaded_a = EVBatteryAnalyzer.load_model('models/model_a_conservative')
    results_a = loaded_a.analyze_telemetry(test_data)
    print(f"   Модель A: {results_a['anomalies_detected']} аномалий, {results_a['severity']}")
    
    loaded_b = EVBatteryAnalyzer.load_model('models/model_b_tolerant')
    results_b = loaded_b.analyze_telemetry(test_data)
    print(f"   Модель B: {results_b['anomalies_detected']} аномалий, {results_b['severity']}")
    print()


def example_4_anomaly_detector_pattern():
    """
    Пример 4: Train/Detect паттерн с AnomalyDetector
    
    Более продвинутый паттерн для production с разделением
    обучения и детекции.
    """
    print("=" * 70)
    print("Пример 4: AnomalyDetector - Train/Detect паттерн")
    print("=" * 70)
    
    # Обучение на "чистых" данных
    print("🧠 Обучение AnomalyDetector на чистых данных...")
    clean_data = generate_normal_telemetry(1000)
    
    detector = AnomalyDetector(contamination=0.01, n_estimators=250)
    detector.train(clean_data)
    
    # Сохранение
    print("\n💾 Сохранение обученного детектора...")
    detector.save_model('models/anomaly_detector_prod', 
                       metadata={'purpose': 'production_detector', 'trained_samples': 1000})
    
    # В другой сессии/сервисе: загрузка и детекция
    print("\n📥 Загрузка детектора (симуляция production environment)...")
    # Примечание: в текущей версии load_model возвращает EVBatteryAnalyzer
    # В production стоит добавить AnomalyDetector.load_model
    loaded_detector = EVBatteryAnalyzer.load_model('models/anomaly_detector_prod')
    
    # Real-time детекция
    print("\n🔍 Real-time детекция на streaming данных...")
    
    # Симулируем поток данных (батчи по 10 точек)
    for batch_num in range(3):
        batch_data = generate_test_telemetry_with_anomalies() if batch_num == 1 else generate_normal_telemetry(10)
        results = loaded_detector.analyze_telemetry(batch_data)
        
        print(f"   Батч {batch_num + 1}: "
              f"{results['anomalies_detected']}/{results['total_samples']} аномалий "
              f"({results['severity']})")
    print()


def example_5_production_workflow():
    """
    Пример 5: Полный production workflow
    
    1. Train offline на исторических данных
    2. Save модель
    3. Deploy в production
    4. Load и inference в реальном времени
    5. Monitoring и versioning
    """
    print("=" * 70)
    print("Пример 5: Production Workflow")
    print("=" * 70)
    
    # OFFLINE TRAINING
    print("\n📚 ФАЗА 1: Offline Training")
    print("-" * 70)
    
    historical_data = generate_normal_telemetry(2000)
    
    # Обучаем baseline модель
    baseline = EVBatteryAnalyzer(
        contamination=0.05,
        n_estimators=300,  # Больше деревьев для production
        critical_threshold=-0.85,
        warning_threshold=-0.55
    )
    
    print("   Обучение baseline модели на 2000 исторических точках...")
    baseline.analyze_telemetry(historical_data)
    
    baseline.save_model(
        'models/production/baseline_v1.0',
        metadata={
            'version': '1.0',
            'stage': 'production',
            'performance': 'baseline',
            'trained_samples': 2000,
            'last_updated': '2024-01-28'
        }
    )
    print("   ✅ Модель сохранена в production")
    
    # PRODUCTION DEPLOYMENT
    print("\n🚀 ФАЗА 2: Production Deployment")
    print("-" * 70)
    
    production_model = EVBatteryAnalyzer.load_model('models/production/baseline_v1.0')
    print("   ✅ Модель загружена в production environment")
    
    # REAL-TIME INFERENCE
    print("\n⚡ ФАЗА 3: Real-time Inference")
    print("-" * 70)
    
    # Симуляция real-time мониторинга
    print("   Мониторинг телеметрии батареи...")
    for minute in range(1, 4):
        incoming_data = generate_normal_telemetry(60)  # 60 точек в минуту
        results = production_model.analyze_telemetry(incoming_data)
        
        status = "🟢 NORMAL" if results['severity'] == 'INFO' else "🔴 ALERT"
        print(f"   Минута {minute}: {status} - "
              f"{results['anomalies_detected']} аномалий ({results['severity']})")
    
    print("\n✅ Production workflow завершен!")
    print()


def main():
    """Главная функция - запуск всех примеров"""
    print("\n" + "=" * 70)
    print(" 🔋 EV-QA-Framework: ML Model Persistence Examples")
    print("=" * 70 + "\n")
    
    # Создаем директорию для моделей
    import os
    os.makedirs('models/production', exist_ok=True)
    
    # Запуск примеров
    example_1_train_and_save()
    example_2_load_and_infer()
    example_3_model_versioning()
    example_4_anomaly_detector_pattern()
    example_5_production_workflow()
    
    print("=" * 70)
    print("✅ Все примеры выполнены успешно!")
    print("=" * 70)
    print("\n💡 Сохраненные модели:")
    print("   - models/battery_analyzer_baseline.joblib")
    print("   - models/model_a_conservative.joblib")
    print("   - models/model_b_tolerant.joblib")
    print("   - models/anomaly_detector_prod.joblib")
    print("   - models/production/baseline_v1.0.joblib")
    print("\n📖 Используйте EVBatteryAnalyzer.load_model() для загрузки")


if __name__ == "__main__":
    main()
