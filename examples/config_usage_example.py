"""
Пример использования конфигурационной системы EV-QA-Framework

Демонстрирует:
1. Создание кастомных конфигураций
2. Сохранение/загрузку конфигураций
3. Использование разных конфигураций для разных типов батарей
"""

import asyncio
from ev_qa_framework import (
    EVQAFramework, 
    FrameworkConfig, 
    SafetyThresholds, 
    MLConfig
)


def example_1_default_config():
    """Пример 1: Использование дефолтной конфигурации"""
    print("=" * 60)
    print("Пример 1: Дефолтная конфигурация")
    print("=" * 60)
    
    qa = EVQAFramework("Default-QA")
    
    print(f"Макс. температура: {qa.config.safety_thresholds.max_temperature}°C")
    print(f"Диапазон напряжения: {qa.config.safety_thresholds.min_voltage}-{qa.config.safety_thresholds.max_voltage}V")
    print(f"ML contamination: {qa.config.ml_config.contamination}")
    print()


def example_2_tesla_config():
    """Пример 2: Строгая конфигурация для Tesla"""
    print("=" * 60)
    print("Пример 2: Конфигурация Tesla (строгие пороги)")
    print("=" * 60)
    
    # Создаем строгие пороги для Tesla
    tesla_thresholds = SafetyThresholds(
        max_temperature=55.0,      # Tesla более консервативна
        min_temperature=-20.0,      # Работа в умеренном климате
        max_temperature_jump=3.0,   # Очень чувствительно к скачкам
        min_voltage=250.0,          # Более узкий диапазон
        max_voltage=450.0,
        min_soc=20.0,              # Предупреждение при SOC < 20%
        critical_soh=80.0,         # Более высокий порог здоровья
        max_current=600.0
    )
    
    # ML конфигурация с высокой точностью
    tesla_ml = MLConfig(
        contamination=0.05,        # Ожидаем меньше аномалий
        n_estimators=250,          # Больше деревьев = точнее
        critical_score_threshold=-0.9,  # Более строгий порог
        warning_score_threshold=-0.6
    )
    
    config = FrameworkConfig(
        safety_thresholds=tesla_thresholds,
        ml_config=tesla_ml,
        default_vin="5YJ3E1EA8KF000001"  # Tesla VIN
    )
    
    # Сохранение конфигурации
    config.save_to_file("tesla_custom.json")
    print("✅ Конфигурация Tesla сохранена в tesla_custom.json")
    
    qa = EVQAFramework("Tesla-QA", config=config)
    print(f"Макс. температура: {qa.config.safety_thresholds.max_temperature}°C")
    print(f"Диапазон напряжения: {qa.config.safety_thresholds.min_voltage}-{qa.config.safety_thresholds.max_voltage}V")
    print(f"ML contamination: {qa.config.ml_config.contamination}")
    print()


def example_3_nissan_leaf_config():
    """Пример 3: Конфигурация для Nissan Leaf (более мягкие пороги)"""
    print("=" * 60)
    print("Пример 3: Конфигурация Nissan Leaf")
    print("=" * 60)
    
    # Nissan Leaf имеет другие характеристики
    leaf_thresholds = SafetyThresholds(
        max_temperature=65.0,      # Менее строгий порог
        min_temperature=-30.0,
        max_temperature_jump=7.0,  # Более терпимо к скачкам
        min_voltage=300.0,         # Другой диапазон напряжения
        max_voltage=400.0,
        min_soc=10.0,
        critical_soh=65.0,
        max_current=400.0
    )
    
    leaf_ml = MLConfig(
        contamination=0.15,        # Больше аномалий ожидается
        n_estimators=150
    )
    
    config = FrameworkConfig(
        safety_thresholds=leaf_thresholds,
        ml_config=leaf_ml,
        default_vin="1N4AZ0CP0FC000001"  # Nissan Leaf VIN
    )
    
    config.save_to_file("nissan_leaf.json")
    print("✅ Конфигурация Nissan Leaf сохранена в nissan_leaf.json")
    
    qa = EVQAFramework("Leaf-QA", config=config)
    print(f"Макс. температура: {qa.config.safety_thresholds.max_temperature}°C")
    print(f"Диапазон напряжения: {qa.config.safety_thresholds.min_voltage}-{qa.config.safety_thresholds.max_voltage}V")
    print()


async def example_4_testing_with_config():
    """Пример 4: Тестирование с кастомной конфигурацией"""
    print("=" * 60)
    print("Пример 4: Тестирование с кастомной конфигурацией")
    print("=" * 60)
    
    # Загружаем сохраненную конфигурацию
    config = FrameworkConfig.load_from_file("tesla_custom.json")
    qa = EVQAFramework("Test-QA", config=config)
    
    # Тестовые данные
    test_data = [
        {'voltage': 350.0, 'current': 100, 'temperature': 30, 'soc': 80, 'soh': 95},
        {'voltage': 355.0, 'current': 95, 'temperature': 31, 'soc': 78, 'soh': 95},
        {'voltage': 360.0, 'current': 90, 'temperature': 32, 'soc': 76, 'soh': 95},
        {'voltage': 365.0, 'current': 85, 'temperature': 33, 'soc': 74, 'soh': 95},
        {'voltage': 370.0, 'current': 80, 'temperature': 56, 'soc': 72, 'soh': 95},  # Температура > 55°C!
    ]
    
    results = await qa.run_test_suite(test_data)
    
    print(f"📊 Результаты тестирования:")
    print(f"   Всего тестов: {results['total_tests']}")
    print(f"   Пройдено: {results['passed']}")
    print(f"   Провалено: {results['failed']}")
    print(f"   Аномалий: {len(results['anomalies'])}")
    
    if results['ml_analysis']:
        print(f"   ML серьезность: {results['ml_analysis']['severity']}")
    
    print()


def example_5_comparison():
    """Пример 5: Сравнение конфигураций"""
    print("=" * 60)
    print("Пример 5: Сравнение конфигураций")
    print("=" * 60)
    
    # Загружаем разные конфигурации
    tesla_cfg = FrameworkConfig.load_from_file("tesla_custom.json")
    leaf_cfg = FrameworkConfig.load_from_file("nissan_leaf.json")
    
    print("Сравнение Tesla vs Nissan Leaf:")
    print("-" * 60)
    print(f"{'Параметр':<30} {'Tesla':<15} {'Leaf':<15}")
    print("-" * 60)
    print(f"{'Макс. температура':<30} {tesla_cfg.safety_thresholds.max_temperature:<15} {leaf_cfg.safety_thresholds.max_temperature:<15}")
    print(f"{'Макс. напряжение':<30} {tesla_cfg.safety_thresholds.max_voltage:<15} {leaf_cfg.safety_thresholds.max_voltage:<15}")
    print(f"{'Скачок температуры':<30} {tesla_cfg.safety_thresholds.max_temperature_jump:<15} {leaf_cfg.safety_thresholds.max_temperature_jump:<15}")
    print(f"{'ML contamination':<30} {tesla_cfg.ml_config.contamination:<15} {leaf_cfg.ml_config.contamination:<15}")
    print()


def main():
    """Главная функция - запуск всех примеров"""
    print("\n🚗 EV-QA-Framework: Configuration Examples\n")
    
    # Запуск примеров
    example_1_default_config()
    example_2_tesla_config()
    example_3_nissan_leaf_config()
    
    # Асинхронный пример
    asyncio.run(example_4_testing_with_config())
    
    example_5_comparison()
    
    print("=" * 60)
    print("✅ Все примеры выполнены успешно!")
    print("=" * 60)


if __name__ == "__main__":
    main()
