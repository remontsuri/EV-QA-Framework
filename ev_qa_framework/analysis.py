"""EV QA Analysis: ML-based battery telemetry and quality assurance.

Модуль машинного обучения для детекции аномалий в телеметрии батареи.
"""

from __future__ import annotations

import datetime
import logging
import os
import warnings
from typing import Any, Dict, List, Tuple

import joblib  # type: ignore  # no stub available
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EVBatteryAnalyzer:
    """
    ML-анализатор телеметрии батареи EV на основе алгоритма Isolation Forest.

    Isolation Forest — это алгоритм обнаружения аномалий, который изолирует выбросы
    путем случайного выбора признака и затем случайного выбора значения разделения
    между максимумом и минимумом выбранного признака. Аномалии изолируются быстрее,
    чем нормальные точки данных.

    Attributes:
        model: Модель IsolationForest из scikit-learn
        scaler: StandardScaler для нормализации данных
        anomalies: DataFrame с обнаруженными аномалиями
        contamination: Доля аномалий в датасете (по умолчанию 0.1 = 10%)
    """

    def __init__(self, contamination: float = 0.1, n_estimators: int = 200, random_state: int = 42,
                 critical_threshold: float = -0.8, warning_threshold: float = -0.5):
        """
        Инициализация анализатора телеметрии.

        Args:
            contamination: Ожидаемая доля аномалий в данных (0.0 - 1.0).
                          Например, 0.1 означает, что ~10% данных могут быть аномальными.
            n_estimators: Количество деревьев в ансамбле (больше = точнее, но медленнее).
                         Рекомендуется 100-200 для баланса точности и скорости.
            random_state: Seed для воспроизводимости результатов.
            critical_threshold: Порог для CRITICAL severity (по умолчанию -0.8)
            warning_threshold: Порог для WARNING severity (по умолчанию -0.5)

        Примечание:
            - contamination влияет на чувствительность: меньше значение = меньше ложных срабатываний
            - n_estimators рекомендуется 100+ для стабильных результатов
        """
        # Создаем модель Isolation Forest с настроенными параметрами
        self.model = IsolationForest(
            contamination=contamination,    # Ожидаемая доля аномалий
            n_estimators=n_estimators,      # Количество деревьев (больше = стабильнее)
            max_samples='auto',             # Авто-выбор размера подвыборки
            random_state=random_state,      # Для воспроизводимости
            n_jobs=-1                       # Использовать все CPU ядра
        )

        # StandardScaler нормализует данные: (x - mean) / std
        # Это важно, так как IsolationForest чувствителен к масштабу признаков
        self.scaler = StandardScaler()

        # Хранилище для обнаруженных аномалий (заполняется после analyze_telemetry)
        # anomalies stored as DataFrame; start empty
        self.anomalies: pd.DataFrame = pd.DataFrame()

        # Сохраняем параметры для доступа извне
        self.contamination = contamination
        self.critical_threshold = critical_threshold
        self.warning_threshold = warning_threshold

    def analyze_telemetry(self, df_telemetry: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ телеметрии батареи на предмет аномалий.

        Алгоритм:
        1. Подготовка данных и выбор признаков.
        2. Нормализация данных через StandardScaler.
        3. Обучение или использование IsolationForest.
        4. Расчет anomaly scores и предсказание аномалий.
        5. Оценка серьезности.

        Args:
            df_telemetry: DataFrame с колонками ['voltage', 'current', 'temp', 'soc'].

        Returns:
            Словарь с результатами анализа.
        """
        if df_telemetry.empty:
            return {
                'total_samples': 0,
                'anomalies_detected': 0,
                'anomaly_percentage': 0.0,
                'severity': 'INFO'
            }

        # Шаг 1: Подготовка данных
        df: pd.DataFrame = df_telemetry.copy()
        if 'temperature' in df.columns and 'temp' not in df.columns:
            df = df.rename(columns={'temperature': 'temp'})

        # Шаг 2: Выбор признаков
        features: List[str] = ['voltage', 'current', 'temp']
        # Ensure all features exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            logger.error("Missing features for ML analysis: %s", missing)
            # Fallback to whatever features are available
            features = [f for f in features if f in df.columns]
            if not features:
                raise ValueError(
                    f"None of the required features {features} found in DataFrame"
                )

        features_df: pd.DataFrame = df[features]

        # Шаг 3: Нормализация данных
        if hasattr(self.scaler, 'mean_'):
            scaled_features = self.scaler.transform(features_df)
        else:
            scaled_features = self.scaler.fit_transform(features_df)

        # Шаг 4: Обучение модели и предсказание аномалий
        if hasattr(self.model, 'estimators_'):
            preds = self.model.predict(scaled_features)
        else:
            preds = self.model.fit_predict(scaled_features)

        anomaly_scores: np.ndarray = self.model.score_samples(scaled_features)

        # Шаг 5: Фильтрация аномалий
        mask: np.ndarray = (preds == -1) | (anomaly_scores < self.warning_threshold)
        self.anomalies = df_telemetry[mask].copy()

        if not self.anomalies.empty:
            self.anomalies['anomaly_score'] = anomaly_scores[mask]

        # Шаг 6: Результат
        total = len(df_telemetry)
        count = len(self.anomalies)
        return {
            'total_samples': total,
            'anomalies_detected': count,
            'anomaly_percentage': (count / total) * 100 if total else 0.0,
            'severity': self._assess_severity(anomaly_scores),
            'mean_score': np.mean(anomaly_scores)
        }

    def _assess_severity(self, anomaly_scores: np.ndarray) -> str:
        """
        Оценка уровня серьезности обнаруженных аномалий.

        Логика оценки:
        - CRITICAL: Есть экстремальные выбросы (score < critical_threshold)
                   Требуется немедленное внимание — возможна критическая неисправность
        - WARNING: Умеренные аномалии (score < warning_threshold)
                  Требуется проверка — возможна деградация системы
        - INFO: Слабые аномалии или их отсутствие (score >= warning_threshold)
               Система в норме, аномалии незначительны

        Args:
            scores: Массив anomaly scores из IsolationForest

        Returns:
            Строка с уровнем серьезности: 'CRITICAL', 'WARNING' или 'INFO'

        Примечание:
            Пороги настраиваются через параметры конструктора и могут корректироваться
            под конкретную систему на основе исторических данных.
        """
        min_score = np.min(anomaly_scores)

        if min_score < self.critical_threshold:
            return 'CRITICAL'  # Экстремальная аномалия — критический уровень
        if min_score < self.warning_threshold:
            return 'WARNING'   # Умеренная аномалия — предупреждение
        return 'INFO'          # Слабая аномалия или норма

    def save_model(self, filepath: str, meta: Dict[str, Any] | None = None) -> None:
        """
        Сохранение обученной модели и scaler в файл.

        Сохраняет:
        - Обученную модель IsolationForest
        - Обученный StandardScaler
        - Параметры модели (contamination, thresholds, etc.)
        - Метаданные (дата обучения, версия, комментарии)

        Args:
            filepath: Путь для сохранения модели (без расширения, добавится .joblib)
            metadata: Опциональные метаданные (комментарии, версия, dataset info)

        Пример:
            >>> analyzer = EVBatteryAnalyzer()
            >>> analyzer.analyze_telemetry(df)
            >>> analyzer.save_model('models/battery_analyzer_v1',
            ...                     metadata={'version': '1.0', 'dataset': 'Tesla_2024'})

        Raises:
            ValueError: Если модель не обучена (не был вызван analyze_telemetry)
        """
        # Проверка, что модель обучена (scaler должен быть fitted)
        if not hasattr(self.scaler, 'mean_'):
            raise ValueError(
                "Модель не обучена! Сначала вызовите analyze_telemetry() или train()"
            )

        # Подготовка данных для сохранения
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'critical_threshold': self.critical_threshold,
            'warning_threshold': self.warning_threshold,
            'save_timestamp': datetime.datetime.now().isoformat(),
            'metadata': meta or {}
        }

        # Добавляем расширение .joblib если его нет
        if not filepath.endswith('.joblib'):
            filepath = filepath + '.joblib'

        # Создаем директорию если не существует
        dir_name = os.path.dirname(filepath)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Сохранение
        joblib.dump(model_data, filepath, compress=3)
        print(f"✅ Модель сохранена: {filepath}")

        if meta:
            print(f"   Метаданные: {meta}")

    @classmethod
    def load_model(cls, filepath: str) -> 'EVBatteryAnalyzer':
        """
        Загрузка сохраненной модели из файла.

        Загружает все компоненты модели и создает новый экземпляр EVBatteryAnalyzer
        с восстановленным состоянием.

        Args:
            filepath: Путь к сохраненной модели (.joblib)

        Returns:
            Новый экземпляр EVBatteryAnalyzer с загруженной моделью

        Пример:
            >>> analyzer = EVBatteryAnalyzer.load_model('models/battery_analyzer_v1.joblib')
            >>> results = analyzer.analyze_telemetry(new_data)

        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если файл поврежден или имеет неверный формат
        """
        # Добавляем расширение если нет
        if not filepath.endswith('.joblib'):
            filepath = filepath + '.joblib'

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл модели не найден: {filepath}")

        try:
            # Загрузка данных
            model_data = joblib.load(filepath)

            # Создание нового экземпляра
            new_analyzer = cls(
                contamination=model_data['contamination'],
                critical_threshold=model_data.get('critical_threshold', -0.8),
                warning_threshold=model_data.get('warning_threshold', -0.5)
            )

            # Восстановление модели и scaler
            new_analyzer.model = model_data['model']
            new_analyzer.scaler = model_data['scaler']

            # Вывод информации о загруженной модели
            save_time = model_data.get('save_timestamp', 'Unknown')
            meta = model_data.get('metadata', {})

            print(f"✅ Модель загружена: {filepath}")
            print(f"   Сохранена: {save_time}")
            if meta:
                print(f"   Метаданные: {meta}")

            return new_analyzer

        except Exception as e:
            raise ValueError(f"Ошибка загрузки модели: {e}") from e

    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о текущей модели.

        Returns:
            Словарь с параметрами модели
        """
        return {
            'contamination': self.contamination,
            'n_estimators': getattr(self.model, 'n_estimators', None),
            'critical_threshold': self.critical_threshold,
            'warning_threshold': self.warning_threshold,
            'is_fitted': hasattr(self.scaler, 'mean_')
        }


class AnomalyDetector(EVBatteryAnalyzer):
    """
    Расширенный класс-детектор аномалий с раздельными методами train/detect.

    Этот класс позволяет:
    1. Обучить модель на "нормальных" данных (train)
    2. Использовать обученную модель для детекции на новых данных (detect)

    Это полезно в продакшене, когда модель обучается один раз на исторических
    данных, а затем используется для real-time детекции.
    """

    def __init__(self, contamination: float = 0.01, n_estimators: int = 200, random_state: int = 42):
        """
        Инициализация детектора аномалий.

        Args:
            contamination: Ожидаемая доля аномалий (по умолчанию 0.01 = 1%).
                          Для обучения на "чистых" данных используйте малое значение.
            n_estimators: Количество деревьев (рекомендуется 200 для стабильности).
            random_state: Seed для воспроизводимости.
        """
        super().__init__(contamination, n_estimators, random_state)
        self._is_trained = False  # Флаг обученности модели

    def train(self, training_data: pd.DataFrame) -> None:
        """
        Обучение модели на "нормальных" данных.

        Рекомендуется использовать данные без аномалий для обучения,
        чтобы модель научилась распознавать нормальное поведение батареи.

        Args:
            data: DataFrame с колонками ['voltage', 'current', 'temp', 'soc'].
                  Данные должны содержать преимущественно нормальные значения.

        Пример:
            >>> normal_data = pd.DataFrame({
            ...     'voltage': np.random.normal(48, 1, 1000),
            ...     'current': np.random.normal(100, 5, 1000),
            ...     'temp': np.random.normal(35, 2, 1000),
            ...     'soc': np.random.normal(85, 5, 1000)
            ... })
            >>> detector = AnomalyDetector()
            >>> detector.train(normal_data)
        """
        features = ['voltage', 'current', 'temp']
        features_df = training_data[features]

        # Обучаем scaler на нормальных данных
        scaled_features = self.scaler.fit_transform(features_df)

        # Обучаем IsolationForest
        self.model.fit(scaled_features)
        self._is_trained = True
        print(f"✅ Модель обучена на {len(training_data)} точках данных")

    def detect(self, inference_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Детекция аномалий на новых данных с использованием обученной модели.

        Args:
            data: DataFrame с новой телеметрией для анализа.

        Returns:
            Кортеж (predictions, scores):
                - predictions: Массив предсказаний (-1 = аномалия, 1 = норма)
                - scores: Массив anomaly scores

        Raises:
            ValueError: Если модель не обучена (нужно сначала вызвать train)

        Пример:
            >>> new_data = pd.DataFrame({
            ...     'voltage': [48, 200],  # 200 — аномалия
            ...     'current': [100, 100],
            ...     'temp': [35, 35],
            ...     'soc': [85, 85]
            ... })
            >>> predictions, scores = detector.detect(new_data)
            >>> print(predictions)  # [1, -1]
        """
        if not self._is_trained:
            raise ValueError("Модель не обучена! Сначала вызовите метод train()")

        features = ['voltage', 'current', 'temp']
        features_df = inference_data[features]

        # Применяем уже обученный scaler
        scaled_features = self.scaler.transform(features_df)

        # Предсказание на новых данных
        preds = self.model.predict(scaled_features)
        anomaly_scores = self.model.score_samples(scaled_features)

        anomaly_count = np.sum(preds == -1)
        print(f"🔍 Обнаружено аномалий: {anomaly_count}/{len(inference_data)}")

        return preds, anomaly_scores


if __name__ == '__main__':
    # Пример использования EVBatteryAnalyzer
    print("=== Тест EVBatteryAnalyzer ===")
    main_analyzer = EVBatteryAnalyzer()

    # Генерируем тестовую телеметрию
    np.random.seed(42)
    main_data = {
        'voltage': np.random.normal(48, 2, 1000),
        'current': np.random.normal(100, 15, 1000),
        'temp': np.random.normal(35, 5, 1000),
        'soc': np.random.normal(85, 10, 1000)
    }
    main_df = pd.DataFrame(main_data)

    # Анализ
    main_results = main_analyzer.analyze_telemetry(main_df)
    print(f"Анализ завершен: {main_results}")
    print(f"Аномалий: {main_results['anomalies_detected']}/{main_results['total_samples']}")
    print(f"Серьезность: {main_results['severity']}")

    # Пример использования AnomalyDetector
    print("\n=== Тест AnomalyDetector (train/detect) ===")
    detector = AnomalyDetector(contamination=0.01, n_estimators=200)

    # Обучение на нормальных данных
    normal_data = pd.DataFrame({
        'voltage': np.random.normal(48, 1, 500),
        'current': np.random.normal(100, 5, 500),
        'temp': np.random.normal(35, 2, 500),
        'soc': np.random.normal(85, 5, 500)
    })
    detector.train(normal_data)

    # Детекция на новых данных с аномалией
    test_data = pd.DataFrame({
        'voltage': [48, 48, 200, 48],  # 200V — явная аномалия
        'current': [100, 100, 100, 100],
        'temp': [35, 35, 35, 35],
        'soc': [85, 85, 85, 85]
    })
    predictions, scores = detector.detect(test_data)
    print(f"Предсказания: {predictions}")
    print(f"Scores: {scores}")

