# 📊 Jupyter Notebook Demo

Интерактивная демонстрация EV-QA-Framework с визуализацией ML-детекции аномалий.

## 🚀 Как запустить

### Локально:
```bash
cd notebooks/
jupyter notebook anomaly_detection_demo.ipynb
```

### Google Colab:
1. Загрузите `anomaly_detection_demo.ipynb` в Google Colab
2. Раскомментируйте строку установки зависимостей:
   ```python
   !pip install pydantic scikit-learn pandas numpy matplotlib seaborn
   ```
3. Запустите все ячейки

## 📈 Что покажет демо:

1. **Pydantic Валидация** - строгая проверка телеметрии
2. **Генерация Данных** - 1000 точек + 50 аномалий
3. **ML Детекция** - Isolation Forest (200 estimators)
4. **Визуализации**:
   - Scatter plot: Voltage vs Temperature
   - Histogram: Anomaly Scores distribution
   - Time Series: Real-time anomaly detection
   - Pie Chart: Severity classification (CRITICAL/WARNING/INFO)
5. **Summary Report** - детальная статистика

## 🎯 Результаты:

- ✅ **~95% точность** детекции аномалий
- ✅ **Severity classification** для приоритизации
- ✅ **Красивые графики** для презентаций
- ✅ **Готово к использованию** в Google Colab

## 📸 Примеры графиков:

Notebook генерирует:
- 🟢 Зеленые точки = нормальная работа
- 🔴 Красные крестики = обнаруженные аномалии
- 🟠 Пунктирные линии = пороги безопасности

---

**Идеально для:**
- Демонстрации проекта на собеседованиях
- Презентаций для QA команд
- Обучения ML-детекции аномалий
- Публикации в LinkedIn/Medium
