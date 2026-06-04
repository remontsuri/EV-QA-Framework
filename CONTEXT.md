# EV-QA-Framework — Контекст проекта для агента

## О проекте

**EV-QA-Framework** — open-source Python фреймворк для ML-powered QA батарей электромобилей. Позволяет анализировать телеметрию BMS (battery management system), детектить аномалии, прогнозировать SOH (state of health), симулировать CAN bus.

**Репозиторий**: `/opt/code/EV-QA-Framework/`
**Лицензия**: MIT
**Технологии**: Python 3.8+, scikit-learn, Pydantic v2, FastAPI, WebSocket, Chart.js, Docker

## Текущая архитектура

```
ev_qa_framework/
  __init__.py          # Экспорт модулей
  framework.py         # Основной QA engine (Pydantic validation + rule-based anomalies)
  models.py            # Pydantic модели (BatteryTelemetryModel, BatteryCellDataModel)
  config.py            # SafetyThresholds, MLConfig, FrameworkConfig
  analysis.py          # ML анализ (Isolation Forest) + EVBatteryAnalyzer + AnomalyDetector
  soh_predictor.py     # LSTM SOH prediction (требует TensorFlow — опционально)
  can_bus.py           # CAN 2.0B + J1939 (29-bit extended) симуляция и приём
  cell_balance.py      # Cell imbalance detection (статистики, outliers, trend, графики)
  thermal_runaway.py   # Thermal runaway prediction (rule-based + ML)
  metrics.py           # Prometheus метрики
  cli.py               # CLI entry point (analyze, can-demo, train-soh, dashboard, emulate)
dashboard/
  app.py               # FastAPI + WebSocket + /metrics endpoint
  grafana/dashboard.json  # Готовый dashboard для Grafana
tests/                 # 15+ тестовых файлов, ~100+ тестов
.github/workflows/
  test.yml             # CI (lint + test × 4 Python версии)
  release.yml          # CD (тег v* → build → publish в PyPI)
```

## Что реализовано ✅

1. **Core QA Engine** — Pydantic валидация + rule-based детекция (температура, скачки, напряжение)
2. **ML Anomaly Detection** — Isolation Forest (scikit-learn), adjustable contamination, severity scoring
3. **SOH Prediction** — LSTM (TensorFlow, опционально)
4. **CAN Bus** — CAN 2.0B + J1939 (29-bit extended ID, PGN 0xFEF6-0xFEF9)
5. **Cell Imbalance Detection** — статистики, outlier detection по std + absolute deviation, severity (NORMAL/WARNING/CRITICAL), trend prediction (linear regression), matplotlib графики
6. **Thermal Runaway Prediction** — rule-based + ML (IsolationForest), dT/dt анализ, настраиваемые веса
7. **Dashboard** — FastAPI + WebSocket + Chart.js, real-time telemetry streaming
8. **Prometheus Metrics** — /metrics endpoint (temperature, voltage, current, SOC, SOH, anomalies, imbalance)
9. **Grafana Dashboard** — готовый dashboard.json для импорта (5 панелей)
10. **CI/CD** — GitHub Actions (test.yml + release.yml)
11. **Docker** — Dockerfile + docker-compose.yml (GHCR образ)

## Чего не хватает / что делать дальше 🔜

### По приоритетам (от пользователя):

#### High priority
1. **PyPI publish** — релизный pipeline есть, но реально не опубликовано. Нужно:
   - Зарегистрироваться на PyPI
   - Добавить PYPI_API_TOKEN в secrets GitHub
   - Запушить тег v1.0.0
   - Убедиться что build/publish работает

2. **Thermal runaway ML модель на реальных данных** — сейчас rule-based + IsolationForest. Для LSTM нужен TensorFlow + реальные данные thermal runaway (Battery Failure Databank, Sandia LIM1TR, MIT dataset).

3. **CAN bus DBC file support** — поддержка импорта DBC файлов (Vector CANdb), генерация парсеров сигналов, совместимость с SavvyCAN/BUSMASTER форматами

#### Medium priority
4. **Physics-informed ML** — интеграция с PyBaMM для генерации симулированных данных thermal runaway и SOH degradation, semi-supervised learning

5. **J1939 broadcast / multi-packet support** — BAM (Broadcast Announce Message), Transport Protocol (TP) для сообщений длиннее 8 байт

6. **Cell imbalance в дашборде** — визуализация imbalance по времени, heatmap по ячейкам

7. **Тесты на TensorFlow SOH Predictor** — CI сейчас пропускает так как TF тяжелый. maybe nightly benchmark

8. **Cell imbalance PGN в J1939** — добавить PGN для данных о дисбалансе ячеек

#### Low priority / speculative
9. **Grafana datasource plugin** (вместо simple Prometheus) — но Prometheus уже решает задачу

10. **VW SDD (Self-Discharge Detection) spec-like модуль** — по аналогии с VW ID.4 recall

11. **LLM-based battery SOH** — эксперименты с LLM для SOH/RUL (исследовательский)

### Технический долг / известно
- TensorFlow тянет 500MB+ — вынесен в extras_require['ml']
- python-can нужен только для реальной CAN, для симуляции — нет
- Setup.py вместо pyproject.toml (надо мигрировать)
- Название `temperature` vs `temp` — анализ использует `temp`, модели используют `temperature`

## Окружение

**Инструменты**: uv (пакетный менеджер), pytest, ruff
**Проект**: `/opt/code/EV-QA-Framework/`
**Виртуальное окружение**: `.venv/` (uv)
**OS**: Linux (контейнер)
**Модель**: deepseek/deepseek-v4-flash через routerai.ru

**Стиль кода**:
- Black форматирование, isort импорты, flake8 линтинг
- Pydantic models со field_validator
- asyncio для async/await (FastAPI + WebSocket)
- Исключения через raise, не print/log
- Документация в докстрингах на русском или английском (код уже смешанный)

## Как деплоить

```bash
# Из корня репозитория
docker compose -f docker-compose.prod.yml up -d

# Или локально
python -m ev_qa_framework.cli dashboard
# → http://localhost:8000
# → http://localhost:8000/metrics
```

## Заметки

- Reddit с этого сервера заблокирован (403) — для Reddit контента использовать RSS или резервный прокси
- Стиль общения пользователя: техничный, русский, без эмодзи, без лишней теории
- Пользователь предпочитает видеть результат сразу, не планы и обсуждения
- Если пользователь сказал "стоп" — немедленно остановить все делегации, сохранить checkpoint, не запускать новые задачи без явного запроса
