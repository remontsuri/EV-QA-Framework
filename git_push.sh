#!/bin/bash
# Git Push Script для публикации EV-QA-Framework на GitHub
# Для Linux/Mac: chmod +x git_push.sh && ./git_push.sh

echo "🚀 Публикация EV-QA-Framework на GitHub..."
echo ""

# Проверка текущего статуса
echo "📊 Проверка изменений..."
git status

echo ""
read -p "Продолжить? (y/n): " response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "❌ Отменено пользователем"
    exit 1
fi

# Добавление всех файлов
echo ""
echo "📦 Добавление файлов в git..."
git add .

# Коммит
echo ""
echo "💾 Создание коммита..."
git commit -m "feat: v1.0.0 - Production release with 64+ tests, Pydantic, ML enhancements

Features:
- Pydantic models for strict telemetry validation (ev_qa_models.py)
- ML anomaly detection with Isolation Forest (200 estimators)
- 64+ comprehensive tests (boundaries, anomalies, ML, validation)
- Professional documentation (README, CONTRIBUTING, examples)
- GitHub issue templates and CHANGELOG
- Comparison table vs Battery-Emulator, BatteryML, BATLab
- Demo examples and sample data

Tests:
- test_ev_qa_limits.py (23+ boundary tests)
- test_ev_qa_anomalies.py (15+ anomaly tests)
- test_ml_analysis.py (12+ ML tests)
- test_pydantic_models.py (14+ validation tests)

Docs:
- IMPROVEMENTS_REPORT.md - detailed changes
- OUTREACH_STRATEGY.md - industry outreach plan
- RELEASE_CHECKLIST.md - launch checklist
- CHANGELOG.md - version history"

# Push на GitHub
echo ""
echo "🌐 Отправка на GitHub..."
git push origin main

# Создание тега
echo ""
echo "🏷️  Создание тега v1.0.0..."
git tag -a v1.0.0 -m "Production release v1.0.0 - EV Battery QA Framework with ML"
git push origin v1.0.0

echo ""
echo "✅ УСПЕШНО! Проект опубликован на GitHub!"
echo ""
echo "📋 Следующие шаги:"
echo "1. Перейдите на https://github.com/remontsuri/EV-QA-Framework"
echo "2. Добавьте Topics в Settings -> General -> Topics"
echo "3. Создайте Release v1.0.0 через GitHub UI"
echo "4. Опубликуйте LinkedIn пост (текст в OUTREACH_STRATEGY.md)"
echo ""
echo "🎉 Готово к запуску! Удачи с продвижением!"
