# Git Push Script для публикации EV-QA-Framework на GitHub
# Запустите этот файл в PowerShell: .\git_push.ps1

Write-Host "🚀 Публикация EV-QA-Framework на GitHub..." -ForegroundColor Green
Write-Host ""

# Проверка текущего статуса
Write-Host "📊 Проверка изменений..." -ForegroundColor Yellow
git status

Write-Host ""
Write-Host "Продолжить? (Y/N): " -ForegroundColor Cyan -NoNewline
$response = Read-Host

if ($response -ne 'Y' -and $response -ne 'y') {
    Write-Host "❌ Отменено пользователем" -ForegroundColor Red
    exit
}

# Добавление всех файлов
Write-Host ""
Write-Host "📦 Добавление файлов в git..." -ForegroundColor Yellow
git add .

# Коммит
Write-Host ""
Write-Host "💾 Создание коммита..." -ForegroundColor Yellow
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
Write-Host ""
Write-Host "🌐 Отправка на GitHub..." -ForegroundColor Yellow
git push origin main

# Создание тега
Write-Host ""
Write-Host "🏷️  Создание тега v1.0.0..." -ForegroundColor Yellow
git tag -a v1.0.0 -m "Production release v1.0.0 - EV Battery QA Framework with ML"
git push origin v1.0.0

Write-Host ""
Write-Host "✅ УСПЕШНО! Проект опубликован на GitHub!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Следующие шаги:" -ForegroundColor Cyan
Write-Host "1. Перейдите на https://github.com/remontsuri/EV-QA-Framework"
Write-Host "2. Добавьте Topics в Settings -> General -> Topics"
Write-Host "3. Создайте Release v1.0.0 через GitHub UI"
Write-Host "4. Опубликуйте LinkedIn пост (текст в OUTREACH_STRATEGY.md)"
Write-Host ""
Write-Host "🎉 Готово к запуску! Удачи с продвижением!" -ForegroundColor Green
