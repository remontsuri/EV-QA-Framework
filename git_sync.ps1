# Git Sync Script - исправляет конфликты и пушит изменения

Write-Host "🔄 Синхронизация с GitHub..." -ForegroundColor Yellow

# Pull с rebase
Write-Host "`n📥 Получение изменений с GitHub..." -ForegroundColor Cyan
git pull origin main --rebase

if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️ Есть конфликты или проблемы с pull" -ForegroundColor Yellow
    Write-Host "Попробуем merge вместо rebase..." -ForegroundColor Yellow
    
    # Отменяем rebase если он начался
    git rebase --abort 2>$null
    
    # Пробуем обычный pull
    git pull origin main --no-rebase
}

# Добавляем все изменения
Write-Host "`n📦 Добавление новых файлов..." -ForegroundColor Cyan
git add .

# Проверяем, есть ли что коммитить
$status = git status --porcelain
if ($status) {
    Write-Host "`n💾 Создание коммита..." -ForegroundColor Cyan
    git commit -m "feat: Add Jupyter demo, social media posts, and final documentation

- Interactive Jupyter notebook with ML anomaly detection visualization
- Ready-to-use LinkedIn and Reddit posts  
- Final status report and documentation
- Graphs: Voltage vs Temp, Time Series, Severity classification"
}
else {
    Write-Host "`n✅ Нет новых изменений для коммита" -ForegroundColor Green
}

# Push
Write-Host "`n🌐 Отправка на GitHub..." -ForegroundColor Cyan
git push origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ УСПЕШНО! Все изменения отправлены на GitHub!" -ForegroundColor Green
    Write-Host "`n📋 Проверьте: https://github.com/remontsuri/EV-QA-Framework" -ForegroundColor Cyan
}
else {
    Write-Host "`n❌ Ошибка при push. Проверьте вывод выше." -ForegroundColor Red
    Write-Host "Возможно, нужно вручную разрешить конфликты." -ForegroundColor Yellow
}

Write-Host "`n🎉 Готово!" -ForegroundColor Green
