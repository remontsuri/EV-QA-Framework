# Launch script for the EV-QA Dashboard and Emulator

Write-Host "🚀 Launching EV-QA Real-time Monitoring System..." -ForegroundColor Cyan

# Start FastAPI Dashboard in background
Write-Host "📦 Installing dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements.txt

Write-Host "🌐 Starting Dashboard on http://localhost:8000" -ForegroundColor Yellow
Start-Process python -ArgumentList "dashboard/app.py" -WindowStyle Normal -NoNewWindow

# Wait a few seconds for the server to start
Start-Sleep -s 3

# Start CAN Emulator
Write-Host "⚡ Starting CAN Bus Emulator..." -ForegroundColor Yellow
python scripts/can_emulator.py
