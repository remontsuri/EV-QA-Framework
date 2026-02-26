@echo off
echo ========================================
echo Running New Tests (Steps 1 and 2)
echo ========================================
echo.

echo [Step 1] Testing Configuration Module...
pytest tests/test_config.py -v
echo.

echo [Step 2] Testing Model Persistence...
pytest tests/test_model_persistence.py -v
echo.

echo ========================================
echo All new tests completed!
echo ========================================
pause
