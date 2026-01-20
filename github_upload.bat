@echo off
echo Final setup for GitHub upload...

REM Refresh PATH
refreshenv

REM Try GitHub CLI
gh --version
if %errorlevel% neq 0 (
    echo GitHub CLI not in PATH yet. Please restart terminal and run:
    echo gh auth login --web
    echo gh repo create ev-battery-qa --public --source=. --description "ML-powered QA Framework for Electric Vehicle battery testing"
    echo git push -u origin main
    goto end
)

REM Login and create repo
echo Logging into GitHub...
gh auth login --web

echo Creating repository...
gh repo create ev-battery-qa --public --source=. --description "ML-powered QA Framework for Electric Vehicle battery testing"

echo Pushing code...
git push -u origin main

echo Done! Repository: https://github.com/vladc/ev-battery-qa

:end
pause