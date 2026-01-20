@echo off
echo Creating GitHub repository...

REM Попробуем создать через curl (требует токен)
curl -H "Authorization: token YOUR_GITHUB_TOKEN" ^
     -X POST ^
     -d "{\"name\":\"ev-battery-qa\",\"description\":\"ML-powered QA Framework for Electric Vehicle battery testing\",\"public\":true}" ^
     https://api.github.com/user/repos

REM Если есть GitHub CLI
gh repo create ev-battery-qa --public --description "ML-powered QA Framework for Electric Vehicle battery testing" --source=.

REM Пушим код
git push -u origin main

echo Repository created: https://github.com/vladc/ev-battery-qa