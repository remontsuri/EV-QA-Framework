#!/bin/bash
# Скрипт для загрузки EV-QA-Framework в GitHub

echo "🚀 Uploading EV-QA-Framework to GitHub..."

# Замените YOUR_USERNAME на ваш GitHub username
GITHUB_USERNAME="vladc"
REPO_NAME="EV-QA-Framework"

# Устанавливаем remote URL
git remote set-url origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git

# Пушим код
git push -u origin main

echo "✅ Upload complete!"
echo "🌐 Repository: https://github.com/$GITHUB_USERNAME/$REPO_NAME"