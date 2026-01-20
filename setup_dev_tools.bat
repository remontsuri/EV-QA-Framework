@echo off
echo Installing tools for better EV-QA-Framework development...

REM Add Python Scripts to PATH
set PYTHON_SCRIPTS=C:\Users\vladc\AppData\Local\Programs\Python\Python312\Scripts
setx PATH "%PATH%;%PYTHON_SCRIPTS%" /M

REM Install GitHub CLI
echo Installing GitHub CLI...
winget install --id GitHub.cli

REM Install VS Code extensions (if VS Code is installed)
echo Installing VS Code extensions...
code --install-extension ms-python.python
code --install-extension ms-python.pylint
code --install-extension ms-python.black-formatter
code --install-extension ms-toolsai.jupyter
code --install-extension GitHub.vscode-pull-request-github

REM Install additional Python tools
echo Installing additional Python tools...
C:\Users\vladc\AppData\Local\Programs\Python\Python312\python.exe -m pip install --upgrade pip
C:\Users\vladc\AppData\Local\Programs\Python\Python312\python.exe -m pip install jupyter notebook ipykernel

echo Setup complete!
echo.
echo Now you can:
echo 1. Use 'gh' command for GitHub operations
echo 2. Use 'python' and 'pip' from anywhere
echo 3. Run Jupyter notebooks
echo 4. Use VS Code with Python extensions
echo.
pause