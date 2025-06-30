@echo off
echo AI LLM FineTuner Setup
echo ======================

echo Installing Python dependencies...
python -m pip install -r requirements.txt

echo.
echo Running setup script...
python setup.py

echo.
echo Setup complete! Press any key to exit...
pause >nul
