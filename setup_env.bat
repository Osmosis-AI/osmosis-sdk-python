@echo off
REM Script to set up a Python virtual environment for osmosis-ai on Windows

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -e .
pip install -r requirements.txt

if not exist .env (
    echo Creating .env file from template...
    copy .env.sample .env
    echo Please edit .env file to add your API keys
)

echo.
echo Environment setup complete!
echo To activate the environment, run: venv\Scripts\activate.bat
echo Remember to edit your .env file to add your API keys.

pause 