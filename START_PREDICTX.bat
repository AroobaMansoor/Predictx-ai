@echo off
title PredictX AI
cd /d "%~dp0"
echo ============================================
echo   PredictX AI - starting...
echo ============================================

where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ERROR] Python nahi mila. python.org se Python 3.11 ya 3.12 install karein
    echo aur install ke waqt "Add Python to PATH" ka box zaroor tick karein.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment ban raha hai...
    python -m venv .venv
    if errorlevel 1 ( echo [ERROR] venv nahi bana. & pause & exit /b 1 )
)

".venv\Scripts\python.exe" -c "import joblib, streamlit, sklearn, xgboost, lightgbm, shap, imblearn, plotly" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Libraries install ho rahi hain - 5 se 10 minute lagenge, internet on rakhein...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo Normal install fail hua - naye Python version wali list try ho rahi hai ^(CatBoost aur PyYAML ke baghair^)...
        ".venv\Scripts\python.exe" -m pip install -r requirements-py315.txt
    )
    ".venv\Scripts\python.exe" -c "import joblib, streamlit, sklearn, xgboost, lightgbm, shap, imblearn, plotly" >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [ERROR] Libraries install nahi huin. Upar wala error ka screenshot bhejein.
        pause
        exit /b 1
    )
)

echo.
echo Saved model check ho raha hai...
".venv\Scripts\python.exe" -c "from src.inference import PredictXPredictor; PredictXPredictor()" >nul 2>nul
if errorlevel 1 (
    echo Model aapke version par load nahi hua - naya model train ho raha hai (15-25 minute^)...
    ".venv\Scripts\python.exe" train.py
    if errorlevel 1 ( echo [ERROR] Training mein masla aaya. Screenshot bhejein. & pause & exit /b 1 )
)

echo.
echo Dashboard khul raha hai: http://localhost:8501
echo Agar browser khud na khule to Chrome mein localhost:8501 likhein.
echo Band karne ke liye yeh kaala window close kar dein.
".venv\Scripts\python.exe" -m streamlit run app.py
pause
