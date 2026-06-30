@echo off
chcp 65001 >nul
title ML-Laser-JG1-Q355 v2.0
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

:: Set Python PATH
set "PATH=C:\Users\liuyuhe\AppData\Local\Programs\Python\Python311;C:\Users\liuyuhe\AppData\Local\Programs\Python\Python311\Scripts;%PATH%"

echo ============================================
echo   ML-Laser-JG1-Q355  v2.0
echo   Laser Cladding ML Optimization
echo ============================================
echo.

:: Check Python
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)

echo [1/2] Python:
python --version

:: Check deps
echo [2/2] Checking deps...
python -c "import pandas, numpy, sklearn" 2>nul
if errorlevel 1 (
    echo Installing deps...
    pip install pandas numpy matplotlib scikit-learn scipy opencv-python pillow shap python-docx openpyxl pyyaml tqdm joblib
)

echo.
echo ============================================
echo  Select:
echo    [1] Image Processing (segmentation + ML)
echo    [2] Paper Figures (15 charts)
echo    [3] 4-Stage Pipeline
echo    [4] Physics Model Fix
echo    [5] Hardness Predictor (interactive)
echo    [6] ML Training (FEniCSx+CCT)
echo    [0] Exit
echo ============================================
echo.
set /p choice=Enter (0-6): 

if "%choice%"=="1" goto RUN_IMAGE
if "%choice%"=="2" goto RUN_FIGURES
if "%choice%"=="3" goto RUN_PIPELINE
if "%choice%"=="4" goto RUN_FIX
if "%choice%"=="5" goto RUN_PREDICTOR
if "%choice%"=="6" goto RUN_ML
if "%choice%"=="0" goto EXIT

echo [ERROR] Invalid option
goto MENU

:RUN_IMAGE
echo.
python image/picture_processing.py
echo.
pause
goto MENU

:RUN_FIGURES
echo.
python analysis/generate_paper_figures.py
echo.
pause
goto MENU

:RUN_PIPELINE
echo.
python -m pipeline
echo.
pause
goto MENU

:RUN_FIX
echo.
python analysis/fix_issues.py
echo.
pause
goto MENU

:RUN_PREDICTOR
echo.
python analysis/hardness_predictor.py
echo.
pause
goto MENU

:RUN_ML
echo.
python analysis/ml_training_integrated.py
echo.
pause
goto MENU

:EXIT
echo Bye!
timeout /t 1 >nul
exit /b 0
