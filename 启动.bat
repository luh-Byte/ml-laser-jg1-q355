@echo off
chcp 65001 >nul
title 激光熔覆金相分析ML系统

echo ============================================
echo   激光熔覆金相分析与机器学习优化系统
echo ============================================
echo.

cd /d "%~dp0"

:: 检查虚拟环境
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境...
    python -m venv .venv
    echo 虚拟环境创建完成
)

:: 检查依赖
echo [2/3] 检查依赖...
.venv\Scripts\python.exe -c "import streamlit" 2>nul
if errorlevel 1 (
    echo 正在安装依赖...
    .venv\Scripts\pip.exe install streamlit pandas numpy matplotlib scikit-learn xgboost optuna shap torch torchvision openpyxl
    echo 依赖安装完成
)

:: 启动Streamlit（单窗口运行）
echo [3/3] 启动Web界面...
echo.
echo 系统启动中，请稍候...
echo.

start http://localhost:8501

echo ============================================
echo  服务已启动！
echo  本地地址: http://localhost:8501
echo  浏览器已自动打开
echo ============================================
echo.
echo 按 Ctrl+C 可停止服务
echo ============================================
echo.

.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8501
