@echo off
chcp 65001 >nul
title 激光熔覆金相分析ML系统 v2.0
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

echo ============================================
echo   激光熔覆金相分析与机器学习优化系统
echo   Version 2.0
echo ============================================
echo.

:: ============================================
:: 步骤1: 检查Python环境
:: ============================================
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到Python！请确保Python 3.10+已安装并添加到系统PATH。
    pause
    exit /b 1
)

echo [1/2] Python环境检查:
python --version

:: ============================================
:: 步骤2: 检查核心依赖
:: ============================================
echo [2/2] 检查依赖...
set NEED_INSTALL=0

python -c "import pandas" 2>nul
if errorlevel 1 set NEED_INSTALL=1

python -c "import numpy" 2>nul
if errorlevel 1 set NEED_INSTALL=1

python -c "import matplotlib" 2>nul
if errorlevel 1 set NEED_INSTALL=1

python -c "import sklearn" 2>nul
if errorlevel 1 set NEED_INSTALL=1

python -c "import cv2" 2>nul
if errorlevel 1 set NEED_INSTALL=1

if %NEED_INSTALL%==1 (
    echo 正在安装核心依赖（首次运行可能需要几分钟）...
    pip install pandas numpy matplotlib scikit-learn scipy opencv-python pillow shap python-docx openpyxl pyyaml tqdm joblib
    if errorlevel 1 (
        echo.
        echo [ERROR] 依赖安装失败！请检查网络连接。
        pause
        exit /b 1
    )
    echo 依赖安装完成
) else (
    echo 依赖已就绪
)

:: ============================================
:: 步骤3: 显示菜单
:: ============================================
:MENU
echo.
echo ============================================
echo  请选择要运行的程序:
echo ============================================
echo   [1] 图像处理主程序（图像分割 + ML分析）
echo   [2] 生成论文级图表（15种高质量图表）
echo   [3] 运行四阶段优化流水线
echo   [4] 物理模型修正分析
echo   [0] 退出
echo ============================================
echo.
set /p choice=请输入选项编号 (0-4): 

if "%choice%"=="1" goto RUN_IMAGE
if "%choice%"=="2" goto RUN_FIGURES
if "%choice%"=="3" goto RUN_PIPELINE
if "%choice%"=="4" goto RUN_FIX
if "%choice%"=="0" goto EXIT

echo.
echo [ERROR] 无效选项，请重新输入！
goto MENU

:: ============================================
:: 选项1: 图像处理主程序
:: ============================================
:RUN_IMAGE
echo.
echo [3/3] 启动图像处理主程序...
echo.
echo ============================================
echo  图像处理主程序
echo  功能: 图像分割 + 定量分析 + ML模型训练
echo  输出: analysis_output/ 目录
echo ============================================
echo.

python image/picture_processing.py

if errorlevel 1 (
    echo.
    echo [ERROR] 程序运行出错！
) else (
    echo.
    echo [OK] 程序运行完成！
    echo 结果保存在 analysis_output/ 目录下
)

echo.
echo 按任意键返回菜单...
pause >nul
goto MENU

:: ============================================
:: 选项2: 生成论文级图表
:: ============================================
:RUN_FIGURES
echo.
echo [3/3] 生成论文级图表...
echo.
echo ============================================
echo  论文级图表生成器
echo  功能: 生成15种高质量论文图表（PNG格式）
echo  输出: analysis_output/figures/paper/ 目录
echo ============================================
echo.

python figures/generate_paper_figures.py

if errorlevel 1 (
    echo.
    echo [ERROR] 图表生成失败！
) else (
    echo.
    echo [OK] 图表生成完成！
    echo 图表保存在 analysis_output/figures/paper/ 目录下
)

echo.
echo 按任意键返回菜单...
pause >nul
goto MENU

:: ============================================
:: 选项3: 四阶段优化流水线
:: ============================================
:RUN_PIPELINE
echo.
echo [3/3] 启动四阶段优化流水线...
echo.
echo ============================================
echo  四阶段优化流水线
echo  阶段1: 数据整合与特征工程
echo  阶段2: 功率响应面模型构建
echo  阶段3: 性能预测模型训练
echo  阶段4: 多目标优化与Pareto分析
echo ============================================
echo.
echo 可用参数:
echo   python -m pipeline            - 执行完整四阶段流程
echo   python -m pipeline --stage N - 执行单个阶段 (N=1,2,3,4)
echo   python -m pipeline --help    - 显示帮助信息
echo.

python -m pipeline

if errorlevel 1 (
    echo.
    echo [ERROR] 流水线运行失败！
) else (
    echo.
    echo [OK] 流水线运行完成！
)

echo.
echo 按任意键返回菜单...
pause >nul
goto MENU

:: ============================================
:: 选项4: 物理模型修正分析
:: ============================================
:RUN_FIX
echo.
echo [3/3] 启动物理模型修正分析...
echo.
echo ============================================
echo  物理模型修正分析
echo  功能: 修正硬度预测公式，验证物理机制模型
echo  输出: analysis_output/ 目录
echo ============================================
echo.

python analysis/fix_issues.py

if errorlevel 1 (
    echo.
    echo [ERROR] 分析失败！
) else (
    echo.
    echo [OK] 分析完成！
)

echo.
echo 按任意键返回菜单...
pause >nul
goto MENU

:: ============================================
:: 退出
:: ============================================
:EXIT
echo.
echo 感谢使用，再见！
echo.
timeout /t 1 >nul
exit /b 0
