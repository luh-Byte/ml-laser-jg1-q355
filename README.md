# 激光熔覆金相分析与机器学习优化系统

## 项目简介

本项目针对JG-1铁基合金Q355钢的激光熔覆工艺，基于机器学习方法实现金相组织定量分析、显微硬度预测及多目标工艺参数优化。

## 核心功能

### 机器学习模块
- **回归模型**：RFR、XGBoost、GBDT、KNN 四种模型
- **贝叶斯优化**：基于 Optuna TPE 算法的超参数自动调优
- **SHAP可解释性**：特征重要性分析与局部解释

### 深度学习模块
- **图像分割**：U-Net、DeepLabV3+、改进ResNet
- **多尺度特征融合**与注意力机制

### 多目标优化
- **BFGS拟牛顿法**：Pareto前沿求解
- **加权求和法**与**ε-约束法**

## 技术栈

| 类别 | 工具 |
|------|------|
| 机器学习 | scikit-learn, XGBoost, Optuna |
| 深度学习 | PyTorch, torchvision |
| 可解释性 | SHAP |
| 前端界面 | Streamlit |
| 数据分析 | Pandas, NumPy, Matplotlib |

## 文件结构

```
├── picture processing.py     # 主程序（含ML/DL/优化模块）
├── streamlit_app.py          # Web可视化界面
├── analysis_output/          # 分析结果输出
└── .streamlit/              # Streamlit配置
```

## 使用方法

### 1. 安装依赖
```bash
pip install scikit-learn xgboost optuna shap torch torchvision streamlit pandas numpy matplotlib openpyxl
```

### 2. 运行主程序
```bash
python "picture processing.py"
```

### 3. 启动Web界面
```bash
streamlit run streamlit_app.py
```

## 输出成果

- Pearson相关系数矩阵热力图
- 模型误差指标对比图（R²/MSE/RMSE/MAE）
- SHAP特征重要性分析
- Bayesian优化性能对比（雷达图）
- Pareto最优工艺参数

## 适用场景

- 激光熔覆工艺参数优化
- 金相组织自动化分析
- 材料性能预测与调控
