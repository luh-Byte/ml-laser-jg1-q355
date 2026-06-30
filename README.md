# ML-Laser-JG1-Q355

Machine Learning-based Laser Power Optimization and Microstructure-Property Coordinated Control for JG-1 Iron-based Alloy Q355 Steel

## 项目概述

本项目应用机器学习方法优化JG-1铁基自熔合金在Q355低碳钢基材上的激光熔覆工艺参数。整合定量金相、电化学阻抗谱(EIS)、X射线衍射(XRD)和磨损测试数据，建立预测模型并进行多目标优化。

## 核心特性

- **数据整合**: 统一CSV，50样本 × 79特征（金相 + EIS + XRD + 磨损）
- **物理模型校准**: Hall-Petch + Orowan公式与实测硬度校准
- **ML模型**: GBR、RFR、GPR、Ridge、Lasso、Linear多模型对比
- **交叉验证**: LOGO-CV（按功率组留一）评估真实泛化能力
- **SHAP可解释性**: 特征重要性分析揭示XRD峰位和摩擦系数是关键预测因子
- **多目标优化**: 硬度vs缺陷率的Pareto前沿分析
- **敏感性分析**: 带95%置信区间的功率敏感性分析

## 项目结构

```
ml-laser-jg1-q355/
├── pyproject.toml              # Poetry项目配置
├── poetry.lock                # 锁定依赖版本
├── .gitignore                  # Git忽略规则
├── README.md                   # 本文件
│
├── pipeline/                   # 核心ML优化流程
│   ├── __init__.py            # 模块初始化
│   ├── __main__.py            # 命令行入口
│   ├── run_pipeline.py        # 流程调度器
│   ├── data_integration.py    # 阶段1: 数据整合
│   ├── power_response_model.py # 阶段2: 功率→组织响应面
│   ├── property_model.py      # 阶段3: 组织→性能模型
│   ├── optimized_pipeline.py   # 阶段4: 链式优化
│   ├── diagnose_overfitting.py # 过拟合诊断
│   └── fix_overfitting.py      # 过拟合解决方案对比
│
├── analysis/                   # 材料科学分析
│   ├── material_science_analysis.py  # 材料科学验证
│   ├── thermodynamics_analysis.py   # 热力学链分析
│   ├── univariate_analysis.py       # 单变量分析
│   └── fix_issues.py                # 物理模型修正
│
├── image/                      # 图像处理模块
│   ├── picture_processing.py  # 金相图像分割与分析
│   └── microsam_integration.py # MicroSAM集成
│
├── figures/                    # 图表生成
│   └── generate_paper_figures.py # 论文级图表生成
│
├── knowledge/                  # 知识库模块
│   ├── knowledge_base.py      # 知识库管理
│   ├── kb_search.py           # 知识检索
│   └── chat_memory.py         # 对话记忆
│
├── utils/                      # 工具模块
│   ├── plot_style.py          # 统一绘图风格
│   ├── cache_manager.py        # 缓存管理
│   ├── merge_experimental_data.py  # 实验数据合并
│   ├── clean_analysis_output.py    # 输出清理
│   ├── orthogonal_design.py        # 正交设计
│   └── unify_data_format.py        # 格式统一
│
├── tests/                      # 测试模块
│   └── test_search.py         # 检索测试
│
├── knowledge_base/            # 知识库文档
│   └── raw_docs/
│       ├── laser_cladding/    # 激光熔覆文档
│       └── material_ml/       # 材料ML文档
│
├── analysis_output/           # 分析结果输出（git忽略）
│   ├── data_full.csv          # 完整数据集
│   ├── *.pkl                  # 模型文件
│   └── figures/               # 图表输出
│       └── *.png
│
├── 启动.bat                    # Windows快速启动
└── CODE_WIKI.md               # 代码文档
```

## 安装

### 环境要求
- Python 3.11–3.13
- [Poetry](https://python-poetry.org/) (推荐) 或 pip

### Poetry安装（推荐）
```bash
cd ml-laser-jg1-q355
poetry install
poetry shell
```

### pip安装
```bash
cd ml-laser-jg1-q355
pip install -e .
```

## 使用方法

### 运行完整Pipeline
```bash
python -m pipeline
```

### 运行单个阶段
```bash
python -m pipeline --stage 1  # 数据整合
python -m pipeline --stage 2  # 功率响应面
python -m pipeline --stage 3  # 性能建模
python -m pipeline --stage 4  # 多目标优化
```

### 查看帮助
```bash
python -m pipeline --help
```

### 运行特定分析脚本
```bash
# 过拟合诊断
python -m pipeline.diagnose_overfitting

# 过拟合解决方案对比
python -m pipeline.fix_overfitting

# 材料科学分析
python analysis/material_science_analysis.py
```

## 关键结果

| 指标 | 值 |
|------|-----|
| 硬度范围（实测） | 224–385 HV |
| 最优模型（LOGO-CV） | RFR (R²=-0.72) |
| 最优功率（最大硬度） | ~1670 W |
| 最优功率（最小缺陷） | 900 W |

**注意**: 真实的LOGO-CV R²为负值，这反映了数据的局限性：
- 只有4个功率水平，模型需要外推到全新功率
- 微观组织特征在组内变异大，组间差异小

### SHAP特征重要性（Top 5）
1. `xrd_main_peak_2theta` — XRD主峰位置（相变指标）
2. `wear_friction_steady` — 稳态摩擦系数
3. `xrd_peak_44_area` — XRD 44°峰面积
4. `eis_Rct_ohm` — 电荷转移电阻
5. `熔覆层平均晶粒尺寸(μm)` — 晶粒尺寸

## 引用

如在研究中使用了本代码，请引用：
```
Wang, Y. (2026). ML-Laser-JG1-Q355: Machine Learning-based Laser Power Optimization
for JG-1 Iron-based Alloy Q355 Steel. GitHub Repository.
```

## 许可证

MIT License
