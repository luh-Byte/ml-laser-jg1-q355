# ML-Laser-JG1-Q355 Code Wiki

> **项目名称**: 基于机器学习的激光功率优化及JG-1铁基合金Q355钢组织性能协同调控研究
> **版本**: v2.0
> **最后更新**: 2026-06-29

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目架构](#2-项目架构)
3. [目录结构](#3-目录结构)
4. [核心模块详解](#4-核心模块详解)
5. [关键类与函数](#5-关键类与函数)
6. [数据流与调用关系](#6-数据流与调用关系)
7. [依赖关系](#7-依赖关系)
8. [项目运行方式](#8-项目运行方式)
9. [配置说明](#9-配置说明)
10. [输出文件说明](#10-输出文件说明)

---

## 1. 项目概述

### 1.1 项目背景

本项目应用机器学习方法优化JG-1铁基自熔性合金在Q355低碳钢基材上的激光熔覆工艺参数。项目整合了定量金相、电化学阻抗谱(EIS)、X射线衍射(XRD)和摩擦磨损试验数据，构建预测模型并进行多目标优化。

### 1.2 核心功能

| 功能模块 | 描述 | 技术栈 |
|---------|------|--------|
| 图像分割 | 金相组织自动分割（熔覆层/析出相/缺陷） | OpenCV传统图像处理 |
| 定量分析 | 组织面积占比、晶粒尺寸、稀释率等 | Pandas数据统计 |
| 数据整合 | 多源实验数据整合（硬度/磨损/EIS/XRD） | 数据清洗+特征工程 |
| 机器学习 | 回归预测（GBR/RFR/GPR） | scikit-learn, XGBoost |
| 超参数优化 | Bayesian优化（LOO-CV） | 贝叶斯优化 |
| 可解释性 | SHAP特征重要性分析 | SHAP |
| 多目标优化 | Pareto前沿求解 | BFGS拟牛顿法 |
| 物理模型 | Hall-Petch/Orowan强化机制 | 材料科学理论 |
| 知识管理 | 知识库搜索与对话记忆 | ChromaDB + Embedding |
| 论文图表 | 15种高质量论文级图表 | Matplotlib优化风格 |

### 1.3 技术路线

```
实验数据采集 → 图像分割与定量表征 → 多源数据整合 → 特征工程
     ↓
机器学习建模(GBR/RFR/GPR) ← 物理模型校准(Hall-Petch+Orowan)
     ↓
SHAP可解释性分析 → 多目标Pareto优化 → 实验验证方案
```

---

## 2. 项目架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         入口层 (Entry Layer)                         │
│  python -m pipeline  ──►  pipeline/run_pipeline.py                 │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│  pipeline/        │     │  image/           │     │  figures/         │
│  四阶段流水线     │     │  图像处理模块      │     │  图表生成模块      │
└─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
          │                         │                         │
          ├─────────────────────────┴─────────────────────────┤
          │                                                     │
          ▼                                                     ▼
┌───────────────────┐                           ┌───────────────────┐
│  analysis/        │                           │  knowledge/       │
│  材料科学分析     │                           │  知识管理模块      │
└─────────┬─────────┘                           └─────────┬─────────┘
          │                                                     │
          └─────────────────────────┬───────────────────────────┘
                                    ▼
                          ┌───────────────────┐
                          │  utils/           │
                          │  工具函数模块      │
                          └───────────────────┘
```

### 2.2 设计原则

- **模块化**: 按功能划分为独立模块，各模块职责单一
- **可复现性**: 固定随机种子(42)，模型缓存机制
- **可扩展性**: 支持新数据、新模型、新图表的扩展
- **数据驱动**: 从实验数据出发，物理模型与数据驱动结合

---

## 3. 目录结构

```
ml-laser-jg1-q355/
├── pyproject.toml                  # Poetry项目配置
├── poetry.lock                     # 锁定依赖版本
├── kb_config.yaml                  # 知识库配置
├── 启动.bat                         # Windows一键启动脚本
│
├── pipeline/                       # 四阶段优化流水线
│   ├── __init__.py                # 模块初始化
│   ├── __main__.py                # 命令行入口 (python -m pipeline)
│   ├── run_pipeline.py             # 流水线统一入口
│   ├── data_integration.py         # 阶段1：数据整合
│   ├── power_response_model.py     # 阶段2：功率响应面建模
│   ├── property_model.py           # 阶段3：性能预测模型
│   ├── optimized_pipeline.py       # 阶段4：链式优化
│   ├── diagnose_overfitting.py     # 过拟合诊断脚本
│   └── fix_overfitting.py          # 过拟合解决方案对比
│
├── image/                          # 图像处理模块
│   ├── __init__.py
│   ├── picture_processing.py       # 金相图像定量分析主程序
│   └── microsam_integration.py     # MicroSAM模型集成
│
├── analysis/                       # 材料科学分析模块
│   ├── __init__.py
│   ├── material_science_analysis.py # 物理性质合理性分析
│   ├── thermodynamics_analysis.py   # 热力学链分析
│   ├── univariate_analysis.py       # 单变量分析
│   └── fix_issues.py              # 物理模型修正
│
├── figures/                        # 图表生成模块
│   ├── __init__.py
│   └── generate_paper_figures.py   # 论文级图表生成器
│
├── knowledge/                      # 知识管理模块
│   ├── __init__.py
│   ├── knowledge_base.py           # 知识库管理(ChromaDB)
│   ├── kb_search.py               # 知识搜索功能
│   └── chat_memory.py             # 对话记忆管理
│
├── utils/                          # 工具函数模块
│   ├── __init__.py
│   ├── plot_style.py               # 统一绘图风格配置
│   ├── cache_manager.py            # 缓存与结果管理
│   ├── merge_experimental_data.py  # 实验数据合并
│   ├── unify_data_format.py        # 数据格式统一
│   ├── clean_analysis_output.py    # 输出清理
│   ├── orthogonal_design.py        # 正交实验设计
│   ├── copy_to_desktop.py          # 桌面复制工具
│   └── get-pip.py                  # pip安装脚本
│
├── tests/                          # 测试模块
│   ├── __init__.py
│   └── test_search.py              # 搜索功能测试
│
├── knowledge_base/                  # 知识库文件
│   └── raw_docs/                   # 原始文档
│       ├── laser_cladding/          # 激光熔覆相关
│       └── material_ml/            # 材料机器学习相关
│
├── analysis_output/                 # 分析结果输出 (git-ignored)
│   ├── data_full.csv               # 整合后的完整数据集
│   ├── figures/                    # 图表输出
│   │   ├── paper/                  # 论文级图表(PNG)
│   │   └── segmentation/           # 分割结果图像
│   ├── ml_results/                 # ML评估报告
│   ├── model_cache/                # 模型缓存
│   ├── power_response_models.pkl   # 阶段2模型
│   ├── property_models.pkl         # 阶段3模型
│   └── optimization_results.pkl    # 阶段4结果
│
└── CODE_WIKI.md                    # 代码文档
```

---

## 4. 核心模块详解

### 4.1 pipeline 模块 — 四阶段优化流水线

#### 模块职责
pipeline模块是项目的核心，按顺序执行四个阶段的数据处理和建模流程，实现从原始实验数据到优化结果的完整链路。

#### 四阶段流程

| 阶段 | 脚本 | 功能描述 | 输入 | 输出 |
|-----|------|---------|------|------|
| 阶段1 | [data_integration.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/pipeline/data_integration.py) | 多源实验数据整合 | 金相CSV + 硬度DOCX + 磨损TXT + EIS数据 + XRD数据 | `完整实验数据汇总.csv` (56行×52列) |
| 阶段2 | [power_response_model.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/pipeline/power_response_model.py) | 功率→组织响应面建模 | `完整实验数据汇总.csv` | GPR响应面模型 + 物理模型校准结果 |
| 阶段3 | [property_model.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/pipeline/property_model.py) | 组织→性能预测模型 | `完整实验数据汇总.csv` | GBR/RFR/GPR模型 + SHAP分析 |
| 阶段4 | [optimized_pipeline.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/pipeline/optimized_pipeline.py) | 多目标优化验证 | 阶段3模型 | Pareto前沿 + 验证方案 |

#### 阶段1：数据整合
- 读取4类实验数据：显微硬度(DOCX)、摩擦磨损(TXT)、电化学阻抗(EIS)、X射线衍射(XRD)
- 与金相定量表征CSV按功率匹配合并
- 添加工艺参数（扫描速度、送粉速率）和化学成分数据
- 生成数据整合报告

#### 阶段2：功率响应面建模
- 建立GPR(高斯过程回归)响应面：激光功率 → 6项微观组织特征
- 使用Leave-One-Power-Out交叉验证
- 基于实测硬度校准物理模型(Hall-Petch + Orowan)
- 建立功率→硬度直接GPR响应面

#### 阶段3：性能预测模型
- 目标变量：实测显微硬度(`mh_mean_hv`)
- 特征组：金相+电化学+XRD+磨损+工艺参数+交叉项+Hall-Petch项+热输入
- 三种模型对比：GBR(梯度提升)、RFR(随机森林)、GPR(高斯过程)
- LOO(留一法)交叉验证
- SHAP特征重要性分析 + 偏依赖图 + 功率敏感性分析

#### 阶段4：链式优化
- 基于GBR模型的多目标优化（最大化硬度 + 最小化缺陷）
- 差分进化算法 + 网格搜索Pareto前沿
- 3D参数空间优化（功率、扫描速度、送粉速率）
- 敏感性分析 + Bootstrap置信区间
- 生成实验验证方案

---

### 4.2 image 模块 — 图像处理与定量分析

#### 模块职责
负责金相图像的读取、预处理、分割、定量表征和结果输出。

#### 核心功能
- **硬件加速配置**: CPU多线程 + OpenCV OpenCL GPU加速
- **图像读取**: 支持TIFF多种格式（I;16, I, L, RGB, RGBA, P）
- **XML元数据解析**: 蔡司显微镜`.tif_meta.xml`文件，提取像素尺寸校准
- **图像分割**: 5类组织分类（背景/熔覆层/析出相/气孔/裂纹）
- **定量表征**: 面积占比、晶粒尺寸(等效圆直径)、稀释率、宽高比
- **MicroSAM集成**: 基于深度学习的语义分割

#### 组织分类体系

| 类别ID | 类别名称 | 颜色(RGB) | 说明 |
|--------|---------|-----------|------|
| 0 | 背景/基体 | (120, 120, 120) | 灰色 |
| 1 | 熔覆层组织 | (100, 160, 110) | 绿色 |
| 2 | 析出相/碳化物 | (255, 200, 0) | 黄色 |
| 3 | 气孔缺陷 | (255, 30, 30) | 红色 |
| 4 | 裂纹缺陷 | (150, 0, 200) | 紫色 |

---

### 4.3 analysis 模块 — 材料科学分析

#### 模块职责
从材料科学角度验证数据合理性，进行物理机制分析。

#### 子模块

| 脚本 | 功能 |
|------|------|
| [material_science_analysis.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/analysis/material_science_analysis.py) | 物理性质合理性验证（晶粒尺寸、稀释率、硬度等） |
| [thermodynamics_analysis.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/analysis/thermodynamics_analysis.py) | 热力学链分析 |
| [fix_issues.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/analysis/fix_issues.py) | 物理模型修正（Hall-Petch + Orowan强化机制） |

#### 物理模型修正内容
- Issue1: 硬度预测公式系数错误 → 基于Hall-Petch关系重新设计
- Issue2: 组织比例求和列名不匹配 → 修正列名
- 强化机制: 细晶强化 + Orowan沉淀强化 + 固溶强化

---

### 4.4 figures 模块 — 论文级图表生成

#### 模块职责
生成符合SCI期刊规范的高质量论文图表，统一风格和配色。

#### 图表清单（共15种）

| 编号 | 图表名称 | 说明 | 输出文件 |
|-----|---------|------|---------|
| fig1 | 相关性矩阵热力图 | Pearson相关系数 | `fig1_correlation_matrix.png` |
| fig2 | 硬度-功率 + 晶粒尺寸-功率 | 双面板折线图 | `fig2_power_trends.png` |
| fig3 | 预测-实验散点图 + 残差分布 | 模型预测评估 | `fig3_prediction_scatter.png` |
| fig4 | 性能指标柱状图 | 稀释率/气孔/裂纹 | `fig4_performance_bars.png` |
| fig5 | 相体积分数饼图 | 4种功率对比 | `fig5_phase_pie.png` |
| fig6 | 晶粒尺寸分布密度图 | KDE+直方图 | `fig6_grain_size_distribution.png` |
| fig7 | 多性能雷达图 | 6维度对比 | `fig7_radar_chart.png` |
| fig8 | 箱线图对比 | 4参数2×2布局 | `fig8_boxplots.png` |
| fig9 | 散点矩阵图 | 4变量相关性 | `fig9_scatter_matrix.png` |
| fig10 | 金相原图vs分割掩码 | 图像分割效果 | `fig10_segmentation.png` |
| fig11 | SHAP特征重要性 | 条形图 | `fig11_shap_importance.png` |
| fig12 | 技术流程图 | 工作流示意 | `fig12_workflow.png` |
| fig13 | Pareto优化散点图 | 硬度vs稀释率 | `fig13_pareto_front.png` |
| fig14 | 数据汇总表格 | Mean ± SEM | `fig14_data_table.png` |
| fig15 | 参数跨功率变化热力图 | 归一化对比 | `fig15_heatmap.png` |

#### 样式特点
- 蓝白渐变背景
- 2.5pt黑色粗边框
- 刻度朝内
- 明亮配色方案（色盲友好）
- 子图编号标签 (a)(b)(c)...
- 输出格式: PNG（600dpi）

---

### 4.5 knowledge 模块 — 知识管理系统

#### 模块职责
提供本地离线的材料科学知识库检索和对话记忆管理功能。

#### 核心组件

| 组件 | 文件 | 功能 |
|-----|------|------|
| 知识库管理 | [knowledge_base.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/knowledge/knowledge_base.py) | 文档导入、文本分块、向量化、ChromaDB存储 |
| 知识搜索 | [kb_search.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/knowledge/kb_search.py) | 相似度检索、Top-N结果返回 |
| 对话记忆 | [chat_memory.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/knowledge/chat_memory.py) | 短期记忆+长期摘要、SQLite存储 |

#### 知识库集合(Collections)

| 集合名称 | 描述 | 文档路径 |
|---------|------|---------|
| laser_cladding | JG-1铁基合金、激光熔覆、堆焊工艺 | `knowledge_base/raw_docs/laser_cladding/` |
| metallography | 金相图谱、图像分割、定量分析 | `knowledge_base/raw_docs/metallography/` |
| material_ml | XGBoost/RFR/SHAP性能预测 | `knowledge_base/raw_docs/material_ml/` |
| literature | 论文文献、实验数据 | `knowledge_base/raw_docs/literature/` |

#### 技术参数
- **嵌入模型**: paraphrase-multilingual-MiniLM-L12-v2 (本地离线)
- **向量数据库**: ChromaDB
- **文本分块**: chunk_size=512, overlap=100
- **检索阈值**: similarity_threshold=-1.0
- **返回数量**: Top 3

---

### 4.6 utils 模块 — 工具函数

#### 模块职责
提供跨模块共享的工具函数和配置。

#### 子模块

| 文件 | 功能 |
|------|------|
| [plot_style.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/utils/plot_style.py) | 统一绘图风格配置（rcParams、配色方案、工具函数） |
| [cache_manager.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/utils/cache_manager.py) | 缓存管理、结果备份、文件清理 |
| [copy_to_desktop.py](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/utils/copy_to_desktop.py) | 结果文件复制到桌面 |

#### plot_style 核心工具函数
- `setup_plot_style()`: 设置统一rcParams
- `style_axes()`: 统一坐标轴样式
- `create_gradient_rect()`: 蓝白渐变背景
- `add_subplot_label()`: 添加子图编号(a)(b)...
- `calc_sem()`: 计算标准误(Mean ± SEM)
- `save_fig()`: 统一图像保存

---

## 5. 关键类与函数

### 5.1 pipeline 关键函数

#### run_pipeline.py

| 函数 | 签名 | 功能 |
|-----|------|------|
| `run_pipeline()` | `() -> bool` | 顺序执行全部4个阶段，成功返回True |
| `run_single_stage(stage_index)` | `(int) -> bool` | 执行单个阶段(1-4) |
| `show_help()` | `() -> None` | 显示命令行帮助信息 |

#### data_integration.py

| 函数 | 签名 | 功能 |
|-----|------|------|
| `read_microhardness()` | `() -> dict` | 从DOCX读取显微硬度数据(HV=xxx格式) |
| `read_wear_data()` | `() -> dict` | 从TXT读取摩擦磨损数据(时间,摩擦系数) |
| `read_eis_data()` | `() -> dict` | 读取EIS拟合数据，计算Rs/Rct/|Z|max |
| `read_xrd_data()` | `() -> dict` | 读取XRD数据，计算主峰位置和峰面积 |
| `main()` | `() -> None` | 主流程：读取4类数据 + 合并到金相CSV |
| `generate_simulated_data()` | `() -> str` | 生成L25(5³)正交模拟数据(代码验证用) |

#### power_response_model.py

| 函数 | 签名 | 功能 |
|-----|------|------|
| `load_data()` | `() -> DataFrame` | 加载完整实验数据汇总CSV |
| `build_power_response_models(df)` | `(DataFrame) -> dict` | 建立6个GPR功率→组织响应面 |
| `plot_power_response(df, results)` | `(DataFrame, dict) -> None` | 绘制6面板响应面图 |
| `calibrate_physics_model(df)` | `(DataFrame) -> dict` | 校准Hall-Petch+Orowan物理模型 |
| `build_power_property_direct(df)` | `(DataFrame) -> dict` | 建立功率→硬度直接GPR |

**物理模型公式**:
```
HV = a + b/√d + c·√carbide + e·porosity + f·crack
  a — 基体能级
  b — Hall-Petch系数(细晶强化)
  c — Orowan系数(沉淀强化)
  e — 气孔影响系数
  f — 裂纹影响系数
```

#### property_model.py

| 函数 | 签名 | 功能 |
|-----|------|------|
| `load_and_prepare()` | `() -> (DataFrame, list, str, dict)` | 加载数据 + 特征工程 |
| `train_models(df, features, target)` | `(DataFrame, list, str) -> tuple` | 训练GBR/RFR/GPR + LOO-CV |
| `shap_analysis(model, X, feature_names)` | `(model, array, list) -> list` | SHAP特征重要性分析 |
| `plot_partial_dependence(...)` | `... -> None` | 绘制关键特征偏依赖图 |
| `plot_model_comparison(results, y, df)` | `(dict, array, DataFrame) -> None` | 三面板模型对比图 |
| `sensitivity_analysis(...)` | `... -> (array, array)` | 功率变化敏感性分析 |

**特征组构成**:
- metallography (6项): 熔覆层%、析出相%、气孔%、裂纹%、晶粒尺寸、稀释率
- eis (4项): Rs, Rct, |Z|max, θmin
- xrd (3项): 主峰2θ, 主峰强度, 44°峰面积
- wear (2项): 稳态摩擦系数, 摩擦系数标准差
- process (2项): 扫描速度, 送粉速率
- 衍生特征: power×组织交叉项, power×工艺交叉项, Hall-Petch项, 热输入

#### optimized_pipeline.py

| 函数 | 签名 | 功能 |
|-----|------|------|
| `load_models()` | `() -> (model, scaler, list)` | 加载阶段3训练好的GBR模型 |
| `make_predictor(model, scaler, features, df_median)` | `... -> callable` | 封装预测函数(P,Vs,Vf)→硬度 |
| `multi_objective_optimize(...)` | `... -> list` | 差分进化+网格搜索Pareto前沿 |
| `sensitivity_analysis(...)` | `... -> DataFrame` | ±200W功率敏感性分析 |
| `confidence_analysis(...)` | `... -> dict` | Bootstrap 95%置信区间 |
| `generate_verification_plan(...)` | `... -> str` | 生成实验验证方案Markdown |

### 5.2 image 关键函数

#### picture_processing.py

| 函数/类 | 类型 | 功能 |
|---------|------|------|
| `configure_hardware_acceleration()` | 函数 | 配置CPU多线程 + OpenCL GPU加速 |
| `parse_zeiss_xml_metadata(xml_path)` | 函数 | 解析蔡司XML元数据，提取像素尺寸 |
| `find_xml_for_tiff(tiff_path)` | 函数 | 为TIFF图像查找对应XML文件 |
| `CLASS_DICT` | 常量 | 5类组织分类字典 |
| `CLASS_COLOR` | 常量 | 各类别RGB颜色映射 |
| `MAG_DICT` | 常量 | 放大倍数字典(50x~1000x) |

### 5.3 knowledge 关键类

#### knowledge_base.py

| 类/函数 | 类型 | 功能 |
|---------|------|------|
| `SentenceTransformerEmbedding` | 类 | 基于sentence-transformers的本地嵌入函数 |
| `read_file(filepath)` | 函数 | 读取TXT/MD/CSV/DOCX/PDF文件 |
| `chunk_text(text, chunk_size, overlap)` | 函数 | 文本智能分块(按句号/换行切分) |

---

## 6. 数据流与调用关系

### 6.1 主数据流图

```
原始实验数据
    │
    ├── 金相图像(TIFF) ──────┐
    ├── 显微硬度(DOCX) ──────┤
    ├── 磨损数据(TXT) ───────┤
    ├── EIS数据 ─────────────┤
    └── XRD数据 ─────────────┤
                            │
                            ▼
                ┌─────────────────────┐
                │  image/              │
                │  picture_processing  │
                │  (图像分割+定量)      │
                └─────────┬───────────┘
                          │ 金相定量表征数据汇总.csv
                          ▼
                ┌─────────────────────┐
                │  pipeline/           │
                │  data_integration    │
                │  (阶段1: 数据整合)    │
                └─────────┬───────────┘
                          │ 完整实验数据汇总.csv
                          ▼
                ┌─────────────────────┐
                │  pipeline/           │
                │  power_response_model│
                │  (阶段2: 响应面建模)  │
                └─────────┬───────────┘
                          │ power_response_models.pkl
                          ▼
                ┌─────────────────────┐
                │  pipeline/           │
                │  property_model      │
                │  (阶段3: 性能建模)    │
                └─────────┬───────────┘
                          │ property_models.pkl
                          ▼
                ┌─────────────────────┐
                │  pipeline/           │
                │  optimized_pipeline  │
                │  (阶段4: 优化验证)    │
                └─────────┬───────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
    optimization_results.pkl   实验验证方案.md
    pareto_optimization.png
    power_sensitivity_detailed.png
```

### 6.2 模块调用关系

```
run.py
  └── pipeline/run_pipeline.py
        ├── pipeline/data_integration.py (阶段1)
        │     └── (读取 DOCX/TXT/XRD/EIS 数据)
        ├── pipeline/power_response_model.py (阶段2)
        │     └── (GPR + 物理模型校准)
        ├── pipeline/property_model.py (阶段3)
        │     ├── utils/plot_style.py
        │     └── (GBR/RFR/GPR + SHAP)
        └── pipeline/optimized_pipeline.py (阶段4)
              └── (差分进化 + Pareto)

image/picture_processing.py (独立运行)
  └── image/microsam_integration.py

figures/generate_paper_figures.py (独立运行)
  └── utils/plot_style.py

knowledge/knowledge_base.py (独立运行)
  ├── kb_config.yaml
  └── ChromaDB

analysis/material_science_analysis.py (独立运行)
utils/cache_manager.py (被run.py调用备份)
```

### 6.3 文件依赖链

```
输入数据:
  data/900W/*.tif (+ XML)
  data/1200W/*.tif (+ XML)
  data/1500W/*.tif (+ XML)
  data/1800W/*.tif (+ XML)
  data/microhardness-data/*.docx
  data/wear-data/*.txt
  data/electrochemical-impedance/*.txt
  data/xrd-data/*.txt

中间产物:
  analysis_output/金相定量表征数据汇总.csv
    ↓
  analysis_output/完整实验数据汇总.csv
    ↓
  analysis_output/power_response_models.pkl
    ↓
  analysis_output/property_models.pkl
    ↓
  analysis_output/optimization_results.pkl
```

---

## 7. 依赖关系

### 7.1 核心依赖

| 依赖包 | 版本范围 | 用途 |
|-------|---------|------|
| Python | >=3.11, <3.14 | 运行时环境 |
| numpy | >=2.0, <3.0 | 数值计算 |
| pandas | >=2.0, <4.0 | 数据处理 |
| scikit-learn | >=1.3, <2.0 | 机器学习模型(GBR/RFR/GPR) |
| scipy | >=1.11, <2.0 | 科学计算(优化/统计) |
| xgboost | >=2.0, <4.0 | XGBoost模型 |
| matplotlib | >=3.7, <4.0 | 数据可视化 |
| opencv-python | >=4.8, <5.0 | 图像处理 |
| shap | >=0.42, <1.0 | 可解释性分析 |
| optuna | >=3.0, <5.0 | 超参数优化 |
| python-docx | >=0.8, <2.0 | Word文档读取 |
| openpyxl | >=3.1, <4.0 | Excel文件处理 |
| pillow | >=10.0, <13.0 | 图像读取/处理 |
| torch | >=2.0, <3.0 | 深度学习(MicroSAM) |
| torchvision | >=0.15, <1.0 | 计算机视觉工具 |
| pyyaml | >=6.0, <7.0 | YAML配置解析 |
| tqdm | >=4.65, <5.0 | 进度条 |
| joblib | >=1.3, <2.0 | 模型序列化 |
| lxml | >=4.9, <7.0 | XML解析 |
| chromadb | ^1.5.9 | 向量数据库 |
| sentence-transformers | ^5.6.0 | 文本嵌入模型 |

### 7.2 开发依赖

| 依赖包 | 版本范围 | 用途 |
|-------|---------|------|
| pytest | >=7.0, <9.0 | 单元测试 |
| black | >=23.0, <25.0 | 代码格式化 |
| ruff | >=0.1, <1.0 | 代码检查(Linting) |
| mypy | >=1.5, <2.0 | 类型检查 |

### 7.3 模块间依赖

```
pipeline/
  ├── 依赖: numpy, pandas, scikit-learn, scipy, matplotlib
  └── 内部依赖: utils/plot_style.py, utils/cache_manager.py

image/
  ├── 依赖: opencv-python, numpy, pillow, pandas, tqdm
  └── 内部依赖: image/microsam_integration.py (可选)

analysis/
  ├── 依赖: pandas, numpy, scipy, matplotlib
  └── 无内部依赖

figures/
  ├── 依赖: matplotlib, numpy, pandas
  └── 内部依赖: utils/plot_style.py

knowledge/
  ├── 依赖: chromadb, sentence-transformers, pyyaml, python-docx
  └── 配置依赖: kb_config.yaml

utils/
  ├── 依赖: matplotlib, numpy
  └── 无内部依赖
```

---

## 8. 项目运行方式

### 8.1 环境要求

- **操作系统**: Windows 10/11
- **Python版本**: 3.11 ~ 3.13
- **内存**: 4GB以上
- **磁盘空间**: 2GB以上

### 8.2 安装方式

#### 方式1：Poetry（推荐）
```bash
cd ml-laser-jg1-q355
poetry install
poetry shell
```

#### 方式2：pip
```bash
cd ml-laser-jg1-q355
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

### 8.3 运行命令

#### 运行完整四阶段流水线
```bash
# 方式1：根目录入口
python run.py

# 方式2：直接运行流水线脚本
python pipeline/run_pipeline.py

# 方式3：Windows一键启动
双击 启动.bat
```

#### 运行单个阶段
```bash
python run.py --stage 1   # 阶段1：数据整合
python run.py --stage 2   # 阶段2：响应面建模
python run.py --stage 3   # 阶段3：性能建模
python run.py --stage 4   # 阶段4：优化验证
```

#### 运行图像处理主程序
```bash
python image/picture_processing.py
```

#### 生成论文级图表
```bash
python figures/generate_paper_figures.py
```

#### 查看帮助
```bash
python run.py --help
```

### 8.4 命令行参数

| 参数 | 说明 | 示例 |
|-----|------|------|
| (无) | 执行完整四阶段流程 | `python run.py` |
| `--stage N` | 执行单个阶段(N=1,2,3,4) | `python run.py --stage 2` |
| `--help / -h` | 显示帮助信息 | `python run.py --help` |
| `--sim` | (data_integration) 生成模拟数据 | `python pipeline/data_integration.py --sim` |

---

## 9. 配置说明

### 9.1 pyproject.toml 配置

```toml
[tool.poetry]
name = "ml-laser-jg1-q355"
version = "1.0.0"
description = "机器学习激光功率优化项目"
authors = ["Wang Yuhang <22080740214@hebau.edu.cn>"]
license = "MIT"

[tool.poetry.dependencies]
python = ">=3.11,<3.14"
# ... 依赖包列表

[tool.black]
line-length = 100
target-version = ["py311", "py312", "py313"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.mypy]
python_version = "3.13"
```

### 9.2 kb_config.yaml 知识库配置

```yaml
embed_model: "paraphrase-multilingual-MiniLM-L12-v2"

knowledge_base:
  chroma_path: "knowledge_base/chroma_store"
  chunk_size: 512
  chunk_overlap: 100
  similarity_threshold: -1.0
  retrieve_top_n: 3
  collections:
    - name: "laser_cladding"
    - name: "metallography"
    - name: "material_ml"
    - name: "literature"

chat_memory:
  sqlite_path: "chat_memory/session_sqlite.db"
  max_short_round: 20
```

### 9.3 绘图样式配置

通过 `utils/plot_style.py` 中的 `setup_plot_style()` 函数统一配置：

| 配置项 | 默认值 |
|-------|-------|
| 字体 | SimHei, Microsoft YaHei |
| 坐标轴线宽 | 2.5 pt |
| 刻度方向 | 朝内(in) |
| DPI | 600 |
| 主色调 | #29a0f5, #ff7801, #2ed52e, #d62728 |
| 渐变颜色 | 顶部#67B3E9, 底部#FFFFFF |

---

## 10. 输出文件说明

### 10.1 主要输出文件

| 文件路径 | 格式 | 说明 |
|---------|------|------|
| `analysis_output/金相定量表征数据汇总.csv` | CSV | 图像分割定量结果 |
| `analysis_output/完整实验数据汇总.csv` | CSV | 整合所有实验数据(56×52) |
| `analysis_output/power_response_models.pkl` | Pickle | 阶段2 GPR响应面模型 |
| `analysis_output/property_models.pkl` | Pickle | 阶段3 ML模型(GBR/RFR/GPR) |
| `analysis_output/optimization_results.pkl` | Pickle | 阶段4 优化结果 |
| `analysis_output/数据整合报告.md` | Markdown | 阶段1报告 |
| `analysis_output/阶段2_响应面分析报告.md` | Markdown | 阶段2报告 |
| `analysis_output/阶段3_性能模型报告.md` | Markdown | 阶段3报告 |
| `analysis_output/阶段4_优化汇总报告.md` | Markdown | 阶段4报告 |
| `analysis_output/实验验证方案.md` | Markdown | 实验验证方案 |

### 10.2 图表输出

| 目录 | 内容 | 格式 |
|-----|------|------|
| `analysis_output/figures/paper/` | 15种论文级图表 | PNG (600dpi) |
| `analysis_output/figures/segmentation/` | 图像分割结果 | PNG |
| `analysis_output/figures/` | 流水线生成的各类图表 | PNG |

### 10.3 数据字段说明

#### 核心字段

| 字段名 | 类型 | 说明 | 单位 |
|-------|------|------|------|
| 激光功率 | str | 激光功率(W) | W |
| 熔覆层组织面积占比(%) | float | 熔覆层面积百分比 | % |
| 析出相/碳化物面积占比(%) | float | 析出相面积百分比 | % |
| 气孔孔隙率(%) | float | 气孔面积百分比 | % |
| 微裂纹面积占比(%) | float | 裂纹面积百分比 | % |
| 熔覆层平均晶粒尺寸(μm) | float | 等效圆直径 | μm |
| 基体稀释率(%) | float | 基体稀释程度 | % |
| mh_mean_hv | float | 实测显微硬度均值 | HV |
| wear_friction_steady | float | 稳态摩擦系数 | - |
| eis_Rct_ohm | float | 电荷转移电阻 | Ω |
| xrd_main_peak_2theta | float | XRD主峰位置 | ° |
| power_w | float | 功率数值(衍生) | W |
| heat_input | float | 热输入(衍生) | J/mm² |
| hall_petch | float | Hall-Petch项(衍生) | μm⁻⁰ᐧ⁵ |

---

## 附录

### A. 常用命令速查

```bash
# 安装依赖
poetry install

# 激活环境
poetry shell

# 运行完整流程
python run.py

# 运行单个阶段
python run.py --stage 3

# 生成论文图表
python figures/generate_paper_figures.py

# 运行图像分析
python image/picture_processing.py

# 查看缓存状态
python utils/cache_manager.py

# 代码格式化
black .

# 代码检查
ruff check .
```

### B. 项目资源

- **GitHub仓库**: https://github.com/luh-Byte/ml-laser-jg1-q355.git
- **README**: [README.md](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/README.md)
- **快速使用指南**: [快速使用指南.md](file:///d:/ML-Laser-JG1-Q355/ml-laser-jg1-q355/快速使用指南.md)
- **许可证**: MIT License

---

*文档版本: v1.0 | 生成日期: 2026-06-29*
