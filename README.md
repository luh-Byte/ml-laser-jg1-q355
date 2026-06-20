# ML-Laser-JG1-Q355

Machine Learning-based Laser Power Optimization and Microstructure-Property Coordinated Control for JG-1 Iron-based Alloy Q355 Steel

## Project Overview

This project applies machine learning methods to optimize laser cladding process parameters for JG-1 iron-based self-fluxing alloy on Q355 low-carbon steel substrate. It integrates quantitative metallography, electrochemical impedance spectroscopy (EIS), X-ray diffraction (XRD), and wear testing data to build predictive models and perform multi-objective optimization.

## Key Features

- **Data Integration**: Unified CSV with 56 samples × 52 features (metallography + EIS + XRD + wear)
- **Physics Model Calibration**: Hall-Petch + Orowan formulas calibrated against real measured hardness
- **ML Models**: GBR, RFR, GPR with Leave-One-Out cross-validation
- **SHAP Explainability**: Feature importance analysis revealing XRD peak position and friction coefficient as key predictors
- **Multi-objective Optimization**: Pareto front for hardness vs. defect rate trade-off
- **Sensitivity Analysis**: Power sensitivity with 95% confidence intervals

## Project Structure

```
ml-laser-jg1-q355/
├── pyproject.toml              # Poetry project config
├── poetry.lock                 # Locked dependency versions
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── data/                       # Experimental data
│   ├── 900W/                   # Micrograph images (TIF + XML)
│   ├── 1200W/
│   ├── 1500W/
│   ├── 1800W/
│   ├── electrochemical-impedance/  # EIS data
│   ├── wear-data/              # Friction/wear data
│   ├── microhardness-data/     # Microhardness measurements (DOCX)
│   └── xrd-data/               # XRD diffraction data
│
├── scripts/
│   └── merge_experimental_data.py
│
├── data_integration.py         # Phase 1: Integrate all experimental data
├── power_response_model.py     # Phase 2: Power→Microstructure GPR response surface
├── property_model.py           # Phase 3: Microstructure→Property ML models
├── optimized_pipeline.py       # Phase 4: Chain optimization + verification plan
│
├── picture processing.py       # Original main program (image segmentation + ML)
├── streamlit_app.py            # Streamlit web interface
├── material_science_analysis.py # Material science validation
├── thermodynamics_analysis.py  # Thermodynamic chain analysis
├── fix_issues.py               # Physics model bug fixes
├── cache_manager.py            # Cache management utilities
│
├── start.bat                   # Quick start script
└── analysis_output/            # Analysis results (git-ignored)
    ├── 完整实验数据汇总.csv
    ├── power_response_models.pkl
    ├── property_models.pkl
    ├── optimization_results.pkl
    └── *.png                   # Visualization plots
```

## Installation

### Prerequisites
- Python 3.11–3.13
- [Poetry](https://python-poetry.org/) (recommended) or pip

### Setup with Poetry (Recommended)
```bash
cd ml-laser-jg1-q355
poetry install
poetry shell
```

### Setup with pip
```bash
cd ml-laser-jg1-q355
pip install -e .
```

## Usage

### Run the Complete Pipeline
```bash
# Phase 1: Data integration
python data_integration.py

# Phase 2: Power→Microstructure response surface
python power_response_model.py

# Phase 3: Property prediction models
python property_model.py

# Phase 4: Optimization + verification plan
python optimized_pipeline.py
```

### Run Original Streamlit App
```bash
streamlit run streamlit_app.py
```

## Key Results

| Metric | Value |
|--------|-------|
| Hardness range (measured) | 224–385 HV |
| GBR LOO-CV R² | 1.000 |
| GBR CV-RMSE | <1 HV |
| Optimal power (max hardness) | ~1670 W |
| Optimal power (min defects) | 900 W |

### SHAP Feature Importance (Top 5)
1. `power_w` — Laser power
2. `wear_friction_steady` — Steady-state friction coefficient
3. `xrd_main_peak_2theta` — XRD main peak position (phase transformation indicator)
4. `xrd_peak_44_area` — XRD peak area
5. `heat_input` — Heat input (derived from power)

## Citation

If you use this code in your research, please cite:
```
Wang, Y. (2026). ML-Laser-JG1-Q355: Machine Learning-based Laser Power Optimization
for JG-1 Iron-based Alloy Q355 Steel. GitHub Repository.
```

## License

MIT License
