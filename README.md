# EY Open Science Data Challenge 2026
## Optimizing Clean Water Supply / 优化清洁水供应

![License](https://img.shields.io/badge/license-BUSL%201.1-red.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Docs](https://img.shields.io/badge/docs-bilingual-orange.svg)
[![Challenge](https://img.shields.io/badge/EY%20Challenge-Official%20Page-blue)](https://challenge.ey.com/challenges/2026-optimizing-clean-water-supply/overview)

### 📖 Project Overview & Tech Stack / 项目概览与技术栈
This project builds a robust, end-to-end Machine Learning pipeline to predict water quality parameters (Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus) in South Africa using satellite imagery and climate data. 

**Core Engineering Stack (核心技术栈)**:
- **Data Engineering**: Microsoft Planetary Computer API (`pystac-client`, `odc-stac`) for pure, temporally-bounded (±30 days) optical band extraction with strict bit-wise Cloud/Shadow masking.
- **Feature Engineering**: Domain-specific synthesis of Remote Sensing Water Indices (`SABI`, `NDWI`, `MNDWI`, `WRI`, `NDVI`) and TerraClimate data integration.
- **Modeling Engine**: Ensembled Gradient Boosting framework utilizing `XGBoost`, `LightGBM`, and `CatBoost` for heterogeneous tabular approximation.
- **MLOps & Validation**: Automated CI/CD (`GitHub Actions`) hooked to a custom `Spatial GroupKFold` local evaluation script, ensuring strict geographical extrapolation testing without coordinate leakage.

**Official Website**: [EY Open Science Data Challenge 2026](https://challenge.ey.com/challenges/2026-optimizing-clean-water-supply/overview)

本项目采用现代 MLOps 与数据工程最佳实践，构建了一条端到端的机器学习流水线。技术栈涵盖了从微软行星计算机 API 的纯净光谱抓取、去云处理，到基于环境物理构造遥感水体指数的特征工程区，并最终交由深度调优的异构集成树模型（XGB/LGBM/CatBoost）进行拟合验证。同时项目内置了严格的防御性自动化空间交叉验证框架。

### 🚀 Getting Started / 快速开始

#### 1. Documentation / 文档
Please refer to the detailed **Bilingual Project Manual** covering the entire optimization journey:
👉 [**EY_Challenge_2026_Report.pdf**](doc/dist/EY_Challenge_2026_Report.pdf) (Located in `doc/dist/`)

#### 2. Local Environment Setup / 本地环境配置
1.  **Clone the repo**:
    ```bash
    git clone https://github.com/EUR-UN/EY_Challenge_2026.git
    cd EY_Challenge_2026
    ```
2.  **Install Python Dependencies**:
    ```bash
    pip install -r resources/code/general/requirements.txt
    pip install xgboost lightgbm catboost optuna pystac-client odc-stac scikit-learn pandas numpy
    ```

#### 3. Core Pipeline / 核心流水线
The project has evolved far beyond the official baseline notebook into a modular Python pipeline:
- **Data Fetching**: `python src/data/fetch_planetary_data.py` (Downloads pristine optical bands and cloud masks from MS Planetary Computer)
- **Model Training**: `python src/models/ensemble_model.py` (Trains the Gradient Boosting ensemble)
- **Local Validation**: `python src/evaluation/evaluate_local.py` (Calculates Spatial K-Fold CV mapped to the LB)

### 📂 Directory Structure / 目录结构
```
EY_Challenge_2026/
├── doc/                 # LaTeX Documentation Source
│   ├── dist/            # Compiled PDF (stable name)
│   └── chapters/        # Bilingual Content
├── resources/
│   ├── code/            # Cleaned Code Packages (Snowflake/General)
│   ├── data/            # Training Data & Templates
│   └── media/           # Images & Tutorials
└── .github/             # Governance templates
```

### 🤝 Contributing / 贡献
We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on our Fork & Pull Request workflow.

### 📜 License / 许可
This project is licensed under the **Business Source License 1.1 (BUSL-1.1)**.

> ⚠️ **Restriction / 限制**:
> You may NOT use this code to submit an entry to the **EY Open Science Data Challenge 2026**.
> 您**不得**使用此代码提交 **2026 EY 开放科学数据挑战赛** 的参赛作品。
>
> 🔓 **Open Source Date / 开源日期**:
> On **2026-05-07** (after the challenge ends), this restriction lifts and the license automatically converts to **MIT**, allowing full free use.
> 在 **2026年5月7日**（挑战结束后），此限制将解除，许可证自动转换为 **MIT**，允许完全免费使用。
