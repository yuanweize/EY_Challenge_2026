# EY Open Science Data Challenge 2026
## Optimizing Clean Water Supply / 优化清洁水供应

![License](https://img.shields.io/badge/license-BUSL%201.1-red.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)
![Docs](https://img.shields.io/badge/docs-bilingual-orange.svg)

> **Repository Structure**:
> - **Upstream (Official)**: `EUR-UN/EY_Challenge_2026`
> - **Lead Developer**: `yuanweize/EY_Challenge_2026`

### 📖 Project Overview / 项目概览
This project aims to develop machine learning models to predict water quality parameters (Total Alkalinity, Electrical Conductance, Dissolved Reactive Phosphorus) in South Africa using satellite imagery and climate data.

本项目旨在利用卫星图像和气候数据，开发机器学习模型以预测南非地区的水质参数（总碱度、电导率、溶解性反应磷）。

### 🚀 Getting Started / 快速开始

#### 1. Documentation / 文档
Please refer to the detailed **Bilingual Project Manual**:
👉 [**Project_Documentation.pdf**](doc/dist/) (Located in `doc/dist/`)

This manual includes:
-   Challenge Rules & Objectives
-   Step-by-step Setup Guides (Snowflake & Local)
-   Resource Inventory
-   FAQ

#### 2. Development / 开发
1.  **Clone the repo**:
    ```bash
    git clone https://github.com/yuanweize/EY_Challenge_2026.git
    cd EY_Challenge_2026
    ```
2.  **Environment Setup**:
    -   **Snowflake Users**: Run `resources/code/snowflake/snowflake_setup.sql`.
    -   **Local Users**: `pip install -r resources/code/general/requirements.txt`.
3.  **Run Benchmarks**:
    -   Open `resources/code/general/Benchmark_Model_Notebook.ipynb`.

### 📂 Directory Structure / 目录结构
```
EY_Challenge_2026/
├── doc/                 # LaTeX Documentation Source
│   ├── dist/            # Compiled PDFs (vX.X)
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
