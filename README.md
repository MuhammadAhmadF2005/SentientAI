# 🧠 SentientAI: End-to-End AI, Machine Learning & Deep Learning Roadmap

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**SentientAI** is a comprehensive, production-oriented repository designed to bridge fundamental computer science concepts with modern Artificial Intelligence, Classical Machine Learning, Statistical Data Science, and cutting-edge Deep Learning architectures.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Curriculum & Roadmap Breakdown](#-curriculum--roadmap-breakdown)
  - [1. Python Basics & Software Engineering](#1-python-basics--software-engineering)
  - [2. Scientific Computing & Libraries](#2-scientific-computing--libraries)
  - [3. Data Science & Statistical Analysis](#3-data-science--statistical-analysis)
  - [4. Classical Machine Learning & Ensembles](#4-classical-machine-learning--ensembles)
  - [5. Deep Learning & Modern AI Architectures](#5-deep-learning--modern-ai-architectures)
- [Datasets & Data Management](#-datasets--data-management)
- [Getting Started & Installation](#-getting-started--installation)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Installing Dependencies](#2-installing-dependencies)
  - [3. Running Notebooks](#3-running-notebooks)
  - [4. Running the Sales Pipeline Microservice](#4-running-the-sales-pipeline-microservice)
- [Roadmap Tracker](#-roadmap-tracker)
- [Contributing & License](#-contributing--license)

---

## 🌟 Overview

This repository serves as both a progressive learning path and a portfolio of hands-on implementations:
- **Production-grade Python**: Clean code, OOP, custom exception systems, and modular REST microservices with FastAPI.
- **Statistical Data Science**: In-depth exploratory data analysis (EDA), automated data profiling, feature engineering, RFM customer segmentation, and market basket analysis.
- **Predictive Machine Learning**: From baseline estimators (KNN, Decision Trees) to advanced ensembles (AdaBoost, Gradient Boosting, XGBoost, LightGBM) and unsupervised clustering (K-Means, DBSCAN).
- **Deep Learning & Frontiers**: From multi-layer perceptrons and CNNs to sequence modeling (RNNs, LSTMs), Physics-Informed Neural Networks (PINNs), and Transformer self-attention mechanisms.

---

## 🗂 Repository Structure

```text
sentientAI/
├── Datasets/                                      # Datasets & benchmark resources
│   ├── diabetes.csv                               # Pima Indians Diabetes classification data
│   └── online_retail.csv                          # Transactional data (excluded via .gitignore)
├── Roadmap/
│   ├── 1_Python_Basics/                           # Core language features & modular pipelines
│   │   ├── Error_Handling/
│   │   │   └── exception.py                       # Custom exception hierarchies & error handling
│   │   └── File_Handling/sales_pipeline/          # Multi-file sales ingestion FastAPI service
│   │       ├── analyzer.py                        # Statistical analysis module
│   │       ├── cleaner.py                         # Data cleaning & validation module
│   │       ├── main.py                            # FastAPI app & endpoint routing
│   │       ├── requirements.txt                   # Pipeline-specific dependencies
│   │       └── summary.ipynb                      # Pipeline test & execution walkthrough
│   ├── 2_Python_Libraries/
│   │   └── train_numpy_pandas.ipynb               # Vectorization, arrays, series & dataframes
│   ├── 3_Data_Science/
│   │   ├── EDA.ipynb                              # Exploratory data analysis workflows
│   │   ├── PCA.ipynb                              # Principal Component Analysis & dimensionality reduction
│   │   ├── SVM.ipynb                              # Support Vector Machines & decision boundaries
│   │   ├── level1_customer_analysis.ipynb         # RFM segmentation, customer cohorts & basket analysis
│   │   └── profile_report.html                    # YData automated data profiling report
│   ├── 4_Machine_Learning/
│   │   ├── AdaBoost.ipynb                         # Adaptive Boosting implementation
│   │   ├── GradientBoosting.ipynb                 # Gradient Boosted Trees
│   │   ├── KNN.ipynb                              # K-Nearest Neighbors classifier & regressor
│   │   ├── KmeansClustering.ipynb                 # K-Means clustering, elbow method & silhouette analysis
│   │   ├── XGBoost_LightGBM.ipynb                 # Gradient boosting benchmarks (XGBoost vs. LightGBM)
│   │   ├── churn_prediction_guide.ipynb           # Step-by-step customer churn modeling guide
│   │   ├── churn_prediction_advanced.ipynb        # Advanced end-to-end churn prediction pipeline
│   │   └── dbscan.ipynb                           # Density-Based Spatial Clustering of Applications with Noise
│   ├── 5_Deep_Learning/
│   │   ├── ANN.ipynb                              # Artificial Neural Networks & Backpropagation
│   │   ├── Convolution.ipynb                      # Convolutional Neural Networks (CNNs) & feature maps
│   │   ├── RNN.ipynb                              # Recurrent Neural Networks for sequences
│   │   ├── LSTM.ipynb                             # Long Short-Term Memory networks for temporal data
│   │   ├── PINN.ipynb                             # Physics-Informed Neural Networks (PDEs + NNs)
│   │   ├── transformer.ipynb                      # Transformer architecture & Self-Attention mechanism
│   │   └── train.ipynb                            # Training loops, loss tracking & evaluation
│   └── Roadmap - Cleaned Syllabus.csv             # Structured syllabus & duration roadmap
├── requirements.txt                               # Root environment dependencies
└── README.md                                      # Project documentation
```

---

## 🚀 Curriculum & Roadmap Breakdown

### 1. Python Basics & Software Engineering
* **Custom Error Handling** (`Roadmap/1_Python_Basics/Error_Handling/exception.py`):
  - Structured `try-except-else-finally` exception flow.
  - Domain-specific custom exceptions (e.g., `InsufficientFundsError`) utilizing Object-Oriented Programming (OOP) patterns.
* **Modular Sales Data Pipeline Microservice** (`Roadmap/1_Python_Basics/File_Handling/sales_pipeline/`):
  - Production-ready **FastAPI** service for regional sales file ingestion.
  - Decoupled `DataCleaner`, `DataAnalyzer`, and `PipelineManager` classes.
  - REST endpoints for upload (`/upload`), batch processing & cleaning (`/process`), summary metrics (`/summary`), and combined file download (`/download`).

### 2. Scientific Computing & Libraries
* **NumPy & Pandas Foundations** (`Roadmap/2_Python_Libraries/train_numpy_pandas.ipynb`):
  - Multi-dimensional array operations, matrix mathematics, and broadcasting.
  - Pandas DataFrames, indexing, masking, group-by aggregations, pivoting, and memory-efficient data wrangling.

### 3. Data Science & Statistical Analysis
* **Exploratory Data Analysis (EDA)** (`Roadmap/3_Data_Science/EDA.ipynb`):
  - Distribution diagnostics, missing data imputation, outlier detection, and correlation analysis.
* **Automated Data Profiling** (`Roadmap/3_Data_Science/profile_report.html`):
  - Interactive HTML report generated using `ydata-profiling` for rapid dataset health assessment.
* **Dimensionality Reduction with PCA** (`Roadmap/3_Data_Science/PCA.ipynb`):
  - Eigenvalue/eigenvector decomposition, scree plots, cumulative explained variance ratio, and high-dimensional projection.
* **Support Vector Machines (SVM)** (`Roadmap/3_Data_Science/SVM.ipynb`):
  - Linear and non-linear classification kernels (RBF, Polynomial), maximal margin classifiers, and support vector interpretation.
* **Customer RFM Segmentation & Cohort Analysis** (`Roadmap/3_Data_Science/level1_customer_analysis.ipynb`):
  - Recency, Frequency, Monetary (RFM) modeling on retail transactions.
  - Customer lifetime value segmentation (High-Value, Medium-Value, Inactive).
  - Co-occurrence item pairing and market basket insights.

### 4. Classical Machine Learning & Ensembles
* **Supervised Classifiers & Distance Metrics**:
  - `KNN.ipynb`: K-Nearest Neighbors modeling with Euclidean and Manhattan distances.
* **Unsupervised Clustering**:
  - `KmeansClustering.ipynb`: Centroid-based clustering, inertia optimization, elbow curve, and silhouette score validation.
  - `dbscan.ipynb`: Density reachability, core samples, noise detection, and non-spherical cluster discovery.
* **Ensemble Methods & Gradient Boosting**:
  - `AdaBoost.ipynb`: Sequential weak learner boosting with adaptive sample weight updates.
  - `GradientBoosting.ipynb`: Stage-wise additive modeling minimizing arbitrary differentiable loss functions.
  - `XGBoost_LightGBM.ipynb`: High-performance tree boosting comparison, histogram-based splitting, and GPU acceleration.
* **End-to-End Churn Prediction Systems**:
  - `churn_prediction_guide.ipynb`: Fundamental walkthrough of customer churn feature selection and classification.
  - `churn_prediction_advanced.ipynb`: Complete machine learning pipeline including class imbalance handling, hyperparameter tuning, cross-validation, and ROC-AUC evaluation.

### 5. Deep Learning & Modern AI Architectures
* **Artificial Neural Networks (ANN)** (`Roadmap/5_Deep_Learning/ANN.ipynb`):
  - Multi-layer perceptrons, forward propagation, backpropagation with gradient descent, activation functions (ReLU, Sigmoid, Softmax), and loss optimization.
* **Convolutional Neural Networks (CNN)** (`Roadmap/5_Deep_Learning/Convolution.ipynb`):
  - 2D convolutions, spatial hierarchy extraction, filter kernels, pooling layers, and feature map visualization.
* **Sequence Modeling: RNNs & LSTMs**:
  - `RNN.ipynb`: Recurrent units, hidden state propagation, and vanishing gradient intuition.
  - `LSTM.ipynb`: Gated memory cells (input, forget, output gates) for long-range temporal dependencies.
* **Physics-Informed Neural Networks (PINN)** (`Roadmap/5_Deep_Learning/PINN.ipynb`):
  - Integration of differential equations and physics loss constraints into neural network training.
* **Transformer Architectures & Attention** (`Roadmap/5_Deep_Learning/transformer.ipynb`):
  - Scaled dot-product attention, multi-head self-attention mechanisms, and positional encoding.
* **Model Training & Evaluation** (`Roadmap/5_Deep_Learning/train.ipynb`):
  - Training loops, regularization, early stopping, and validation tracking.

---

## 📊 Datasets & Data Management

Datasets used across the project are organized under the `Datasets/` directory:

| Dataset | File Name | Description / Source |
| :--- | :--- | :--- |
| **Online Retail** | `Datasets/online_retail.csv` | Transnational dataset containing transactions for a UK-based online retail store. Download via [UCI ML Repository](https://archive.ics.uci.edu/dataset/352/online+retail). |
| **Pima Indians Diabetes** | `Datasets/diabetes.csv` | Medical diagnostic measurements for binary diabetes classification. |

> **Note**: Due to file size constraints, large datasets such as `online_retail.csv` are omitted from Git tracking via `.gitignore`. Download the raw dataset and place it in `Datasets/` (or the project root) prior to running dependent notebooks.

---

## 🛠 Getting Started & Installation

### 1. Environment Setup

Clone the repository and create a Python virtual environment:

```bash
# Clone the repository
git clone https://github.com/MuhammadAhmadF2005/SentientAI.git
cd SentientAI

# Create virtual environment
python -m venv .venv
```

Activate the virtual environment:
* **Windows (PowerShell):**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Windows (CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Installing Dependencies

Install the core dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Register the IPython kernel for Jupyter:
```bash
python -m ipykernel install --user --name sentient_ds --display-name "Python (sentient_ds)"
```

### 3. Running Notebooks

Launch the Jupyter Notebook server:
```bash
jupyter notebook
```
Navigate to any notebook in the `Roadmap/` folders and ensure your kernel is set to **Python (sentient_ds)**.

### 4. Running the Sales Pipeline Microservice

To run the standalone FastAPI regional sales pipeline:
```bash
cd Roadmap/1_Python_Basics/File_Handling/sales_pipeline
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open your browser at `http://127.0.0.1:8000/docs` to interact with the Swagger API interface.

---

## 📈 Roadmap Tracker

| Phase | Core Domain | Key Highlights | Status |
| :---: | :--- | :--- | :---: |
| **1** | **Python Engineering** | Custom exceptions, OOP, file handling, FastAPI sales service | ✅ Completed |
| **2** | **Scientific Libraries** | NumPy array computations, Pandas manipulation & indexing | ✅ Completed |
| **3** | **Data Science** | EDA, PCA, SVM, RFM customer analytics, YData profiling | ✅ Completed |
| **4** | **Machine Learning** | KNN, K-Means, DBSCAN, AdaBoost, XGBoost, LightGBM, Churn modeling | ✅ Completed |
| **5** | **Deep Learning** | ANNs, CNNs, RNNs, LSTMs, PINNs, Transformers & Attention | ✅ Completed |
| **6** | **GenAI & LLM Tuning** | LoRA, QLoRA, Fine-tuning, RAG pipelines | 🔄 In Progress |

---

## 🤝 Contributing & License

Contributions, suggestions, and feedback are always welcome! Feel free to open an issue or submit a Pull Request.

This project is licensed under the [MIT License](LICENSE).
