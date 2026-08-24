# SentientAI: End-to-End Artificial Intelligence, Machine Learning, and Deep Learning Framework

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**SentientAI** is a structured, production-grade repository encompassing the complete continuum of modern machine intelligence: from foundational software engineering and object-oriented data pipelines, to statistical data science, classical machine learning algorithms, and deep neural network architectures.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Architecture](#repository-architecture)
- [Curriculum and Module Specifications](#curriculum-and-module-specifications)
  - [Module 1: Python Engineering and Modular Pipelines](#module-1-python-engineering-and-modular-pipelines)
  - [Module 2: Scientific Computing and Numerical Operations](#module-2-scientific-computing-and-numerical-operations)
  - [Module 3: Statistical Data Science and Exploratory Analysis](#module-3-statistical-data-science-and-exploratory-analysis)
  - [Module 4: Supervised, Unsupervised, and Ensemble Learning](#module-4-supervised-unsupervised-and-ensemble-learning)
  - [Module 5: Deep Learning and Neural Architectures](#module-5-deep-learning-and-neural-architectures)
- [Dataset Specifications and Management](#dataset-specifications-and-management)
- [Installation and Execution Guide](#installation-and-execution-guide)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Dependency Installation](#2-dependency-installation)
  - [3. Interactive Notebook Execution](#3-interactive-notebook-execution)
  - [4. Pipeline Microservice Deployment](#4-pipeline-microservice-deployment)
- [Roadmap and Milestone Progress](#roadmap-and-milestone-progress)
- [License and Contribution Policy](#license-and-contribution-policy)

---

## Project Overview

The objective of this repository is to establish a rigorous, end-to-end framework demonstrating theoretical knowledge, algorithmic implementations, and production-level code standards across multiple subdisciplines of artificial intelligence:

* **Software Engineering Practices**: Object-oriented programming (OOP), custom exception hierarchies, decoupled architecture, and REST API microservice development with FastAPI.
* **Exploratory and Statistical Analysis**: Data validation, automated profiling, feature extraction, Recency-Frequency-Monetary (RFM) customer segmentation, and market basket co-occurrence modeling.
* **Machine Learning Pipelines**: Supervised classification and regression, unsupervised clustering, tree-based ensemble methods (AdaBoost, Gradient Boosting, XGBoost, LightGBM), and end-to-end customer churn prediction architectures.
* **Deep Neural Networks**: Multi-layer perceptrons, convolutional feature extraction, sequential modeling with Recurrent Neural Networks and LSTMs, Physics-Informed Neural Networks (PINNs), and Transformer self-attention mechanisms.

---

## Repository Architecture

```text
sentientAI/
├── Datasets/                                      # Datasets and benchmark resources
│   ├── diabetes.csv                               # Diagnostic dataset for binary classification
│   └── online_retail.csv                          # Transactional data (excluded via .gitignore)
├── Roadmap/
│   ├── 1_Python_Basics/                           # Software engineering and modular systems
│   │   ├── Error_Handling/
│   │   │   └── exception.py                       # Custom exception hierarchies and error handling
│   │   └── File_Handling/sales_pipeline/          # Multi-file sales ingestion FastAPI service
│   │       ├── analyzer.py                        # Statistical aggregation module
│   │       ├── cleaner.py                         # Data sanitization and validation module
│   │       ├── main.py                            # FastAPI application and routing
│   │       ├── requirements.txt                   # Service-specific dependencies
│   │       └── summary.ipynb                      # Pipeline test and execution walkthrough
│   ├── 2_Python_Libraries/
│   │   └── train_numpy_pandas.ipynb               # Vectorization, array ops, and DataFrame manipulation
│   ├── 3_Data_Science/
│   │   ├── EDA.ipynb                              # Exploratory data analysis workflows
│   │   ├── PCA.ipynb                              # Principal Component Analysis and dimensionality reduction
│   │   ├── SVM.ipynb                              # Support Vector Machines and decision boundaries
│   │   ├── level1_customer_analysis.ipynb         # RFM segmentation, customer cohorts, and basket analysis
│   │   └── profile_report.html                    # Automated YData Profiling exploratory report
│   ├── 4_Machine_Learning/
│   │   ├── AdaBoost.ipynb                         # Adaptive Boosting implementation
│   │   ├── GradientBoosting.ipynb                 # Gradient Boosted Decision Trees
│   │   ├── KNN.ipynb                              # K-Nearest Neighbors classification and regression
│   │   ├── KmeansClustering.ipynb                 # K-Means clustering, inertia, and silhouette analysis
│   │   ├── XGBoost_LightGBM.ipynb                 # Gradient boosting benchmarks (XGBoost vs. LightGBM)
│   │   ├── churn_prediction_guide.ipynb           # Step-by-step customer churn modeling guide
│   │   ├── churn_prediction_advanced.ipynb        # Production-grade churn prediction pipeline
│   │   └── dbscan.ipynb                           # Density-Based Spatial Clustering of Applications with Noise
│   ├── 5_Deep_Learning/
│   │   ├── ANN.ipynb                              # Artificial Neural Networks and Backpropagation
│   │   ├── Convolution.ipynb                      # Convolutional Neural Networks (CNNs) and feature maps
│   │   ├── RNN.ipynb                              # Recurrent Neural Networks for sequence modeling
│   │   ├── LSTM.ipynb                             # Long Short-Term Memory networks for temporal data
│   │   ├── PINN.ipynb                             # Physics-Informed Neural Networks (PDEs + NNs)
│   │   ├── transformer.ipynb                      # Transformer architecture and Scaled Dot-Product Attention
│   │   └── train.ipynb                            # Training loops, loss metrics, and convergence tracking
│   └── Roadmap - Cleaned Syllabus.csv             # Curriculum syllabus and duration specification
├── requirements.txt                               # Root environment dependency manifest
└── README.md                                      # Project documentation
```

---

## Curriculum and Module Specifications

### Module 1: Python Engineering and Modular Pipelines
* **Exception Handling Architecture** (`Roadmap/1_Python_Basics/Error_Handling/exception.py`):
  - Robust `try-except-else-finally` exception flow.
  - Implementation of domain-specific exception classes (e.g., `InsufficientFundsError`) using Object-Oriented Programming (OOP) paradigms.
* **Sales Data Pipeline Microservice** (`Roadmap/1_Python_Basics/File_Handling/sales_pipeline/`):
  - Decoupled, production-ready **FastAPI** service for multi-file regional sales ingestion.
  - Segregated `DataCleaner`, `DataAnalyzer`, and `PipelineManager` domain classes.
  - Endpoints supporting file upload validation (`/upload`), batch cleaning and concatenation (`/process`), summary aggregation (`/summary`), and clean CSV export (`/download`).

### Module 2: Scientific Computing and Numerical Operations
* **NumPy and Pandas Computation** (`Roadmap/2_Python_Libraries/train_numpy_pandas.ipynb`):
  - Multidimensional tensor manipulations, broadcasting, and vectorized operations.
  - High-performance Pandas indexing, masking, group aggregations, pivot operations, and memory optimization.

### Module 3: Statistical Data Science and Exploratory Analysis
* **Exploratory Data Analysis (EDA)** (`Roadmap/3_Data_Science/EDA.ipynb`):
  - Statistical distributions, missing value imputation, outlier detection, and correlation analysis.
* **Automated Data Profiling** (`Roadmap/3_Data_Science/profile_report.html`):
  - Comprehensive HTML profiling report generated using `ydata-profiling` for dataset diagnostics.
* **Principal Component Analysis (PCA)** (`Roadmap/3_Data_Science/PCA.ipynb`):
  - Covariance matrix computation, eigenvalue/eigenvector decomposition, scree plots, cumulative explained variance ratio, and orthogonal projection.
* **Support Vector Machines (SVM)** (`Roadmap/3_Data_Science/SVM.ipynb`):
  - Maximal margin classifiers, linear vs. non-linear kernels (Radial Basis Function, Polynomial), and support vector identification.
* **Customer RFM Segmentation & Cohort Analysis** (`Roadmap/3_Data_Science/level1_customer_analysis.ipynb`):
  - Recency, Frequency, and Monetary (RFM) modeling applied to transactional data.
  - Rule-based customer lifetime segmentation (High-Value, Medium-Value, Inactive).
  - Item pair co-occurrence analysis for market basket evaluation.

### Module 4: Supervised, Unsupervised, and Ensemble Learning
* **Instance-Based Classification**:
  - `KNN.ipynb`: K-Nearest Neighbors classifier and regressor parameterized by distance metrics (Euclidean, Manhattan).
* **Unsupervised Clustering Algorithms**:
  - `KmeansClustering.ipynb`: Centroid optimization, inertia convergence, elbow method, and silhouette evaluation.
  - `dbscan.ipynb`: Density reachability, epsilon neighborhood parameters, core sample selection, and non-convex cluster recovery.
* **Ensemble Methods and Gradient Boosting**:
  - `AdaBoost.ipynb`: Sequential boosting via iterative sample reweighting and weak learner combination.
  - `GradientBoosting.ipynb`: Additive forward stage-wise gradient boosting over continuous loss surfaces.
  - `XGBoost_LightGBM.ipynb`: Performance and efficiency benchmarking between XGBoost and LightGBM histogram-based split finders.
* **End-to-End Customer Churn Pipelines**:
  - `churn_prediction_guide.ipynb`: Fundamental guide covering exploratory feature analysis and baseline modeling.
  - `churn_prediction_advanced.ipynb`: Full machine learning lifecycle including preprocessing pipelines, handling class imbalance, hyperparameter optimization, and ROC-AUC evaluation.

### Module 5: Deep Learning and Neural Architectures
* **Artificial Neural Networks (ANN)** (`Roadmap/5_Deep_Learning/ANN.ipynb`):
  - Multi-layer perceptron forward propagation, backpropagation with gradient descent, activation functions (ReLU, Sigmoid, Softmax), and loss optimization.
* **Convolutional Neural Networks (CNN)** (`Roadmap/5_Deep_Learning/Convolution.ipynb`):
  - 2D convolutional kernels, spatial feature hierarchy extraction, pooling layers, and intermediate feature map visualization.
* **Sequential Architectures (RNN & LSTM)**:
  - `RNN.ipynb`: Recurrent units, hidden state dynamics, and theoretical analysis of gradient degradation.
  - `LSTM.ipynb`: Gated memory units (input, forget, output gates) for modeling long-range temporal dependencies.
* **Physics-Informed Neural Networks (PINN)** (`Roadmap/5_Deep_Learning/PINN.ipynb`):
  - Embedding partial differential equations (PDEs) directly into neural network loss formulations for scientific machine learning.
* **Transformer Architectures and Self-Attention** (`Roadmap/5_Deep_Learning/transformer.ipynb`):
  - Mathematical formulation of Scaled Dot-Product Attention, Multi-Head Attention blocks, and positional encoding.
* **Model Training Framework** (`Roadmap/5_Deep_Learning/train.ipynb`):
  - Training pipeline design, regularization strategies, early stopping mechanisms, and validation telemetry.

---

## Dataset Specifications and Management

Datasets utilized throughout the codebase are structured under the `Datasets/` directory:

| Dataset | Relative Path | Description / Source |
| :--- | :--- | :--- |
| **Online Retail** | `Datasets/online_retail.csv` | Transnational transactional dataset from a UK-based online retailer. Available via the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail). |
| **Pima Indians Diabetes** | `Datasets/diabetes.csv` | Diagnostic record dataset for binary diabetes classification. |

> **Note**: Due to storage limits and repository constraints, large data files such as `online_retail.csv` are omitted from version control via `.gitignore`. Download the raw dataset and place it in `Datasets/` (or the project root) prior to executing dependent notebooks.

---

## Installation and Execution Guide

### 1. Environment Setup

Clone the repository and initialize an isolated Python virtual environment:

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
* **Windows (Command Prompt):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Dependency Installation

Install the required packages using `pip`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Register the dedicated IPython kernel:

```bash
python -m ipykernel install --user --name sentient_ds --display-name "Python (sentient_ds)"
```

### 3. Interactive Notebook Execution

Launch the Jupyter Notebook environment:

```bash
jupyter notebook
```

Open the relevant notebook from the `Roadmap/` directory and ensure the active kernel is set to **Python (sentient_ds)**.

### 4. Pipeline Microservice Deployment

To execute the modular FastAPI sales pipeline:

```bash
cd Roadmap/1_Python_Basics/File_Handling/sales_pipeline
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Navigate to `http://127.0.0.1:8000/docs` to interact with the OpenAPI/Swagger specification.

---

## Roadmap and Milestone Progress

| Phase | Core Domain | Key Implementations | Status |
| :---: | :--- | :--- | :---: |
| **1** | **Python Engineering** | Custom exception hierarchies, OOP architecture, FastAPI sales service | Completed |
| **2** | **Scientific Computing** | Vectorized array operations, multi-index DataFrame manipulation | Completed |
| **3** | **Statistical Data Science** | EDA, PCA dimensionality reduction, SVM classifiers, RFM analytics, YData profiling | Completed |
| **4** | **Machine Learning** | KNN, K-Means, DBSCAN, AdaBoost, XGBoost, LightGBM, Churn modeling pipelines | Completed |
| **5** | **Deep Learning** | ANNs, CNNs, RNNs, LSTMs, Physics-Informed Neural Networks, Transformers | Completed |
| **6** | **Generative AI & LLMs** | Fine-tuning, Parameter-Efficient Fine-Tuning (LoRA / QLoRA), Retrieval-Augmented Generation | Completed |

---

## License and Contribution Policy

Contributions, technical discussions, and issue submissions are welcome. Please ensure that all contributions adhere to standard code formatting, test coverage, and documentation guidelines.

This project is licensed under the [MIT License](LICENSE).
