# Online Retail Sales and Customer Analysis

This repository contains a basic data analysis of the Online Retail transaction dataset. The analysis is presented in a Jupyter Notebook: `level1.ipynb`.

---

## Setup and Run Instructions

This project runs inside a custom Python virtual environment (`.venv`) containing pandas, numpy, matplotlib, and seaborn.

### 1. Activating the Virtual Environment
Activate the environment from your terminal inside the project directory:

* **PowerShell:**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
* **Command Prompt:**
  ```cmd
  .venv\Scripts\activate.bat
  ```
* **macOS / Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Dependencies
The required packages are defined in `requirements.txt`:
* `pandas` & `openpyxl` - for data loading and cleaning
* `numpy` - for numeric operations
* `matplotlib` & `seaborn` - for static line and bar charts
* `ipykernel` & `notebook` - for Jupyter environment run support

### 3. Running the Jupyter Notebook
To start the notebook interface:
```powershell
.venv\Scripts\jupyter notebook
```
Then select and run `level1.ipynb`. Make sure you use the registered `sentient_ds` kernel (displayed as "Python (sentient_ds)" in the notebook interface) which maps to this virtual environment.

---

## Analysis Outline

The notebook is divided into five sections:

1. **Data Cleaning**: Removes transactions with missing Customer IDs, filters out returns (negative quantities) and non-positive prices, drops duplicate rows, and creates a total revenue column.
2. **Monthly Sales and Top Products**: Calculates monthly revenue trends and plots the top 10 products by quantity sold using standard matplotlib plots.
3. **Rule-Based Customer Grouping**: Computes Recency, Frequency, and Monetary values, and segments customers into three tiers:
   * **High Value**: Total spend > $2,000 and total unique orders >= 5.
   * **Low Value (Inactive)**: Last purchase was more than 180 days ago.
   * **Medium Value**: All other active customers.
4. **Unique Active Customers by Month**: Charts monthly customer counts to observe customer participation trends.
5. **Basic Product Associations**: Counts overlapping pairs of items within invoices and lists the top 10 most common co-occurrences.
