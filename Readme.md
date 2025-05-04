# Earnings Call Stock Ranking System (ECSRS)

## 1. Project Overview

This repository implements the Earnings Call Stock Ranking System (ECSRS), a multi-factor machine learning model that predicts post-earnings stock performance. It integrates sentiment analysis of earnings calls with earnings surprises and firm size to capture the Post-Earnings Announcement Drift (PEAD) phenomenon. The system uses a semi-supervised clustering strategy for label generation and trains a Random Forest classifier to categorize earnings events as Bullish, Neutral, or Bearish. A Streamlit-based web application allows users to interact with the system through an intuitive interface.

## 2. System Architecture

The ECSRS pipeline consists of the following components:

1. **Data Collection & Preprocessing**
2. **Sentiment Analysis (FinBERT)**
3. **Cumulative Abnormal Return (CAR) Computation**
4. **Unsupervised Clustering & Label Assignment**
5. **Supervised Classification using Random Forest**
6. **Portfolio Backtesting**
7. **Real-time Prediction Interface (Streamlit App)**

## 3. Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gary30hii/EarningsCallStockRankingSystem--ECSRS.git
   cd EarningsCallStockRankingSystem--ECSRS
   ```

2. **Install the required Python packages:**

   You can install them manually, or with:

   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include:
   - `pandas`, `numpy`, `scikit-learn`, `shap`, `streamlit`, `transformers`, `requests`, `torch`, `nltk`, `matplotlib`, `seaborn`

3. **Install and configure Git LFS (Large File Storage):**

   Git LFS is used to manage large dataset files in this project.

   ```bash
   git lfs install
   git lfs pull
   ```

   > ⚠️ *Make sure Git LFS is installed before pulling large dataset files from the repository.*

4. **LLaMA Sentiment Analysis (Optional but recommended for comparison):**

   - Install **[Ollama](https://ollama.com)** from their official website.
   - After installation, download the LLaMA 3.2 3B model locally by running:

     ```bash
     ollama run llama3:3b
     ```

   - In `Stock_list.ipynb`, update the `MODEL_NAME` variable to match your downloaded model name (e.g., `llama3.2:latest`) to ensure the LLaMA sentiment analysis runs correctly.

   > ⚠️ *LLaMA is one of the sentiment analysis models compared in this project. While the pipeline fully supports FinBERT, using LLaMA allows reproduction of the multi-model sentiment comparison analysis.*

5. **Insert your [Financial Modeling Prep API](https://site.financialmodelingprep.com/) key** into `Stock_list.ipynb` to enable stock data collection.

## 4. Step-by-Step Reproduction Guide

### Step 1: Data Collection, Sentiment Analysis, and Feature Engineering

Run `Stock_list.ipynb`.

- Collects data on ~500 US stocks from FMP API (2019–2024)
- Computes:
  - Earnings surprise
  - Cumulative Abnormal Returns (CAR)
  - Firm size (market cap)
  - FinBERT sentiment score (for management only)
- Outputs dataset for clustering with full features

### Step 2: Semi-Supervised Labeling via Clustering

Run `Semi_Supervised_Labeling.ipynb`.

- Standardizes features: sentiment (method_2), surprise, firm size
- Removes outliers (IQR + z-score)
- Applies Gaussian Mixture Model clustering (k=6)
- Labels clusters using average CAR:
  - Bullish: highest CAR
  - Bearish: lowest clusters
  - Neutral: remaining
- Outputs:
  - `training_data.csv` (2020–2023)
  - `test_data.csv` (2024)

### Step 3: Supervised Classification using Random Forest

Run `Random_Forest.ipynb`.

- Trains Random Forest on GMM-labeled training data
- Features: `Earnings_Surprise`, `Firm_Size`, `method_2`
- Hyperparameter tuning via Grid Search
- Evaluates:
  - Accuracy, Precision, Recall, F1
  - Feature importance
  - SHAP summary plots

### Step 4: Portfolio Backtesting and Performance Evaluation

Run `Trading_Testing.ipynb`.

- Simulates trading decisions for:
  - **Portfolio A**: sentiment-only (FinBERT)
  - **Portfolio B**: ECSRS predictions
- Benchmarks against S&P 500
- Metrics:
  - CAGR
  - Sharpe Ratio
  - Max Drawdown
  - Volatility

### Step 5: ECSRS Inference Pipeline

Run `pipeline.ipynb`.

- Accepts:
  - Management transcript
  - Actual EPS
  - Estimated EPS
  - Firm size
- Returns:
  - ECSRS classification (Bullish / Neutral / Bearish)
  - FinBERT sentiment score
  - Prediction confidence

## 5. ECSRS Web Application (Streamlit)

### Launch the App

To run the ECSRS user interface:

```bash
streamlit run app.py
```

### Features

- Upload transcript (management only)
- Input EPS (actual & estimated) and firm size
- Displays:
  - Prediction (Bullish / Neutral / Bearish)
  - Sentiment score (FinBERT)
  - Classifier confidence
- Export predictions and browse historical outputs (SQLite)

## 6. Evaluation Metrics

The following metrics are used throughout the project:

| Metric          | Description                                     |
|-----------------|-------------------------------------------------|
| **CAGR**        | Compound Annual Growth Rate                     |
| **Sharpe Ratio**| Risk-adjusted return                           |
| **Max Drawdown**| Largest observed loss from peak to trough       |
| **Volatility**  | Standard deviation of returns                   |
| **Precision**   | Correct positive predictions / All positive predictions |
| **Recall**      | Correct positive predictions / All actual positives |
| **F1-Score**    | Harmonic mean of Precision and Recall           |

## 7. Citation

If using or referencing this system, please cite:

```
Gary Hii Ngo Cheong. (2025). Earnings Call Stock Ranking System (ECSRS): A Semi-Supervised Multi-Factor Learning Framework for Post-Earnings Stock Performance Prediction. University of Nottingham Malaysia. BSc Final Year Dissertation.
```

## Note on Data Availability

> **⚠️ Important:** The full dataset required to fully reproduce the ECSRS pipeline (including all CSV files generated during data collection, sentiment analysis, and feature engineering) is not included in the submitted ZIP file due to file size limitations.
>
> To access the complete version of the project with all required data files, please refer to the GitHub repository:  
> [https://github.com/gary30hii/EarningsCallStockRankingSystem--ECSRS](https://github.com/gary30hii/EarningsCallStockRankingSystem--ECSRS)
>
> All steps in the pipeline—including clustering, model training, and portfolio backtesting—are best evaluated using the GitHub version.
>
> Please ensure you are working from the GitHub repository if you intend to reproduce the results end-to-end.


