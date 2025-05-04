Earnings Call Stock Ranking System (ECSRS)
==========================================

1. Project Overview
-------------------

This repository implements the Earnings Call Stock Ranking System (ECSRS), a multi-factor machine learning model that predicts post-earnings stock performance. It integrates sentiment analysis of earnings calls with earnings surprises and firm size to capture the Post-Earnings Announcement Drift (PEAD) phenomenon. The system uses a semi-supervised clustering strategy for label generation and trains a Random Forest classifier to categorize earnings events as Bullish, Neutral, or Bearish. A Streamlit-based web application allows users to interact with the system through an intuitive interface.

2. System Architecture
----------------------

The ECSRS pipeline consists of the following components:

1. Data Collection & Preprocessing
2. Sentiment Analysis (FinBERT)
3. Cumulative Abnormal Return (CAR) Computation
4. Unsupervised Clustering & Label Assignment
5. Supervised Classification using Random Forest
6. Portfolio Backtesting
7. Real-time Prediction Interface (Streamlit App)

3. Setup Instructions
---------------------

1. Clone the repository:

   git clone https://github.com/gary30hii/EarningsCallStockRankingSystem--ECSRS.git
   cd EarningsCallStockRankingSystem--ECSRS

2. Install the required Python packages:

   You can install them manually, or run:

   pip install -r requirements.txt

   Key dependencies:
   - pandas, numpy, scikit-learn, shap, streamlit, transformers,
     requests, torch, nltk, matplotlib, seaborn

3. LLaMA Sentiment Analysis (Optional but Recommended):

   - Install Ollama from https://ollama.com
   - Download LLaMA 3.2 3B model:

     ollama run llama3:3b

   - Update the MODEL_NAME variable in Stock_list.ipynb
     (e.g., llama3.2:latest)

   Note: LLaMA enables comparison with FinBERT-based sentiment analysis.

4. Insert your Financial Modeling Prep (FMP) API key
   into Stock_list.ipynb to enable stock data collection.

4. Step-by-Step Reproduction Guide
----------------------------------

Step 1: Data Collection, Sentiment Analysis, and Feature Engineering

- Run Stock_list.ipynb
- Collects data on ~500 US stocks (2019–2024)
- Computes:
    - Earnings surprise
    - CAR (Cumulative Abnormal Return)
    - Firm size (market cap)
    - FinBERT sentiment score (management only)

Step 2: Semi-Supervised Labeling via Clustering

- Run Semi_Supervised_Labeling.ipynb
- Standardizes features and removes outliers
- Applies Gaussian Mixture Model (k=6)
- Labels clusters based on average CAR:
    - Bullish: highest CAR
    - Bearish: lowest
    - Neutral: others
- Outputs:
    - training_data.csv (2020–2023)
    - test_data.csv (2024)

Step 3: Supervised Classification (Random Forest)

- Run Random_Forest.ipynb
- Trains model on GMM-labeled data
- Features: Earnings_Surprise, Firm_Size, method_2
- Grid Search for hyperparameters
- Evaluation: accuracy, precision, recall, F1-score
- Visualizations: feature importance, SHAP plots

Step 4: Portfolio Backtesting and Evaluation

- Run Trading_Testing.ipynb
- Simulates trading:
    - Portfolio A: FinBERT-only
    - Portfolio B: ECSRS predictions
- Benchmarks vs. S&P 500
- Metrics: CAGR, Sharpe Ratio, Max Drawdown, Volatility

Step 5: ECSRS Inference Pipeline

- Run pipeline.ipynb
- Inputs:
    - Transcript (management)
    - Actual EPS
    - Estimated EPS
    - Firm size
- Outputs:
    - ECSRS classification (Bullish, Neutral, Bearish)
    - FinBERT sentiment score
    - Prediction confidence

5. ECSRS Web Application (Streamlit)
------------------------------------

To launch the app:

   streamlit run app.py

Features:
- Upload management transcript
- Input EPS values and firm size
- Output:
    - ECSRS prediction
    - Sentiment score (FinBERT)
    - Model confidence
- Historical predictions saved via SQLite

6. Evaluation Metrics
---------------------

Metric Descriptions:

- CAGR: Compound Annual Growth Rate
- Sharpe Ratio: Risk-adjusted return
- Max Drawdown: Largest peak-to-trough drop
- Volatility: Standard deviation of returns
- Precision: Correct positives / Predicted positives
- Recall: Correct positives / Actual positives
- F1-Score: Harmonic mean of precision and recall

7. Citation
-----------

If using or referencing this system, please cite:

Gary Hii Ngo Cheong. (2025). Earnings Call Stock Ranking System (ECSRS): A Semi-Supervised Multi-Factor Learning Framework for Post-Earnings Stock Performance Prediction. University of Nottingham Malaysia. BSc Final Year Dissertation.
