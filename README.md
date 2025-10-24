# Advanced Time Series Forecasting of U.S. Inflation using SARIMA, ETS, and XGBoost Models

## 1. Project Objective 🎯

This project implements and evaluates multiple time series models for forecasting the **U.S. inflation rate**. The primary goal is to leverage historical Consumer Price Index (CPI) data to generate reliable short-term inflation predictions, comparing the efficacy of **Seasonal Autoregressive Integrated Moving Average (SARIMA/SARIMAX)**, **Exponential Smoothing (ETS)**, and **XGBoost** methodologies. This work demonstrates proficiency in end-to-end time series analysis, including data acquisition, feature engineering, model selection, diagnostics, comparative evaluation, and forecasting.

---

## 2. Data Acquisition and Preparation 📊

* **Data Source:** Federal Reserve Economic Data (FRED) database (`pandas-datareader`).
* **Target Series:** `CPIAUCSL` (U.S. CPI, All Urban Consumers, All Items, SA).
* **Exogenous Series:**
    * `UNRATE` (U.S. Civilian Unemployment Rate, SA).
    * `FEDFUNDS` (Effective Federal Funds Rate).
* **Frequency:** Monthly
* **Time Period:** 1960-01-01 to Present (dynamically updated on execution).
* **Target Variable:** YoY Percentage Change of `CPIAUCSL`.
* **Preprocessing:** Forward-filled missing values in exogenous series; aligned all series by date index.

---

## 3. Exploratory Data Analysis (EDA) 📈

* Visualized the long-term YoY inflation rate (1961-Present) to identify trends, seasonality, and periods of volatility (e.g., 1970s energy crisis, 2009 recession, post-2020 surge). This informed the need for models capable of handling non-stationarity and seasonality.
    ``

---

## 4. Modeling Methodology 🛠️

### 4.1. Train/Test Split

* Data split chronologically: ~80% Training (1961-01-01 to 2012-08-01), ~20% Testing (2012-09-01 to Present).

### 4.2. SARIMA / SARIMAX Models

* **Parameter Selection:** `pmdarima.auto_arima` used on training data (with and without exogenous variables) to find optimal parameters minimizing AIC. Seasonality (`m=12`) included.
* **Best Model Found:** `SARIMAX(0, 1, 3)x(0, 0, 1, 12)` identified for both (exogenous variables did not improve AIC fit on training data).
* **Diagnostics:** Residual analysis performed (plots: residuals, histogram, Q-Q, ACF) indicated a reasonable fit with no significant remaining autocorrelation.

### 4.3. ETS Model (Benchmark)

* **Specification:** ETS(A,Ad,A) - Additive Error, Additive Damped Trend, Additive Seasonality (`m=12`) fitted to training data.

### 4.4. XGBoost Model (Machine Learning)

* **Feature Engineering:** Created lagged features for inflation (lags 1, 2, 3, 6, 12 months). Combined lags with exogenous variables (`UNRATE`, `FEDFUNDS`).
* **Model:** `xgboost.XGBRegressor` trained on the engineered features and target inflation from the training set. Standard hyperparameters used.

---

## 5. Evaluation and Results 🎯

Models were evaluated on the test set using RMSE and MAE:

| Model   | RMSE       | MAE        | Notes                                         |
| :------ | :--------- | :--------- | :-------------------------------------------- |
| SARIMA  | 2.0614     | 1.4520     | Baseline statistical model.                   |
| SARIMAX | 2.0614     | 1.4520     | Exog. variables didn't improve test metrics. |
| ETS     | 2.2708     | 1.4302     | Performed slightly worse than SARIMA.         |
| XGBoost | **0.4970** | **0.3668** | Significantly outperformed statistical models. |

* **Discussion:** The **XGBoost model demonstrated substantially better performance** on the test set, particularly in tracking the volatile post-2020 period. This highlights the advantage of feature engineering (lags) and machine learning algorithms for capturing complex dynamics in economic time series, especially during periods of structural change.
    ``

---

## 6. Future Forecast 🔮

* The best-performing **XGBoost model** was re-trained on the *entire* dataset.
* An iterative prediction process was used to generate forecasts for the next **12 months** (Sep 2025 - Aug 2026), using predicted values to update future lag features (assuming exogenous variables remain constant).
* **Result:** The XGBoost model predicts inflation **to quickly stabilize and remain consistently around 2.86-2.88% over the next year.**
    ``

---

## 7. Technologies Used 💻

* **Language:** Python 3.10
* **Core Libraries:**
    * `pandas` & `pandas-datareader`
    * `numpy`
    * `matplotlib`
    * `pmdarima` (for Auto SARIMA/SARIMAX)
    * `statsmodels` (for ETS)
    * `xgboost` (for XGBoost ML model)
    * `scikit-learn` (for metrics, train/test split helper)
* **Environment:** Jupyter Notebook (VS Code)

---

## 8. Potential Future Work 🚀

* **Advanced Feature Engineering:** Incorporate more sophisticated time-based features (e.g., cyclical features, interaction terms).
* **Hyperparameter Tuning:** Optimize XGBoost (and potentially SARIMA/ETS) parameters using GridSearchCV or RandomizedSearchCV.
* **Deep Learning Comparison:** Implement and compare LSTM or GRU models.
* **Exogenous Variable Forecasting:** Incorporate forecasts for exogenous variables instead of assuming they remain constant.
* **Rolling Forecast Evaluation:** Implement a more robust backtesting procedure.

---

## 9. How to Run ▶️

1.  Clone this repository: `git clone [Your Repository URL]`
2.  Navigate to the project directory: `cd inflation-forecasting`
3.  Create and activate a Python 3.10 virtual environment:
    ```bash
    py -3.10 -m venv venv
    .\venv\Scripts\activate
    ```
4.  Install dependencies:
    ```bash
    pip install jupyter pandas pandas-datareader matplotlib statsmodels pmdarima scikit-learn xgboost
    ```
5.  Launch Jupyter Notebook or run within VS Code:
    ```bash
    jupyter notebook inflation_forecasting.ipynb
    ```
    (Or simply open the `.ipynb` file in VS Code).