# Advanced Time Series Forecasting of U.S. Inflation using SARIMA and ETS Models

## 1. Project Objective 🎯

This project implements and evaluates statistical time series models for forecasting the **U.S. inflation rate**. The primary goal is to leverage historical Consumer Price Index (CPI) data to generate reliable short-term inflation predictions, comparing the efficacy of **Seasonal Autoregressive Integrated Moving Average (SARIMA)** and **Exponential Smoothing (ETS)** methodologies. This work demonstrates proficiency in end-to-end time series analysis, from data acquisition and preprocessing to model selection, diagnostics, evaluation, and forecasting.

---

## 2. Data Acquisition and Preparation 📊

* **Data Source:** Federal Reserve Economic Data (FRED) database, accessed via the `pandas-datareader` library.
* **Series ID:** `CPIAUCSL` (Consumer Price Index for All Urban Consumers: All Items in U.S. City Average, Seasonally Adjusted). This is the standard headline CPI measure reported by the U.S. Bureau of Labor Statistics (BLS).
* **Frequency:** Monthly
* **Time Period:** 1960-01-01 to Present (dynamically updated on execution).
* **Target Variable:** The raw CPI index was transformed into the **Year-over-Year (YoY) Percentage Change** to represent the inflation rate, a standard practice in economic analysis. Initial `NaN` values resulting from the 12-month lag were removed.

---

## 3. Exploratory Data Analysis (EDA) 📈

* The YoY inflation rate time series (1961-Present) was plotted to visually inspect long-term trends, seasonality, and structural breaks.
* Key observations included periods of high volatility (e.g., 1970s-1980s), periods of relative stability (e.g., 1990s-2000s), the deflationary shock around 2009, and the recent inflationary surge post-2020. This visual analysis informed the need for models capable of handling non-stationarity and potential seasonality.
    ``

---

## 4. Modeling Methodology 🛠️

### 4.1. Train/Test Split

* The dataset was split chronologically to preserve temporal dependencies:
    * **Training Set:** First ~80% of the data (1961-01-01 to 2012-08-01).
    * **Test Set:** Remaining ~20% of the data (2012-09-01 to Present).

### 4.2. SARIMA Model

* **Parameter Selection:** The `pmdarima.auto_arima` function was employed on the training set to perform an automated stepwise search for the optimal SARIMA(p,d,q)(P,D,Q)m parameters, minimizing the **Akaike Information Criterion (AIC)**. Seasonality (`m=12`) was explicitly included.
* **Best Model Identified:** `SARIMAX(0, 1, 3)x(0, 0, 1, 12)`
* **Model Diagnostics:** Standard residual diagnostics indicated the model captured most autocorrelation, although residuals showed some deviation from perfect normality, potentially due to historical volatility.

### 4.3. ETS Model (Benchmark)

* An **Exponential Smoothing (ETS)** model (ETS(A,Ad,A) - Additive Error, Additive Damped Trend, Additive Seasonality, m=12) was fitted to the training data as a benchmark comparison.

---

## 5. Evaluation and Results 🎯

Both models generated forecasts over the test set horizon. Performance was evaluated using:

* **Root Mean Squared Error (RMSE):** Penalizes larger errors more heavily.
* **Mean Absolute Error (MAE):** Represents the average magnitude of the errors.

| Model  | RMSE   | MAE    |
| :----- | :----- | :----- |
| SARIMA | **2.0614** | 1.4520 |
| ETS    | 2.2708 | **1.4302** |

* **Discussion:** The **SARIMA model achieved a lower RMSE**, indicating better performance in minimizing larger forecast deviations on the test set. The **ETS model had a slightly lower MAE**, suggesting its average error magnitude was marginally smaller. Visual inspection confirmed both models captured the general level but struggled with the sharp post-2020 volatility. Given its better RMSE, SARIMA was selected for the final future forecast.
    ``

---

## 6. Future Forecast 🔮

* The chosen **SARIMA model (`SARIMAX(0, 1, 3)x(0, 0, 1, 12)`)** was re-trained on the *entire* dataset (1961-Present).
* Forecasts were generated for the next **12 months** (Sep 2025 - Aug 2026).
* **Result:** The model predicts inflation to fluctuate between approximately 2.8% and 3.4% over the next year, staying relatively stable around the 3% mark.
    ``

---

## 7. Technologies Used 💻

* **Language:** Python 3.10
* **Core Libraries:**
    * `pandas` & `pandas-datareader`: Data acquisition, manipulation, time series handling.
    * `numpy`: Numerical computation.
    * `matplotlib`: Data visualization.
    * `pmdarima`: Automated SARIMA parameter selection, diagnostics, forecasting.
    * `statsmodels`: ETS modeling, statistical tests, diagnostic tools.
    * `scikit-learn`: `train_test_split`, `mean_squared_error`, `mean_absolute_error`.
* **Environment:** Jupyter Notebook within Visual Studio Code.

---

## 8. Potential Future Work 🚀

* **Incorporate Exogenous Variables:** Extend SARIMAX to include predictors like unemployment rate (`UNRATE`), federal funds rate (`FEDFUNDS`), etc.
* **Alternative Models:** Compare against Machine Learning (e.g., XGBoost) or Deep Learning (e.g., LSTM) models.
* **Rolling Forecast Evaluation:** Implement a more rigorous backtesting strategy.
* **Probabilistic Forecasting:** Generate prediction intervals to quantify uncertainty.

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
    pip install jupyter pandas pandas-datareader matplotlib statsmodels pmdarima scikit-learn
    ```
5.  Launch Jupyter Notebook or run within VS Code:
    ```bash
    jupyter notebook inflation_forecasting.ipynb
    ```
    (Or simply open the `.ipynb` file in VS Code).