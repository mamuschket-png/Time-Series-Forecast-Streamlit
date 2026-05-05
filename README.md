# Retail Sales Forecasting — Streamlit App

A machine learning-powered web application that enables shop owners to forecast 
future unit sales, understand key sales drivers, and optimize inventory planning.

Built as part of a Data Science Master's program, this project demonstrates the 
full ML workflow — from exploratory data analysis and feature engineering to 
hyperparameter optimization, model evaluation and interactive deployment via Streamlit.

*Created for learning purposes. No claim of production-ready performance is made.*

---

## What the App Does

- **Data Overview** — Explore historical sales data, seasonality patterns and key drivers influencing sales
- **Model Comparison** — Compare 7 forecasting models:
  - Statistical: ARIMA, SARIMA, Triple Exponential Smoothing, Prophet
  - Machine Learning: XGBoost, Random Forest, Linear Regression
- **Forecast** — Predict future sales using the champion model (Random Forest) with iterative forecasting
- **Inventory Planning** — Estimate reorder needs based on forecasted sales and current stock

---

## Project Structure
```
retail-sales-forecasting/
├── Data/
│   ├── Processed/          # Cleaned and feature-engineered data
│   └── Raw/                # Original data files
├── Scripts/                # Jupyter Notebooks (EDA, modeling, MLflow)
├── Streamlit App/
│   └── app.py              # Main Streamlit application
├── mlruns/                 # MLflow tracking artifacts
├── mlflow.db               # MLflow backend store
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.11
- Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configuration

No configuration needed — all paths are set relative to the project root.
Simply clone the repository and run the app.

---

## How to Start the App

```bash
cd retail-sales-forecasting
streamlit run "Streamlit App/app.py"
```

Then open your browser at: **http://localhost:8501**

---

## How the Forecast Works

1. The champion model (Random Forest) was selected based on the lowest SMAPE on the test set (Q1 2014)
2. Features used for prediction:
   - **Lag features** — unit sales from 7, 14, 21 and 28 days ago
   - **Rolling statistics** — mean, std, min and max over 7, 14, 21 and 28-day windows
   - **Time-based features** — day of week, day of month, month, quarter, is_weekend
   - **Calendar features** — is_holiday
3. Iterative forecasting: each predicted value is used as input for the next step — simulating a realistic forecasting scenario
4. Uncertainty increases with forecast horizon — results beyond 30 days should be interpreted with caution

---

## Known Limitations

*This project was created for learning purposes and demonstrates the application of 
various forecasting techniques including their individual limitations. 
No claim of production-ready performance is made.*

**(1) Forecasting**
- ML models (XGBoost, Random Forest, Linear Regression) were evaluated using **one-step-ahead forecasting**
- Statistical models (ARIMA, SARIMA, Triple ES, Prophet) were evaluated using **iterative forecasting** (simulating multi-day prediction)
- This gives ML models a structural advantage — direct metric comparison is therefore limited
- Lag and rolling features are based on real observed values during one-step evaluation. In iterative forecasting, these are progressively replaced by predicted values — increasing uncertainty with each additional forecasting step.

**(2) Hyperparameter Optimization**
- XGBoost: Hyperopt + Cross-Validation (TimeSeriesSplit) → no data leakage from test set into training
- RF & LR: Hyperopt without CV → mild data leakage from test set into hyperparameter selection
  - Hyperopt evaluates directly on the test set using real (observed) lag features instead of predicted values → the model sees information unavailable in true forecasting
  - Result: hyperparameters are implicitly tuned for one-step-ahead, not multi-step forecasting

**(3) Selection of the Best Model**
- Champion model was selected based on the lowest SMAPE (scale-independent, symmetric error metric robust against near-zero values)
- Due to the structural advantage of ML models (one-step vs. iterative), the comparison is limited
- A fair comparison would require consistent evaluation methodology across all models

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `mlflow.db` not found | Ensure `mlflow.db` is in the project root |
| Data file not found | Ensure `Data/Processed/` folder exists after cloning |
| Module not found | Run `pip install -r requirements.txt` |

---

## License & Credits

- Data: Provided as part of the MasterSchool Data Science curriculum
- Developed as part of a Data Science Master's program at MasterSchool
- Author: Matthias Muschket