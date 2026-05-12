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
## Project Structure

  retail-sales-forecasting/
    ├── Data/
    │   ├── Processed/          # Cleaned and feature-engineered data
    │   └── Raw/                # Original data files
    ├── Scripts/                # Jupyter Notebooks (EDA, modeling, MLflow export)
    │   ├── 0_Data_Preparation_and_EDA.ipynb        # EDA (Exploratory Data Analysis)
    │   ├── 1_Statistical_models.ipynb              # ARIMA, SARIMA, Triple ES, Prophet
    │   ├── 2_Feature_Engineering_Models.ipynb      # XGBoost, Random Forest, Linear Regression
    │   └── Export MLflow data.py                   # One-time export script for Streamlit deployment
    ├── Streamlit App/
    │   ├── app.py              # Main Streamlit application
    │   └── exports/            # Pre-exported model artifacts (no MLflow at runtime)
    │       ├── champion_model.pkl
    │       ├── metrics.csv
    │       ├── params.json
    │       └── plots/          # Forecast & feature importance plots
    ├── mlruns/                 # MLflow tracking artifacts (local only)
    ├── mlflow.db               # MLflow backend store (local only)
    ├── requirements.txt
    └── README.md
```

---

## MLflow & Deployment Notes

MLflow was used during model development for experiment tracking, hyperparameter 
logging, and artifact storage. However, MLflow is not compatible with Streamlit Cloud 
due to two fundamental issues:

1. **Absolute artifact paths** — MLflow stores artifact paths as absolute local paths 
   in the tracking database. These paths do not exist on Streamlit Cloud.
2. **Protobuf incompatibility** — MLflow 3.x uses an outdated protobuf API that is 
   incompatible with the protobuf versions installed on Streamlit Cloud (Python 3.14 
   environment).

### Workaround
All MLflow artifacts were exported once locally using `Scripts/Export MLflow data.py`:
- Champion model → `Streamlit App/exports/champion_model.pkl`
- Model metrics → `Streamlit App/exports/metrics.csv`
- Hyperparameters → `Streamlit App/exports/params.json`
- Forecast & feature importance plots → `Streamlit App/exports/plots/`

The Streamlit app loads these files directly, with no dependency on MLflow at runtime.

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
- Machine Learning (ML) models (XGBoost, Random Forest, Linear Regression) were evaluated using one-step-ahead forecasting
- Statistical models (ARIMA, SARIMA, Triple ES, Prophet) were evaluated using iterative forecasting (simulating multi-day prediction)
- => This gives ML models a structural advantage - direct metric comparison is therefore limited
- Note: Lag and rolling features (see Feature Overview in Data Overview) are based on real observed values during one-step evaluation (teacher forcing). In iterative forecasting, these are progressively replaced by predicted values with increasing uncertainty with each additional forecasting step.

**(2) Hyperparameter Optimization**
- **XGBoost:** Hyperopt + TimeSeriesSplit CV → no test set leakage. For each candidate hyperparameter set, the model is trained and evaluated across all CV folds; the mean fold score (SMAPE across folds) is returned to Hyperopt. The test set is never touched during this process. Only after the best hyperparameter set is selected, a final evaluation on the test set is performed.
- **RF & LR:** Hyperopt without CV → mild leakage. The score returned to Hyperopt is computed directly on the test set, meaning hyperparameter selection is implicitly guided by test set performance. The chosen hyperparameters are therefore optimized for this specific test set rather than generalizing from held-out validation folds.

**(3) Forecasting**
- The champion model was trained on 2013 data only (test set = Q1 2014). In production, retraining on all available data would improve forecast quality.
- Case-specific model optimization, tailored to the required forecast horizon, may further improve forecast quality.
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