# Testen im Terminal:
# conda activate work
# cd /Users/matthiasmuschket/PycharmProjects/260414_Time_Series_Masterschool
# streamlit run "Streamlit App/app.py"

# ── IMPORTS ──────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import os

# ── KONFIGURATION ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecast Tool",
    layout="wide"
)

mlflow.set_tracking_uri("sqlite:////Users/matthiasmuschket/PycharmProjects/260414_Time_Series_Masterschool/mlflow.db")

# ── DATEN LADEN ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_data
def load_data():
    df = pd.read_csv(
        os.path.join(BASE_DIR, "Data", "Processed", "time_series_added_features.csv"),
        parse_dates=['date']
    )
    df_oil = pd.read_csv(
        os.path.join(BASE_DIR, "Data", "Raw", "oil.csv"),
        parse_dates=['date']
    )
    df = pd.merge(df, df_oil, how='left', on='date')
    return df

df = load_data()

@st.cache_resource
def load_model():
    runs = mlflow.search_runs(experiment_names=["sales_forecast"])
    best_run = runs.sort_values("metrics.SMAPE").iloc[0]
    model = mlflow.sklearn.load_model(f"runs:/{best_run.run_id}/model")
    return model, best_run

model, best_run = load_model()

# ── SIDEBAR NAVIGATION ───────────────────────────────────────────────
st.sidebar.title("Sales Forecast Tool")
st.sidebar.markdown("---")

if st.sidebar.button("Home"):
    st.session_state.page = "Home"
if st.sidebar.button("Data Overview"):
    st.session_state.page = "Data Overview"
if st.sidebar.button("Model Comparison & Selection"):
    st.session_state.page = "Model Comparison & Selection"
if st.sidebar.button("--- FORECAST ---"):
    st.session_state.page = "--- FORECAST ---"


# ── (0) Home-Button  ─────────────────────────────────────────────
# ── (0.1) ...  ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "Home"  # ← Standard ist Home, nicht Data Overview

page = st.session_state.page

if page == "Home":
    st.title("Sales Forecast Tool")
    st.markdown("""
    Welcome to the Sales Forecast Tool.

    This app helps shop owners to forecast future sales and make better business decisions.

    **Use the sidebar to navigate:**
    - **Data Overview** – Explore historical sales data
    - **Model Comparison** – Compare forecasting models
    - **Forecast** – Predict future sales
    - **Business Insights** – Get actionable recommendations
    """)

    st.write("")
    st.write("")

# ── (0.2) NOTE  ─────────────────────────────────────────────
    with st.expander("Limitations"):
        st.markdown(
            """**This project was created for learning purposes and demonstrates the application of various forecasting techniques including their individual limitations. No claim of production-ready performance is made.**""")

        st.markdown("""
                ### Important Limitations

                **(1) Selection of the best model**
                - Machine Learning (ML) models (XGBoost, Random Forest, Linear Regression) were evaluated using one-step-ahead forecasting
                - Statistical models (ARIMA, SARIMA, Triple ES, Prophet) were evaluated using iterative forecasting (simulating multi-day prediction)
                - => This gives ML models a structural advantage - direct metric comparison is therefore limited
                - Note: Lag and rolling features (see Feature Overview in Data Overview) are based on real observed values during one-step evaluation (teacher forcing). In iterative forecasting, these are progressively replaced by predicted values with increasing uncertainty with each additional forecasting step.

                **(2) Hyperparameter optimization**
                - **XGBoost:** Hyperopt + TimeSeriesSplit CV → no test set leakage. For each candidate hyperparameter set, the model is trained and evaluated across all CV folds; the mean fold score (SMAPE across folds) is returned to Hyperopt. The test set is never touched during this process. Only after the best hyperparameter set is selected, a final evaluation on the test set is performed.
                - **RF & LR:** Hyperopt without CV → mild leakage. The score returned to Hyperopt is computed directly on the test set, meaning hyperparameter selection is implicitly guided by test set performance. The chosen hyperparameters are therefore optimized for this specific test set rather than generalizing from held-out validation folds.

                **(3) Forecasting**
                - The champion model was trained on 2013 data only (test set = Q1 2014). In production, retraining on all available data would improve forecast quality.
                """)


# ── (1) DATA OVERVIEW ─────────────────────────────────────────────
if page == "Data Overview":
    st.title("Data Overview")
    st.write("Historical sales data")

    # ── (1.1) Time - Series Plot ──────────────────────────
    #st.subheader("Unit Sales Over Time")

    rolling_mean = df["unit_sales"].rolling(window=7).mean()

    with st.expander("Unit Sales Over Time"):
        fig_ts, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["date"], df["unit_sales"], label='Original', alpha=0.7)
        ax.plot(df["date"], rolling_mean, label='Rolling Mean (7 days)', color='red')
        ax.set_xlabel("Date")
        ax.set_ylabel("Unit Sales")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig_ts)

        st.caption(
          "Note: 2 missing days (2013-12-25 and 2014-01-01) were imputed using forward fill. Most forecasting models — both statistical and ML-based — cannot handle missing values in time series data.")

    #st.write("")
    #st.write("")

    # ── (1.2) ACF ──────────────────────────
    #st.subheader("Autocorrelation (ACF)")

    from statsmodels.graphics.tsaplots import plot_acf

    with st.expander("Autocorrelation (ACF)"):
        fig_acf, ax = plt.subplots(figsize=(12, 5))
        plot_acf(df['unit_sales'], ax=ax, lags=50)
        plt.tight_layout()
        st.pyplot(fig_acf)
        st.info("Significant spikes at lags 7, 14, 21 confirm 7-day seasonality in unit sales.")

    # ── (1.3) BOXPLOT: UNIT SALES BY DAY CATEGORY ──────────────────────────
    import seaborn as sns
    #st.subheader("Unit Sales by Day Category")

    # Day category ableiten
    def get_day_category(row):
        if row['is_holiday'] == 1 and row['is_weekday'] == 0:
            return 'holiday (weekend)'
        elif row['is_holiday'] == 1 and row['is_weekday'] == 1:
            return 'holiday (weekday)'
        elif row['is_weekday'] == 0:
            return 'weekend'
        else:
            return 'weekday'

    df['day_category'] = df.apply(get_day_category, axis=1)

    # n pro Kategorie berechnen
    n_counts = df.groupby('day_category')['unit_sales'].count()
    df['day_category_n'] = df['day_category'].map(
        lambda x: f"{x} (n={n_counts[x]})"
    )

    # Sortierung nach Median
    order = df.groupby('day_category_n')['unit_sales'].median().sort_values().index

    # Plot
    with st.expander("Unit Sales by Day Category"):
        fig_box, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='unit_sales', y='day_category_n', order=order, ax=ax)
        sns.stripplot(data=df, x='unit_sales', y='day_category_n',
                      color='black', alpha=0.3, size=5, order=order, ax=ax)
        ax.set_title('Unit Sales: Weekday vs. Weekend vs. Holiday')
        ax.grid(False)
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig_box)

        # T-Test als verschachtelter Expander
        with st.expander("T-Test Results"):
            st.markdown("""
            **Key Findings (Bonferroni-corrected p-values):**

            | Comparison | p-value (corrected) | Significant? | Interpretation |
            |---|---|---|---|
            | Weekday vs. Weekend | < 0.001 | **Yes** | Significantly more sales on weekends |
            | Holiday (weekday) vs. Weekday | 0.481 | No | Holidays on weekdays = regular weekdays |
            | Holiday (weekend) vs. Weekend | 1.000 | No | Holidays on weekends = regular weekends |

            **Conclusions:**
            > - No independent holiday effect — holidays behave like the respective day type
            > - Weekend is the key driver for higher sales
            """)
            st.caption("Note: Due to small sample sizes in holiday groups (n=9, n=15), the t-test was chosen as it is specifically designed for small samples. Bonferroni correction was applied to control for multiple testing (6 pairwise comparisons between 4 groups).")

    # ── (1.4) Ölpreis ──────────────────────────
    with st.expander("Oil Price vs. Unit Sales"):
        fig_oil, ax = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=df, x='unit_sales', y='dcoilwtico', ax=ax)
        ax.set_title("Unit Sales vs. Oil Price")
        ax.set_xlabel("Unit Sales")
        ax.set_ylabel("Oil Price (WTI)")
        plt.tight_layout()
        st.pyplot(fig_oil)

        st.info("The scatter plot shows no relationship between oil price and unit sales.")

    # ── (1.5) Features Used for Model Building ──────────────────────────
    with st.expander("Features Used for Model Building"):

        # (1.5.1) Eingangstext
        st.markdown("""
            Feature reduction was applied for **Linear Regression only**, as multicollinearity 
            distorts coefficient estimates. Tree-based models (XGBoost, Random Forest) handle 
            multicollinearity natively and were trained on the full feature set.

            **Features removed for Linear Regression:**
            - `is_weekday` — opposite of `is_weekend`
            - `dayofyear` — highly correlated with `month` (r=0.997)
            - `quarter` — highly correlated with `month` (r=0.974)
            - `weekofyear` — highly correlated with `month` (r=0.975)
            - `rolling_max_7/21/28`
            - `rolling_std_7/21/28`
            - `rolling_min_7/14/28`
            - `rolling_mean_7/14/28`
            """)

        # (1.5.2) Korrelationsmatrix erzeugen (Nur numerische Features, kein Datum und kein Target)
        with st.expander("Correlation Heatmap"):
            features = df.drop(columns=['date', 'unit_sales', 'day_category', 'day_category_n'], errors='ignore')
            corr = features.corr()

            fig_corr, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig_corr)

        st.write("")
        st.write("")


        # (1.5.3) Final Text
        st.markdown("""
        **Reduced set of features used for training of the Linear Regression model:**

        | Feature name| Description | Critical for Iterative Forecast? |
        |---|---|---|
        | dayofweek | Day of the week (0=Monday) | No |
        | dayofmonth | Day of the month | No |
        | month | Month of the year | No |
        | quarter | Quarter of the year | No |
        | is_weekend | Binary: weekend or not | No |
        | is_holiday | Binary: public holiday or not | No |
        | lag_7 | Sales 7 days ago | **Yes** |
        | lag_14 | Sales 14 days ago | **Yes** |
        | lag_21 | Sales 21 days ago | **Yes** |
        | lag_28 | Sales 28 days ago | **Yes** |
        | rolling_mean_7/14/21/28 | Rolling mean over 7/14/21/28 days | **Yes** |
        | rolling_std_7/14/21/28 | Rolling std over 7/14/21/28 days | **Yes** |
        | rolling_max_7/14/21/28 | Rolling max over 7/14/21/28 days | **Yes** |
        | rolling_min_7/14/21/28 | Rolling min over 7/14/21/28 days | **Yes** |
        """)


# ── PAGE 2: MODEL COMPARISON & SELECTION ──────────────────────────────────────────
elif page == "Model Comparison & Selection":
    st.title("Model Comparison")
    st.markdown("Overview of all forecasting models evaluated on Q1 2014 (Jan–Mar).")

    # ── (2.1) Intro Text ────────────────────────────────────────────────────
    st.markdown("""
    All model were optimized according to the **SMAPE (Symmetric Mean Absolute Percentage Error)** using Hyperopt for hyperparameter tuning:
    - Scale-independent → comparable across different time series
    - Symmetric → over- and underestimation are equally penalized
    - Robust against near-zero values (unlike MAPE)

    The Champion Model was selected based on the lowest SMAPE on the test set (Q1 2014, Jan–Mar).""")

    st.write("")
    st.write("")

    # ── (2.2) Runs aus MLflow laden ─────────────────────────────────────────
    runs = mlflow.search_runs(experiment_names=["sales_forecast"])

    metrics_cols = {
        "tags.mlflow.runName": "Model",
        "metrics.SMAPE": "SMAPE",
        "metrics.RMSE": "RMSE",
        "metrics.MAD": "MAD",
        "metrics.MAPE": "MAPE",
        "metrics.Bias": "Bias",
    }

    df_runs = runs[list(metrics_cols.keys())].rename(columns=metrics_cols)
    df_runs = df_runs.sort_values("SMAPE").reset_index(drop=True)
    champion = df_runs.iloc[0]["Model"]

    # ── (2.3) Visual Model Comparison ─────────────────────────────────────────
    with st.expander("Visual Model Comparison"):
        st.subheader("Forecast vs. Actual — All Models")

        client = mlflow.tracking.MlflowClient()
        runs = mlflow.search_runs(experiment_names=["sales_forecast"])

        for _, run in runs.iterrows():
            run_id = run["run_id"]
            model_name = run["tags.mlflow.runName"]

            try:
                local_path = client.download_artifacts(run_id, "forecast_plot.png", "/tmp")
                st.markdown(f"**{model_name}**")
                st.image(local_path)
                st.markdown("---")
            except Exception:
                st.warning(f"No forecast plot available for {model_name}")

    # ── (2.4) Model Metrics & Comparison ─────────────────────────────────────────
    # (2.4.1) Model Metric Tabelle
    with st.expander("Model Metrics & Comparison"):
        st.subheader("Model Metrics")
        # Metriken-Tabelle mit Hervorhebung des besten Wertes pro Spalte
        st.dataframe(
            df_runs[["Model", "SMAPE", "RMSE", "MAD", "MAPE", "Bias"]]
            .style
            # Niedrigster Wert = bestes Modell (grün markiert)
            .highlight_min(subset=["SMAPE", "RMSE", "MAD", "MAPE"], color="lightgreen")
            # Bias: Wert nahe 0 ist am besten → Absolutwert-Vergleich nötig
            .apply(
                lambda x: ['background-color: lightgreen' if abs(v) == min(abs(x)) else '' for v in x],
                subset=["Bias"]
            ),
            use_container_width=True
        )

        # (2.4.2) SMAPE Comparison Bar Chart
        st.subheader("SMAPE Comparison")
        fig_comp, ax = plt.subplots(figsize=(10, 5))
        colors = ["green" if m == champion else "steelblue" for m in df_runs["Model"]]
        ax.barh(df_runs["Model"], df_runs["SMAPE"], color=colors)
        ax.set_xlabel("SMAPE")
        ax.set_title("SMAPE by Model (lower is better)")
        ax.axvline(x=df_runs["SMAPE"].min(), color="green", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_comp)

        # (2.4.3) Kurze Interpretation der Ergebnisse
        st.markdown(f"""
                Model performance was evaluated based on **SMAPE (Symmetric Mean Absolute Percentage Error)** — 
                a scale-independent, symmetric error metric that measures the average percentage deviation 
                of the forecast relative to the magnitude of both actual and predicted values. 
                Unlike MAPE, SMAPE is robust against near-zero values and penalizes over- and underestimation equally.

                Model performance was relatively consistent across all approaches according to SMAPE. The champion model 
                **{champion}** achieved a SMAPE of **{df_runs.iloc[0]['SMAPE']:.4f}**, 
                which is **{((df_runs.iloc[1]['SMAPE'] - df_runs.iloc[0]['SMAPE']) / df_runs.iloc[1]['SMAPE'] * 100):.1f}%** 
                lower than the second-best model **{df_runs.iloc[1]['Model']}** 
                (SMAPE: {df_runs.iloc[1]['SMAPE']:.4f}).
                """)

        st.success(
            f"Champion Model: **{champion}** — selected based on the lowest SMAPE on the test set (Q1 2014, Jan–Mar)")

        st.write("")

        # (2.4.4) Button zur Feature Importance
        with st.expander("What Drives the Forecast? — Feature Importance of the Champion Model (Random Forest)"):

            client = mlflow.tracking.MlflowClient()

            # Champion Run finden
            champion_run = runs[runs["tags.mlflow.runName"] == champion].iloc[0]
            champion_run_id = champion_run["run_id"]

            try:
                # Feature Importance Plot aus MLflow laden
                local_path = client.download_artifacts(
                    champion_run_id, "feature_importance.png", "/tmp"
                )
                st.image(local_path)
            except Exception:
                st.warning("Feature importance plot not available.")

    # ── (2.5) Hyperparameter ─────────────────────────────────────────
    with st.expander("Model configuration: Hyperparameters"):

        st.caption(
            "Best hyperparameters were identified using the Hyperopt package for Python.")

        client = mlflow.tracking.MlflowClient()
        runs = mlflow.search_runs(experiment_names=["sales_forecast"])

        for _, run in runs.iterrows():
            model_name = run["tags.mlflow.runName"]
            run_id = run["run_id"]

            # Parameter aus MLflow laden
            params = client.get_run(run_id).data.params

            st.markdown(f"**{model_name}**")
            if params:
                st.json(params)
            else:
                st.write("No hyperparameters (e.g. Linear Regression)")
            st.markdown("---")

# ── PAGE 3: FORECAST ──────────────────────────────────────────────────
elif page == "--- FORECAST ---":
    st.title("--- FORECAST ---")

    st.info("""
    **Methodological Note:** The champion model was selected based on one-step-ahead evaluation. 
    In the app, however, iterative forecasting is applied for multi-day predictions to simulate 
    a realistic forecasting scenario. As a result, actual forecast performance may differ from 
    the evaluation metrics reported in the Model Comparison section.
    """)

    # ── Slider für Forecast-Horizont ──────────────────────────────────
    forecast_horizon = st.slider(
        "Forecast Horizon (days)",
        min_value=7,
        max_value=90,
        value=30,
        step=1
    )

    # ── Iterativer Forecast ───────────────────────────────────────────
    df_forecast = df.copy()
    predictions = []

    for i in range(forecast_horizon):
        last_date = df_forecast['date'].max()
        next_date = last_date + pd.Timedelta(days=1)

        new_row = {
            'date':        next_date,
            'dayofweek':   next_date.dayofweek,
            'dayofmonth':  next_date.day,
            'dayofyear':   next_date.dayofyear,
            'weekofyear':  next_date.isocalendar().week,
            'month':       next_date.month,
            'quarter':     next_date.quarter,
            'is_weekend':  int(next_date.dayofweek >= 5),
            'is_weekday':  int(next_date.dayofweek < 5),
            'is_holiday':  0,
        }

        all_sales = df_forecast['unit_sales'].values

        new_row['lag_7']  = all_sales[-7]
        new_row['lag_14'] = all_sales[-14]
        new_row['lag_21'] = all_sales[-21]
        new_row['lag_28'] = all_sales[-28]

        new_row['rolling_mean_7']  = np.mean(all_sales[-7:])
        new_row['rolling_mean_14'] = np.mean(all_sales[-14:])
        new_row['rolling_mean_21'] = np.mean(all_sales[-21:])
        new_row['rolling_mean_28'] = np.mean(all_sales[-28:])

        new_row['rolling_std_7']   = np.std(all_sales[-7:])
        new_row['rolling_std_14']  = np.std(all_sales[-14:])
        new_row['rolling_std_21']  = np.std(all_sales[-21:])
        new_row['rolling_std_28']  = np.std(all_sales[-28:])

        new_row['rolling_max_7']   = np.max(all_sales[-7:])
        new_row['rolling_max_14']  = np.max(all_sales[-14:])
        new_row['rolling_max_21']  = np.max(all_sales[-21:])
        new_row['rolling_max_28']  = np.max(all_sales[-28:])

        new_row['rolling_min_7']   = np.min(all_sales[-7:])
        new_row['rolling_min_14']  = np.min(all_sales[-14:])
        new_row['rolling_min_21']  = np.min(all_sales[-21:])
        new_row['rolling_min_28']  = np.min(all_sales[-28:])

        # feature_cols bereinigen
        feature_cols = [col for col in df.columns if col not in [
            'date', 'unit_sales', 'day_category', 'day_category_n',
            'Unnamed: 0', 'dcoilwtico'  # ← diese entfernen
        ]]

        X_new = pd.DataFrame([new_row])[feature_cols]

        y_pred = model.predict(X_new)[0]
        predictions.append({'date': next_date, 'unit_sales_forecast': y_pred})

        new_row['unit_sales'] = y_pred
        df_forecast = pd.concat([df_forecast, pd.DataFrame([new_row])], ignore_index=True)

    df_predictions = pd.DataFrame(predictions)

    # ── Plot ──────────────────────────────────────────────────────────
    fig_fc, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df['date'], df['unit_sales'], label='Historical', color='steelblue')
    ax.plot(df_predictions['date'], df_predictions['unit_sales_forecast'],
            label='Forecast', color='red', linewidth=2)
    ax.axvline(x=df['date'].max(), color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
    ax.set_xlabel("Date")
    ax.set_ylabel("Unit Sales")
    ax.set_title(f"Sales Forecast — Next {forecast_horizon} Days")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig_fc)

    # ── Forecast Tabelle ──────────────────────────────────────────────
    with st.expander("Forecast Table"):
        st.dataframe(
            df_predictions.rename(columns={
                'date': 'Date',
                'unit_sales_forecast': 'Forecasted Unit Sales'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.caption("Uncertainty increases with forecast horizon.")

    # ── Business Insight: Lagerplanung ───────────────────────────────────
    st.markdown("---")
    st.subheader("Inventory Planning")

    current_stock = st.number_input(
        "Current stock (units in store):",
        min_value=0,
        value=500,
        step=50
    )

    # Gesamtverkauf im Forecast-Horizont
    total_forecast = int(df_predictions['unit_sales_forecast'].sum())

    # Differenz
    reorder_needed = total_forecast - current_stock

    st.metric("Expected total sales", f"{total_forecast} units")
    st.metric("Current stock", f"{current_stock} units")

    if reorder_needed > 0:
        st.error(f"Reorder needed: **{reorder_needed} units** to cover the forecast period.")
    else:
        st.success(f"Stock sufficient. Expected surplus: **{abs(reorder_needed)} units**")




