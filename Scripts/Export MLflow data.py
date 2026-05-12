"""
MLflow Export Script
====================
Einmalig lokal ausführen um alle MLflow-Daten als Dateien zu exportieren.

Ausführen im Terminal:
    conda activate work
    cd /Users/matthiasmuschket/PycharmProjects/260414_Time_Series_Masterschool
    python export_mlflow.py
"""

import mlflow
import mlflow.sklearn
import pickle
import json
import os
import shutil

# ── KONFIGURATION ─────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR    = os.path.dirname(SCRIPT_DIR)
TRACKING_URI   = f"sqlite:///{os.path.join(PROJECT_DIR, 'mlflow.db')}"
EXPERIMENT_NAME = "sales_forecast"
OUTPUT_DIR     = os.path.join(PROJECT_DIR, "Streamlit App", "exports")

# ── SETUP ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
mlflow.set_tracking_uri(TRACKING_URI)
client = mlflow.tracking.MlflowClient()

print("Lade Runs aus MLflow...")
runs = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
runs = runs.sort_values("metrics.SMAPE").reset_index(drop=True)
print(f"{len(runs)} Runs gefunden.")

# ── 1. CHAMPION-MODELL als Pickle exportieren ─────────────────────────────────
print("\n[1/4] Exportiere Champion-Modell...")
best_run = runs.iloc[0]
model = mlflow.sklearn.load_model(f"runs:/{best_run.run_id}/model")
model_path = os.path.join(OUTPUT_DIR, "champion_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"    Gespeichert: {model_path}")
print(f"    Champion: {best_run['tags.mlflow.runName']} (SMAPE: {best_run['metrics.SMAPE']:.4f})")

# ── 2. METRIKEN als CSV exportieren ───────────────────────────────────────────
print("\n[2/4] Exportiere Metriken...")
metrics_cols = {
    "tags.mlflow.runName": "Model",
    "metrics.SMAPE": "SMAPE",
    "metrics.RMSE": "RMSE",
    "metrics.MAD": "MAD",
    "metrics.MAPE": "MAPE",
    "metrics.Bias": "Bias",
}
df_metrics = runs[list(metrics_cols.keys())].rename(columns=metrics_cols)
df_metrics = df_metrics.sort_values("SMAPE").reset_index(drop=True)
metrics_path = os.path.join(OUTPUT_DIR, "metrics.csv")
df_metrics.to_csv(metrics_path, index=False)
print(f"    Gespeichert: {metrics_path}")
print(df_metrics.to_string(index=False))

# ── 3. HYPERPARAMETER als JSON exportieren ────────────────────────────────────
print("\n[3/4] Exportiere Hyperparameter...")
all_params = {}
for _, run in runs.iterrows():
    run_id = run["run_id"]
    model_name = run["tags.mlflow.runName"]
    params = client.get_run(run_id).data.params
    all_params[model_name] = params
    print(f"    {model_name}: {len(params)} Parameter")

params_path = os.path.join(OUTPUT_DIR, "params.json")
with open(params_path, "w") as f:
    json.dump(all_params, f, indent=2)
print(f"    Gespeichert: {params_path}")

# ── 4. PLOTS (Forecast + Feature Importance) exportieren ─────────────────────
print("\n[4/4] Exportiere Plots...")
plots_dir = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(plots_dir, exist_ok=True)

for _, run in runs.iterrows():
    run_id = run["run_id"]
    model_name = run["tags.mlflow.runName"]
    safe_name = model_name.replace(" ", "_").replace("/", "_")

    # Forecast Plot
    try:
        local_path = client.download_artifacts(run_id, "forecast_plot.png", "/tmp")
        dest = os.path.join(plots_dir, f"{safe_name}_forecast.png")
        shutil.copy(local_path, dest)
        print(f"    Forecast-Plot gespeichert: {dest}")
    except Exception as e:
        print(f"    Kein Forecast-Plot für {model_name}: {e}")

    # Feature Importance Plot (nur Champion)
    try:
        local_path = client.download_artifacts(run_id, "feature_importance.png", "/tmp")
        dest = os.path.join(plots_dir, f"{safe_name}_feature_importance.png")
        shutil.copy(local_path, dest)
        print(f"    Feature-Importance-Plot gespeichert: {dest}")
    except Exception:
        pass  # Nicht alle Modelle haben Feature Importance

# ── ZUSAMMENFASSUNG ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Export abgeschlossen! Folgende Dateien wurden erstellt:")
for root, dirs, files in os.walk(OUTPUT_DIR):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        print(f"  {filepath}  ({size/1024:.1f} KB)")
print("="*60)
print("\nNächster Schritt: Alle Dateien in 'Streamlit App/exports/' committen und pushen.")