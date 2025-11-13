# 003-Beta

## Overview

003-Beta is an advanced Python project for time series and quantitative modeling. It includes a complete pipeline for training, evaluating, calibrating, backtesting, and deploying deep learning (1D CNN) models for classification-oriented financial prediction, with an integrated API for real-time and batch inference. This repository leverages MLflow for experiment tracking, model management, and metric/artifact logging.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Data Preparation](#data-preparation)
- [Running the Main Pipeline](#running-the-main-pipeline)
- [MLflow Integration](#mlflow-integration)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Streamlit Dashboard](#streamlit-dashboard)
- [Notes](#notes)
- [References](#references)

---

## Features

- End-to-end pipeline for deep learning classification and backtesting.
- MLflow-based experiment tracking, parameter, metric, and artifact management.
- Temperature calibration, threshold tuning, and distribution adaptation (prior shift/BBSE).
- API for online prediction and model serving (FastAPI).
- Utilities for model drift monitoring, feature exploration, normalization, and pipeline modularization.
- Ready-to-use scripts for training, evaluation, and model analysis.

---

## Project Structure

(Partial listing due to API result limits; explore the [repository tree](https://github.com/Alex6606/003-Beta/tree/main/) for the full structure.)

```
.
├── api_server.py           # FastAPI REST API server
├── backtest.py             # Advanced backtesting utilities
├── calibration.py          # Temperature calibration routines
├── cnn_model.py            # 1D CNN model definition
├── data/                   # Data directory
├── data_split.py           # Dataset splits
├── data_utils.py           # Data loading/utility helpers
├── drift_dashboard.py      # Model drift Streamlit dashboard
├── drift_monitor.py        # Drift monitoring/logging
├── feature_inspection.py   # Feature inspection scripts
├── features_auto.py        # Automated feature engineering
├── features_pipeline.py    # Feature engineering pipeline
├── indicators.py           # Financial indicators
├── labeling.py             # Label generation/processing
├── latest_model_uri.txt    # (Auto-generated) path to latest trained MLflow model
├── losses.py               # Loss functions (Focal, etc.)
├── main.py                 # Entrypoint for the full pipeline
├── metrics.py              # Evaluation metrics
├── mlflow_tracking.py      # MLflow integration helpers
├── normalization.py        # Normalization routines
├── prior_correction.py     # Prior distribution correction (BBSE)
├── requirements.txt        # Python requirements
├── sequence_builder.py     # Sequence construction (time series)
├── streamlit_app.py        # (Optional) Streamlit interface
├── threshold_tuning.py     # Threshold selection/tuning
├── trainer.py              # Model training loop
├── training_pipeline.py    # End-to-end pipeline logic
├── visualization.py        # Visualization utilities
```

---

## Setup & Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/Alex6606/003-Beta.git
   cd 003-Beta
   ```

2. **Install Python Dependencies**
   ```sh
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Data Preparation**
   - Place raw data in the `data/` folder as expected by scripts such as `data_utils.py`.

---

## Running the Main Pipeline

The entire process—data loading, model training, evaluation, MLflow tracking, artifact generation, and preparing the model for serving—is orchestrated by the `main.py` script.

**To launch the pipeline:**

```sh
python main.py
```

This single command will:
- Execute all pipeline steps (data split, model build, training, calibration, threshold tuning, and backtesting).
- Log all relevant parameters, metrics, and artifacts in MLflow.
- Output the latest trained model’s MLflow URI to `latest_model_uri.txt` (for the API server).

**There is no need to compose or run pipeline functions manually—just run `main.py`.**

---

## MLflow Integration

MLflow is extensively used to track parameters, metrics, artifacts, and models.

### Tracking Setup

- MLflow logs run data to a local directory (`./mlruns`) by default (see `mlflow_tracking.py`):
  ```python
  mlflow.set_tracking_uri("file:./mlruns")
  ```
- Each experiment run logs full results, parameters (F1, accuracy, etc.), and relevant artifacts (thresholds, biases, calibration, feature info, model snapshot, etc.).
- Models are saved as MLflow artifacts and referenced by their `model_uri` (see also `latest_model_uri.txt`).

**To Start the MLflow UI (optional):**
```sh
mlflow ui
```
(then visit http://localhost:5000)

---

## Running the API

`api_server.py` provides a FastAPI-powered REST API for model prediction and analytics.

### Steps

1. **Run the pipeline first** (`python main.py`) so that `latest_model_uri.txt` gets created with the trained model path.

2. **Start the API server:**

   ```sh
   uvicorn api_server:app --host 127.0.0.1 --port 9999 --reload
   ```
   By default, the API loads the model referenced in `latest_model_uri.txt`.

3. **Check model health:**

   - Occasionally, the model may not load successfully on the first API server start.
   - Always check the health status after launching the API:
     - Visit [`http://127.0.0.1:9999/health`](http://127.0.0.1:9999/health) in your browser or use an HTTP tool.
     - If the health endpoint shows that the model is **not loaded** or returns an error, **restart the API server from your terminal** until the model is loaded correctly.

4. **Browse and interact:**
   - API documentation is available at [http://127.0.0.1:9999/docs](http://127.0.0.1:9999/docs) (Swagger UI).

---

## API Endpoints

Sample endpoints (see source for more):

- `GET /`             — API health/main response.
- `GET /health`       — Model health status (check here after every server start as noted above).
- `GET /schema`       — Returns model feature schema.
- `POST /reload`      — Reloads the model from latest artifact.
- `POST /predict-ticker` — Predicts signals for OHLCV bar series.
    - Input: JSON with ticker data and optional parameters.
    - Output: Prediction (buy/hold/sell) and probability/confidence.
- Plus additional endpoints for detailed predictions, feature stats, etc.

**Example cURL Request:**
```sh
curl -X POST "http://127.0.0.1:9999/predict-ticker" \
     -H  "accept: application/json" \
     -H  "Content-Type: application/json" \
     -d '{"ticker": "AAPL","lookback_days": 2500}'
```

---

## Streamlit Dashboard

The project includes an interactive dashboard for monitoring drift, insights, and visual explorations.

**To activate:**

```sh
streamlit run drift_dashboard.py
```

- This launches an interactive dashboard (in your browser), enabling real-time drift diagnostics, feature explorations, and other visual analysis.
- You can also run `streamlit run streamlit_app.py` if you use the optional dashboard.

---

## Notes

- The latest model's MLflow URI is stored in `latest_model_uri.txt` after a successful run and is used by the API for serving.
- For custom model selection, change the contents of `latest_model_uri.txt` to match the desired run/model.
- For a complete directory listing or further scripts, see the full [GitHub repository](https://github.com/Alex6606/003-Beta/tree/main/).

---

## References

- [MLflow Documentation](https://mlflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://streamlit.io/)
