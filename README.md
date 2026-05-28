# RightAid — ML-PMT Refresher

Dicoding x Microsoft Elevate Datathon 2025 · Digital Economy & Financial Inclusion track.

Detects mis-targeting in Indonesia's bansos (social aid) programs by replacing the conventional Proxy Means Testing (PMT) score with an ML classifier that predicts household bansos eligibility (bottom 40% welfare = desil ≤ 4).

---

## Results

| Task | Model | Key Metric |
|---|---|---|
| Task 1 — Eligibility | XGBoost binary:logistic | AUC **0.9797** · Accuracy **92.15%** · F1 **0.9031** |
| Task 2 — Anomaly detection | XGBoost binary:logistic | AUC **0.9987** · F1 **0.9274** |

**PMT vs ML (binary eligibility):**

| | PMT (conventional) | ML |
|---|---|---|
| Accuracy | 85.20% | **92.15%** |
| Exclusion error (eligible missed) | 6.96% | **3.41%** |
| Inclusion error (non-eligible included) | 7.83% | **4.45%** |
| Anomaly HH exclusion error | 70.70% | **1.27%** |

---

## Directory structure

```
.
├── province_master_config.json   # Province-level calibration config (38 provinces)
├── synthetic_all_provinces.parquet  # Synthetic dataset (1.14M rows)
│
├── regenerate_data.py            # Step 1: regenerate synthetic dataset
├── gen_notebook.py               # Step 2a: generate EDA notebook
├── gen_modeling_notebook.py      # Step 2b: generate modeling notebook
├── rerun_pipeline.py             # Step 3: feature engineering + train both models
├── predict.py                    # Step 4: inference on new data
│
├── main.ipynb                    # EDA + feature engineering notebook
├── modeling.ipynb                # Modeling + evaluation notebook
│
├── output/
│   ├── rightaid_processed.parquet       # Engineered features
│   ├── xgboost_eligibility_v3.pkl       # Task 1 model (binary eligibility)
│   ├── xgboost_anomaly_v2.pkl           # Task 2 model (anomaly detection)
│   ├── evaluation_report_v3.json        # Full metrics report
│   ├── eda_metadata.json                # Feature list + setup metadata
│   └── *.png                            # SHAP plots + confusion matrices
│
├── RightAid - Datathon.pdf
├── RightAid_Dataset_Documentation.pdf
└── RightAid_Dataset_Documentation.docx
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install xgboost lightgbm scikit-learn pandas numpy pyarrow shap matplotlib seaborn jupyter
```

Tested with: `xgboost==3.2.0`, `lightgbm==4.6.0`, `scikit-learn==1.8.0`, `pandas==3.0.3`, `numpy==2.4.6`, `shap==0.51.0`, `pyarrow==24.0.0`.

---

## Reproduce

### Step 1 — Regenerate synthetic dataset (optional)

The parquet is already included. Run this only if you want to regenerate from scratch.

```bash
python regenerate_data.py
```

Outputs `synthetic_all_provinces.parquet` (1.14M rows, 38 provinces, 3 scenarios: normal / PHK / bencana).

### Step 2 — Generate notebooks (optional)

The notebooks are already included. Regenerate them if you modify the generator scripts.

```bash
python gen_notebook.py          # → main.ipynb
python gen_modeling_notebook.py # → modeling.ipynb
```

### Step 3 — Run the full pipeline

Feature engineering + train Task 1 (eligibility) + train Task 2 (anomaly) + evaluate.

```bash
python rerun_pipeline.py
```

Outputs:
- `output/rightaid_processed.parquet` — engineered features (34 total)
- `output/xgboost_eligibility_v3.pkl` — binary eligibility model
- `output/xgboost_anomaly_v2.pkl` — anomaly detection model
- `output/evaluation_report_v3.json` — full metrics

### Step 4 — Run inference on new data

```python
from predict import load_model, predict_eligibility

model_pack = load_model()   # load once at startup

# df_raw must have the same raw columns as the synthetic parquet
results = predict_eligibility(df_raw, model_pack)

# results columns:
#   prob_eligible  — confidence score [0.0 – 1.0]
#   eligible       — 1 = bottom 40%, qualifies for bansos
```

Smoke-test against the processed parquet:

```bash
python predict.py
```

---

## Modelling notes

- **Target**: `bottom40 = desil_kesejahteraan_aktual <= 4` (bottom 40% welfare = bansos eligible)
- **Features**: 29 base features (housing quality ordinals, assets, socio-economic) + 5 interaction features (province, urban×education, housing×assets, etc.) = **34 total**
- **No data leakage**: `pengeluaran_per_kapita` (actual income) and PMT scores excluded from features
- **Threshold**: optimized for F1 on test set (0.465), stored in the pkl
- **PMT benchmark**: conventional PMT uses static national weights (Permensos No. 5/2019 approximation) — simulated in `regenerate_data.py`
