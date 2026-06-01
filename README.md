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

---

## Backend

The `backend/` directory contains a **FastAPI** REST API that wraps the trained models and exposes them for the RightAid dashboard.

### Directory layout

```
backend/
├── main.py              # FastAPI app + all route handlers
├── auth.py              # JWT authentication helpers
├── config.py            # Pydantic-settings config (reads .env)
├── models.py            # Pydantic request/response schemas
├── session_store.py     # In-memory session store (generated datasets)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Production container (built from repo root)
└── services/
    ├── data_generator.py  # Synthetic household data generation
    ├── predictor.py       # Model inference + PMT benchmark comparison
    ├── shap_service.py    # Per-record SHAP explanation
    └── policy_brief.py    # Azure OpenAI policy brief generation
```

### Environment configuration

Copy `.env.example` to `.env` inside `backend/` and fill in the values:

```bash
cp backend/.env.example backend/.env
```

| Variable | Description |
|---|---|
| `SECRET_KEY` | JWT signing secret — generate with `openssl rand -hex 32` |
| `RIGHTAID_USERS_JSON` | JSON array of user objects `[{"email":…,"password":…,"name":…,"role":…}]` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Model deployment name (default: `gpt-4o`) |
| `PROVINCE_CONFIG_PATH` | Path to `province_master_config.json` (default: `/app/province_master_config.json`) |
| `MODEL_ELIGIBILITY_PATH` | Path to eligibility model pkl (default: `/app/models/xgboost_eligibility_v3.pkl`) |
| `MODEL_ANOMALY_PATH` | Path to anomaly model pkl (default: `/app/models/xgboost_anomaly_v2.pkl`) |

### Run locally (development)

The backend must be started from the **repo root** so that it can find the model files and `province_master_config.json`.

```bash
# 1. Create and activate a virtual environment (if not already done)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install backend dependencies
pip install -r backend/requirements.txt

# 3. Configure environment
cp backend/.env.example backend/.env
# Edit backend/.env — set SECRET_KEY, RIGHTAID_USERS_JSON, and Azure OpenAI vars

# 4. Override the default Docker paths for local dev
export PROVINCE_CONFIG_PATH=$(pwd)/province_master_config.json
export MODEL_ELIGIBILITY_PATH=$(pwd)/output/xgboost_eligibility_v3.pkl
export MODEL_ANOMALY_PATH=$(pwd)/output/xgboost_anomaly_v2.pkl

# 5. Start the server
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.  
Interactive docs (Swagger UI): `http://localhost:8000/docs`

> **Note:** Run Step 3 of the [Reproduce](#reproduce) section first to generate the model pickle files if they don't exist yet.

### Run with Docker

The Dockerfile is designed to be built from the **repo root** so it can copy model files and shared scripts into the image.

```bash
# Build
docker build -t rightaid-api .

# Run (pass secrets via environment variables)
docker run -p 8000:8000 \
  -e SECRET_KEY="$(openssl rand -hex 32)" \
  -e RIGHTAID_USERS_JSON='[{"email":"guest@rightaid.id","password":"guest123","name":"Guest Analyst","role":"guest"}]' \
  -e AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/" \
  -e AZURE_OPENAI_KEY="your-key" \
  -e AZURE_OPENAI_DEPLOYMENT="gpt-4o" \
  rightaid-api
```

### API endpoints

All endpoints except `/health` and `POST /api/auth/login` require a Bearer JWT token in the `Authorization` header.

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/auth/login` | Authenticate — returns JWT token |
| `POST` | `/api/auth/logout` | Logout (client-side token discard) |
| `GET` | `/api/stats/national` | National poverty statistics |
| `GET` | `/api/stats/trend` | Monthly exclusion/inclusion error trend |
| `GET` | `/api/model/comparison` | PMT vs ML model metrics side-by-side |
| `GET` | `/api/provinces` | List all 38 provinces with derived stats |
| `POST` | `/api/generate` | Generate a synthetic household dataset for a province/scenario |
| `GET` | `/api/data/{session_id}` | Paginated view of a generated dataset (with ML scores if predicted) |
| `GET` | `/api/data/{session_id}/export` | Export dataset as CSV |
| `POST` | `/api/predict/{session_id}` | Run ML inference on a generated dataset |
| `GET` | `/api/shap/{session_id}/{record_id}` | SHAP feature-importance explanation for a single record |
| `POST` | `/api/policy-brief` | Generate an AI policy brief via Azure OpenAI |

#### Example: login and generate data

```bash
# 1. Authenticate
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"guest@rightaid.id","password":"guest123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")

# 2. Generate 1000 households in DKI Jakarta, normal scenario
curl -s -X POST http://localhost:8000/api/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"province_id":"DKI Jakarta","scenario":"normal","anomaly_pct":0.05,"n":1000}'
```
