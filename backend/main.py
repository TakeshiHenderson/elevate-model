import io
import math
import json
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from auth import authenticate_user, create_access_token, get_current_user
from config import settings
from models import (
    GenerateRequest, GenerateResponse,
    LoginRequest, LoginResponse,
    PolicyBriefRequest, PolicyBriefResponse,
)
import session_store
from services import data_generator, predictor, shap_service, policy_brief

app = FastAPI(title="RightAid API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CurrentUser = Annotated[dict, Depends(get_current_user)]


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}


# ── Auth ──────────────────────────────────────────────────────────────────────
@app.post("/api/auth/login", response_model=LoginResponse)
def login(body: LoginRequest):
    user = authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"token": create_access_token(user), "user": {k: v for k, v in user.items() if k != "password"}}


@app.post("/api/auth/logout")
def logout(_: CurrentUser):
    return {"ok": True}


# ── Static stats ──────────────────────────────────────────────────────────────
@app.get("/api/stats/national")
def national_stats(_: CurrentUser):
    return {
        "poorPopulation": "24,06 Juta",
        "povertyRate": "8,57%",
        "giniCoef": "0,381",
        "socialBudget": "Rp496,8 T",
        "exclusionError": "46%",
        "inclusionError": "23%",
        "dtsenProgress": 78,
    }


@app.get("/api/stats/trend")
def trend_stats(_: CurrentUser):
    return [
        {"month": "Mar 2024", "exclusion": 32.1, "inclusion": 24.3},
        {"month": "Apr 2024", "exclusion": 31.4, "inclusion": 23.8},
        {"month": "Mei 2024", "exclusion": 30.8, "inclusion": 23.1},
        {"month": "Jun 2024", "exclusion": 29.5, "inclusion": 22.6},
        {"month": "Jul 2024", "exclusion": 28.9, "inclusion": 22.0},
        {"month": "Agu 2024", "exclusion": 27.3, "inclusion": 21.4},
        {"month": "Sep 2024", "exclusion": 26.7, "inclusion": 20.9},
        {"month": "Okt 2024", "exclusion": 25.1, "inclusion": 20.2},
        {"month": "Nov 2024", "exclusion": 23.8, "inclusion": 19.7},
        {"month": "Des 2024", "exclusion": 22.4, "inclusion": 19.1},
    ]


@app.get("/api/model/comparison")
def model_comparison(_: CurrentUser):
    return {
        "pmt": {"f1": 0.82, "auc": 0.85, "exclusionErr": 6.96, "inclusionErr": 7.83},
        "ml":  {"f1": 0.90, "auc": 0.98, "exclusionErr": 3.41, "inclusionErr": 4.45},
    }


# Official national poverty line (Garis Kemiskinan), BPS Sept 2024 — Rp/capita/month.
POVERTY_LINE_RP = 595_242


@app.get("/api/provinces")
def list_provinces(_: CurrentUser):
    import math as _m
    from scipy.special import erf as _erf

    with open(settings.PROVINCE_CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)

    result = []
    for name, c in cfg.items():
        pk    = c.get("pengeluaran_per_kapita", {})
        mu    = pk.get("lognormal_mu", 14.0)
        sigma = pk.get("lognormal_sigma", 0.5)
        mean_rp = pk.get("mean_rp") or round(_m.exp(mu + sigma ** 2 / 2))

        # P(expenditure < poverty line) for a LogNormal(mu, sigma)
        z = (_m.log(POVERTY_LINE_RP) - mu) / sigma
        poverty_rate = round(0.5 * (1 + _erf(z / _m.sqrt(2))) * 100, 1)
        # Exact Gini of a LogNormal distribution = erf(sigma / 2)
        gini_coef = round(float(_erf(sigma / 2)), 3)

        result.append({
            "id":            name,
            "name":          name,
            "meanExpenditure": int(mean_rp),
            "povertyRate":   poverty_rate,
            "giniCoef":      gini_coef,
            "urbanPct":      round(c.get("urban_pct_dhs", 0.5) * 100, 1),
            "avgHHSize":     c.get("hh_size_mean", 4.0),
            "pctMotorcycle": round(c.get("pct_motorcycle", 0.0) * 100, 1),
            "pctCar":        round(c.get("pct_car", 0.0) * 100, 1),
            "pctElectricity":round(c.get("pct_electricity", 0.0) * 100, 1),
        })
    # Poorest first: lowest mean per-capita expenditure (reliable across all provinces).
    return sorted(result, key=lambda x: x["meanExpenditure"])


# ── Generate ──────────────────────────────────────────────────────────────────
@app.post("/api/generate", response_model=GenerateResponse)
def generate_data(body: GenerateRequest, _: CurrentUser):
    if body.anomaly_pct < 0 or body.anomaly_pct > 0.5:
        raise HTTPException(400, "anomaly_pct must be between 0 and 0.5")
    if body.n < 100 or body.n > 50_000:
        raise HTTPException(400, "n must be between 100 and 50000")

    df = data_generator.generate(body.province_id, body.scenario, body.anomaly_pct, body.n)
    sid = session_store.create(body.province_id, body.scenario, body.anomaly_pct, df)

    preview = data_generator.to_records(df.head(5))
    return {
        "session_id": sid,
        "province_id": body.province_id,
        "scenario": body.scenario,
        "total_records": len(df),
        "preview": preview,
    }


# ── Data ──────────────────────────────────────────────────────────────────────
@app.get("/api/data/{session_id}")
def get_data(session_id: str, _: CurrentUser, page: int = 1, limit: int = 100):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found or expired")

    df   = sess["df"]
    pred = sess.get("predictions")
    total = len(df)
    pages = math.ceil(total / limit)
    start = (page - 1) * limit
    chunk = df.iloc[start: start + limit]

    records = data_generator.to_records(chunk)

    # Merge prediction columns if available
    if pred is not None:
        eng: pd.DataFrame = pred["_eng"]
        for rec in records:
            row_idx = df[df["id"] == rec["id"]].index
            if len(row_idx):
                loc = eng.index.get_loc(row_idx[0])
                rec["mlScore"]   = round(float(pred["_prob_elig"][loc]), 4)
                rec["mlEligible"]= int(pred["_pred_elig"][loc])
                rec["confidence"]= round(float(pred["_prob_elig"][loc]), 4)

    return {"records": records, "total": total, "page": page, "pages": pages}


@app.get("/api/data/{session_id}/export")
def export_csv(session_id: str, _: CurrentUser):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found or expired")

    df   = sess["df"]
    pred = sess.get("predictions")

    export_cols = [
        "id", "kecamatan", "province", "urban_rural",
        "jml_anggota_keluarga", "usia_kepala_keluarga", "gender_kepala_keluarga",
        "pendidikan_kepala_keluarga", "sektor_pekerjaan_kk",
        "pengeluaran_per_kapita", "jenis_lantai", "sumber_air_minum",
        "kepemilikan_motor", "kepemilikan_mobil",
        "skor_pmt_konvensional", "desil_pmt_konvensional",
        "desil_kesejahteraan_aktual", "is_anomaly", "scenario",
    ]
    out = df[export_cols].copy()

    if pred is not None:
        eng: pd.DataFrame = pred["_eng"]
        out["ml_prob_eligible"] = pred["_prob_elig"]
        out["ml_eligible"]      = pred["_pred_elig"]

    buf = io.StringIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    fname = f"rightaid_{sess['province_id'].replace(' ', '_')}_{sess['scenario']}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


# ── Predict ───────────────────────────────────────────────────────────────────
@app.post("/api/predict/{session_id}")
def predict(session_id: str, _: CurrentUser):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found or expired")

    result = predictor.run(sess["df"])
    session_store.set_predictions(session_id, result)

    return {
        "confusion_matrix":     result["confusion_matrix"],
        "confusion_matrix_pmt": result["confusion_matrix_pmt"],
        "metrics":     result["metrics"],
        "metrics_pmt": result["metrics_pmt"],
        "decile_dist": result["decile_dist"],
        "mistargeting_candidates": result["mistargeting_candidates"],
    }


# ── SHAP ──────────────────────────────────────────────────────────────────────
@app.get("/api/shap/{session_id}/{record_id}")
def shap_explain(session_id: str, record_id: str, _: CurrentUser):
    sess = session_store.get(session_id)
    if not sess:
        raise HTTPException(404, "Session not found or expired")
    if sess.get("predictions") is None:
        raise HTTPException(400, "Run /api/predict first")

    try:
        return shap_service.explain(record_id, sess["predictions"])
    except ValueError as e:
        raise HTTPException(404, str(e))


# ── Policy Brief ──────────────────────────────────────────────────────────────
@app.post("/api/policy-brief", response_model=PolicyBriefResponse)
def generate_policy_brief(body: PolicyBriefRequest, _: CurrentUser):
    sess = session_store.get(body.session_id)
    if not sess:
        raise HTTPException(404, "Session not found or expired")
    if sess.get("predictions") is None:
        raise HTTPException(400, "Run /api/predict first")

    pred   = sess["predictions"]
    total  = len(sess["df"])
    elig_n = int(pred["confusion_matrix"]["tp"] + pred["confusion_matrix"]["fn"])

    stats = {
        "total_records":   total,
        "eligible_pct":    elig_n / total * 100,
        "pmt_exclusion_err": pred["metrics_pmt"]["exclusionErr"],
        "ml_exclusion_err":  pred["metrics"]["exclusionErr"],
        "pmt_inclusion_err": pred["metrics_pmt"]["inclusionErr"],
        "ml_inclusion_err":  pred["metrics"]["inclusionErr"],
        "anomaly_pct":    sess["anomaly_pct"],
        "ml_f1":          pred["metrics"]["f1"],
        "ml_auc":         pred["metrics"]["auc"],
        "pmt_f1":         pred["metrics_pmt"]["f1"],
    }

    return policy_brief.generate(body.province_id, body.scenario, stats)
