"""Loads both models once at startup; runs eligibility + anomaly inference."""
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import settings

# ── Feature engineering (mirrors rerun_pipeline.py) ──────────────────────────
EMP_MAP     = {2: 0, 1: 1, 3: 2, 0: 3}
FLOOR_MAP   = {11:0,12:1,21:2,22:2,31:3,32:3,33:4,34:4,35:5,36:5,96:1}
WALL_MAP    = {11:0,12:0,13:1,21:1,22:1,23:2,24:2,31:3,32:3,33:4,34:4,96:1}
ROOF_MAP    = {11:0,12:0,13:1,21:1,31:2,32:1,33:3,34:3,35:3,96:1}
WATER_MAP   = {11:4,12:4,13:3,14:3,21:3,31:2,32:2,41:1,42:1,43:2,51:2,61:2,71:2,96:2}
TOILET_MAP  = {12:4,16:3,17:3,21:2,31:1,41:1,51:0,96:1}
FUEL_MAP    = {1:4,2:4,3:3,4:3,5:2,6:1,7:1,8:0,9:0,10:0,11:0}
LISTRIK_MAP = {0:0,450:1,900:2,1300:3,2200:4}


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    e = df.copy()
    e['employment_vulnerability'] = e['status_pekerjaan_kk'].map(EMP_MAP).fillna(2).astype(int)
    e['floor_quality']   = e['jenis_lantai'].map(FLOOR_MAP).fillna(2).astype(int)
    e['wall_quality']    = e['jenis_dinding'].map(WALL_MAP).fillna(2).astype(int)
    e['roof_quality']    = e['jenis_atap'].map(ROOF_MAP).fillna(2).astype(int)
    e['water_quality']   = e['sumber_air_minum'].map(WATER_MAP).fillna(2).astype(int)
    e['toilet_quality']  = e['fasilitas_bab'].map(TOILET_MAP).fillna(1).astype(int)
    e['fuel_quality']    = e['bahan_bakar_memasak'].map(FUEL_MAP).fillna(1).astype(int)
    e['daya_listrik_ord']= e['daya_listrik_terpasang'].map(LISTRIK_MAP).fillna(0).astype(int)
    e['is_urban_bin']    = (e['urban_rural'] == 1).astype(int)
    e['aset_score']      = (e['kepemilikan_motor'] + e['kepemilikan_mobil']
                            + e['has_tv'] + e['has_fridge'])
    e['housing_quality_composite'] = (
        e['floor_quality'] + e['wall_quality'] + e['roof_quality']
        + e['water_quality'] + e['toilet_quality']
    ) / 5.0
    e['asset_income_mismatch'] = (
        (e['aset_score'] >= 3) & (e['housing_quality_composite'] <= 2.0)
    ).astype(int)
    e['shelter_collapse_flag'] = (
        (e['floor_quality'] <= 1) & (e['water_quality'] <= 1) & (e['toilet_quality'] <= 1)
    ).astype(int)
    e['province_id']        = e['province'].astype('category').cat.codes.astype('int32')
    e['urban_x_pendidikan'] = e['is_urban_bin'] * e['pendidikan_kepala_keluarga']
    e['urban_x_aset_score'] = e['is_urban_bin'] * e['aset_score']
    e['housing_x_assets']   = e['housing_quality_composite'] * e['aset_score']
    e['edu_x_listrik']      = e['pendidikan_kepala_keluarga'] * e['daya_listrik_ord']
    e['bottom40']           = (e['desil_kesejahteraan_aktual'] <= 4).astype(int)
    return e


def _cm(y_true, y_pred) -> dict:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _metrics(y_true, y_pred, y_prob=None) -> dict:
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    excl = fn / (fn + tp + 1e-9) * 100
    incl = fp / (fp + tn + 1e-9) * 100
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    auc  = float(roc_auc_score(y_true, y_prob)) if y_prob is not None else 0.0
    return {"exclusionErr": round(excl, 2), "inclusionErr": round(incl, 2),
            "f1": round(f1, 4), "auc": round(auc, 4)}


# ── Load models once ──────────────────────────────────────────────────────────
def _load():
    with open(settings.MODEL_ELIGIBILITY_PATH, 'rb') as f:
        elig_pack = pickle.load(f)
    with open(settings.MODEL_ANOMALY_PATH, 'rb') as f:
        anom_pack = pickle.load(f)
    return elig_pack, anom_pack


_elig_pack, _anom_pack = _load()
FEATURES         = _elig_pack['features']
_model_elig      = _elig_pack['model']
_threshold_elig  = _elig_pack['threshold']
_model_anom      = _anom_pack['model']


def run(df: pd.DataFrame) -> dict:
    """Run both models. Returns full predictions dict stored in session."""
    eng = _engineer(df)
    X   = eng[FEATURES]

    prob_elig = _model_elig.predict_proba(X)[:, 1]
    pred_elig = (prob_elig >= _threshold_elig).astype(int)

    prob_anom = _model_anom.predict_proba(X)[:, 1]
    pred_anom = (prob_anom >= 0.5).astype(int)

    y_actual  = eng['bottom40'].values
    pmt_pred  = (eng['desil_pmt_konvensional'].values <= 4).astype(int)

    cm_ml  = _cm(y_actual, pred_elig)
    cm_pmt = _cm(y_actual, pmt_pred)
    m_ml   = _metrics(y_actual, pred_elig, prob_elig)
    m_pmt  = _metrics(y_actual, pmt_pred)

    # Decile distributions (1–10)
    actual_dist = [int((eng['desil_kesejahteraan_aktual'] == d).sum()) for d in range(1, 11)]
    pmt_dist    = [int((eng['desil_pmt_konvensional'] == d).sum())    for d in range(1, 11)]
    # ML dist: ML-eligible per actual decile (shows which deciles ML captures)
    ml_dist = [
        int(((eng['desil_kesejahteraan_aktual'] == d) & (pred_elig == 1)).sum())
        for d in range(1, 11)
    ]

    # Mis-targeting candidates: actual eligible but PMT says no, OR anomaly flagged
    eng2 = eng.copy()
    eng2['id']         = df['id'].values
    eng2['kecamatan']  = df['kecamatan'].values
    eng2['prob_elig']  = prob_elig
    eng2['pred_elig']  = pred_elig
    eng2['pred_anom']  = pred_anom
    eng2['pmt_pred']   = pmt_pred

    cand = eng2[
        ((eng2['bottom40'] == 1) & (eng2['pmt_pred'] == 0)) |
        (eng2['is_anomaly'] == 1)
    ].sort_values('prob_elig', ascending=False).head(50)

    candidates = [
        {
            "id": r["id"],
            "kecamatan": r["kecamatan"],
            "actualDecile": int(r["desil_kesejahteraan_aktual"]),
            "pmtDecile": int(r["desil_pmt_konvensional"]),
            "mlEligible": int(r["pred_elig"]),
            "confidence": round(float(r["prob_elig"]), 4),
            "isAnomaly": bool(r["is_anomaly"]),
            "status": str(r.get("anomaly_type", "none")),
        }
        for _, r in cand.iterrows()
    ]

    return {
        "confusion_matrix": cm_ml,
        "confusion_matrix_pmt": cm_pmt,
        "metrics": m_ml,
        "metrics_pmt": m_pmt,
        "decile_dist": {"actual": actual_dist, "pmt": pmt_dist, "ml": ml_dist},
        "mistargeting_candidates": candidates,
        # stored for SHAP lookup
        "_eng": eng2,
        "_prob_elig": prob_elig,
        "_pred_elig": pred_elig,
    }
