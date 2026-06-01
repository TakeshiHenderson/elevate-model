"""Per-record SHAP explanations. Explainer is built once from the loaded model."""
import numpy as np
import pandas as pd
import shap

from services.predictor import FEATURES, _model_elig

_explainer = shap.TreeExplainer(_model_elig)

FEATURE_LABELS = {
    "floor_quality": "Kualitas Lantai",
    "wall_quality": "Kualitas Dinding",
    "roof_quality": "Kualitas Atap",
    "water_quality": "Akses Air Bersih",
    "toilet_quality": "Fasilitas Sanitasi",
    "fuel_quality": "Bahan Bakar Memasak",
    "daya_listrik_ord": "Daya Listrik",
    "electricity": "Akses Listrik",
    "luas_lantai_per_kapita": "Luas Lantai/Kapita",
    "aset_score": "Skor Aset",
    "aset_elektronik_inti": "Aset Elektronik",
    "kepemilikan_motor": "Kepemilikan Motor",
    "kepemilikan_mobil": "Kepemilikan Mobil",
    "has_tv": "Memiliki TV",
    "has_fridge": "Memiliki Kulkas",
    "has_mobile": "Memiliki HP",
    "aset_lahan_pertanian": "Lahan Pertanian",
    "aset_peternakan": "Aset Peternakan",
    "pendidikan_kepala_keluarga": "Pendidikan KK",
    "dependency_ratio": "Rasio Ketergantungan",
    "status_kepemilikan_rumah": "Status Rumah",
    "sektor_pekerjaan_kk": "Sektor Pekerjaan",
    "jml_anggota_keluarga": "Jumlah Anggota",
    "employment_vulnerability": "Kerentanan Kerja",
    "usia_kepala_keluarga": "Usia KK",
    "is_urban_bin": "Area Urban",
    "housing_quality_composite": "Komposit Hunian",
    "asset_income_mismatch": "Mismatch Aset-Pendapatan",
    "shelter_collapse_flag": "Flag Hunian Kritis",
    "province_id": "Provinsi",
    "urban_x_pendidikan": "Urban × Pendidikan",
    "urban_x_aset_score": "Urban × Aset",
    "housing_x_assets": "Hunian × Aset",
    "edu_x_listrik": "Pendidikan × Listrik",
}


def explain(record_id: str, predictions: dict) -> dict:
    eng: pd.DataFrame = predictions["_eng"]
    row = eng[eng["id"] == record_id]
    if row.empty:
        raise ValueError(f"Record {record_id} not found in session")

    X_row = row[FEATURES]
    shap_vals = _explainer.shap_values(X_row)

    # binary:logistic returns 1D or (1, n_features) for a single row
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
    elif shap_vals.ndim == 2:
        sv = shap_vals[0]
    else:
        sv = shap_vals

    features = [
        {"feature": FEATURE_LABELS.get(f, f), "value": round(float(v), 4)}
        for f, v in sorted(zip(FEATURES, sv), key=lambda x: abs(x[1]), reverse=True)
    ]

    idx = row.index[0]
    return {
        "record_id": record_id,
        "features": features,
        "prediction": round(float(predictions["_prob_elig"][eng.index.get_loc(idx)]), 4),
        "eligible": int(predictions["_pred_elig"][eng.index.get_loc(idx)]),
    }
