"""
RightAid inference — load trained model and predict bansos eligibility.

Usage:
    from predict import load_model, predict_eligibility

    model_pack = load_model()
    results = predict_eligibility(df_raw, model_pack)
    # results has columns: eligible (0/1), prob_eligible (float)
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('/home/takeshi/elevate/output')

EMP_MAP     = {2: 0, 1: 1, 3: 2, 0: 3}
FLOOR_MAP   = {11:0, 12:1, 21:2, 22:2, 31:3, 32:3, 33:4, 34:4, 35:5, 36:5, 96:1}
WALL_MAP    = {11:0, 12:0, 13:1, 21:1, 22:1, 23:2, 24:2, 31:3, 32:3, 33:4, 34:4, 96:1}
ROOF_MAP    = {11:0, 12:0, 13:1, 21:1, 31:2, 32:1, 33:3, 34:3, 35:3, 96:1}
WATER_MAP   = {11:4, 12:4, 13:3, 14:3, 21:3, 31:2, 32:2, 41:1, 42:1, 43:2, 51:2, 61:2, 71:2, 96:2}
TOILET_MAP  = {12:4, 16:3, 17:3, 21:2, 31:1, 41:1, 51:0, 96:1}
FUEL_MAP    = {1:4, 2:4, 3:3, 4:3, 5:2, 6:1, 7:1, 8:0, 9:0, 10:0, 11:0}
LISTRIK_MAP = {0:0, 450:1, 900:2, 1300:3, 2200:4}


def load_model(path: str | Path = OUTPUT_DIR / 'xgboost_eligibility_v3.pkl') -> dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


def engineer_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    df['employment_vulnerability'] = df['status_pekerjaan_kk'].map(EMP_MAP).fillna(2).astype(int)
    df['floor_quality']   = df['jenis_lantai'].map(FLOOR_MAP).fillna(2).astype(int)
    df['wall_quality']    = df['jenis_dinding'].map(WALL_MAP).fillna(2).astype(int)
    df['roof_quality']    = df['jenis_atap'].map(ROOF_MAP).fillna(2).astype(int)
    df['water_quality']   = df['sumber_air_minum'].map(WATER_MAP).fillna(2).astype(int)
    df['toilet_quality']  = df['fasilitas_bab'].map(TOILET_MAP).fillna(1).astype(int)
    df['fuel_quality']    = df['bahan_bakar_memasak'].map(FUEL_MAP).fillna(1).astype(int)
    df['daya_listrik_ord']= df['daya_listrik_terpasang'].map(LISTRIK_MAP).fillna(0).astype(int)
    df['is_urban_bin']    = (df['urban_rural'] == 1).astype(int)

    df['aset_score'] = (
        df['kepemilikan_motor'] + df['kepemilikan_mobil']
        + df['has_tv'] + df['has_fridge']
    )
    df['housing_quality_composite'] = (
        df['floor_quality'] + df['wall_quality'] + df['roof_quality']
        + df['water_quality'] + df['toilet_quality']
    ) / 5.0
    df['asset_income_mismatch'] = (
        (df['aset_score'] >= 3) & (df['housing_quality_composite'] <= 2.0)
    ).astype(int)
    df['shelter_collapse_flag'] = (
        (df['floor_quality'] <= 1) &
        (df['water_quality'] <= 1) &
        (df['toilet_quality'] <= 1)
    ).astype(int)

    df['province_id']        = df['province'].astype('category').cat.codes.astype('int32')
    df['urban_x_pendidikan'] = df['is_urban_bin'] * df['pendidikan_kepala_keluarga']
    df['urban_x_aset_score'] = df['is_urban_bin'] * df['aset_score']
    df['housing_x_assets']   = df['housing_quality_composite'] * df['aset_score']
    df['edu_x_listrik']      = df['pendidikan_kepala_keluarga'] * df['daya_listrik_ord']

    return df


def predict_eligibility(df_raw: pd.DataFrame, model_pack: dict) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - prob_eligible : float [0, 1]
      - eligible      : int   0 or 1  (1 = bottom-40%, qualifies for bansos)
    """
    model     = model_pack['model']
    features  = model_pack['features']
    threshold = model_pack['threshold']

    df_eng = engineer_features(df_raw)
    X = df_eng[features]

    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    return pd.DataFrame({'prob_eligible': prob, 'eligible': pred}, index=df_raw.index)


if __name__ == '__main__':
    # Quick smoke-test on the processed parquet
    df = pd.read_parquet(OUTPUT_DIR / 'rightaid_processed.parquet')
    sample = df.sample(5, random_state=0)

    pack    = load_model()
    results = predict_eligibility(sample, pack)

    print('threshold :', pack['threshold'])
    print('features  :', len(pack['features']))
    print()
    print(results)
    if 'bottom40' in sample.columns:
        print('\nGround truth:')
        print(sample['bottom40'].values)
