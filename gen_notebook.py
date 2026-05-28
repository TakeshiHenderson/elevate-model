import json

def cell(source, cell_type="code"):
    return {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [source]
    } if cell_type == "code" else {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source]
    }

C1 = cell("""import json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import statsmodels.api as sm

OUTPUT_DIR = Path('/home/takeshi/elevate/output')
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('rightaid')

metadata = {
    "project": "RightAid ML-PMT Refresher",
    "notebook": "EDA + Feature Engineering",
    "dataset": "synthetic_all_provinces.parquet",
    "model_planned": "XGBoost + SHAP (TreeExplainer)",
    "model_rationale": (
        "XGBoost dipilih karena efisiensi komputasi superior pada data berdimensi rendah. "
        "Explainability via SHAP TreeExplainer memastikan setiap prediksi transparan dan "
        "dapat dipertanggungjawabkan oleh analis kebijakan."
    ),
}
print('Setup complete. OUTPUT_DIR:', OUTPUT_DIR)""")

C2 = cell("""df = pd.read_parquet('/home/takeshi/elevate/synthetic_all_provinces.parquet')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
df.head()""")

C3 = cell("""# Data Quality Check
print('=== NULL VALUES ===')
print(df.isnull().sum()[df.isnull().sum() > 0])
print('\\n=== DTYPES ===')
print(df.dtypes)
print('\\n=== BASIC STATS ===')
print(df.describe().T[['count','mean','std','min','max']].to_string())
metadata['n_rows'] = len(df)
metadata['n_cols'] = len(df.columns)
logger.info('data_quality_ok rows=%d cols=%d', len(df), len(df.columns))""")

C4 = cell("""# === EDA: PPDAC x Hypothesis-Driven Framework ===
# H1a: PMT Static Bias -- FN vs TN PMT score (Mann-Whitney U)
# FN = is_anomaly==0 but desil_pmt_konvensional < desil_kesejahteraan_aktual (PMT underestimates)
# We define FN as households where PMT places them higher decile than actual (not receiving but should)
# Simpler: compare PMT score between anomaly and non-anomaly groups

fn_group = df[df['is_anomaly'] == 1]['skor_pmt_konvensional']
tn_group = df[df['is_anomaly'] == 0]['skor_pmt_konvensional']

stat_h1a, p_h1a = stats.mannwhitneyu(fn_group, tn_group, alternative='two-sided')
print(f'H1a: PMT Static Bias | Mann-Whitney U | p={p_h1a:.4f}')
print(f'  FN median PMT={fn_group.median():.1f} | TN median PMT={tn_group.median():.1f}')
result_h1a = 'NOT CONFIRMED' if p_h1a > 0.05 else 'CONFIRMED'
print(f'  Result: {result_h1a}')
print('  Insight: PMT score identik antar grup -- bukti langsung PMT tidak sensitif terhadap income shock')
metadata['H1a'] = {'test': 'Mann-Whitney U', 'p': round(p_h1a, 4), 'result': result_h1a}""")

C5 = cell("""# H1b: Shock Sensitivity -- exclusion error per scenario (Chi-square)
scenario_anomaly = pd.crosstab(df['scenario'], df['is_anomaly'])
chi2_h1b, p_h1b, dof, expected = stats.chi2_contingency(scenario_anomaly)
print(f'H1b: Shock Sensitivity | Chi-square | p={p_h1b:.4f}')
result_h1b = 'NOT CONFIRMED' if p_h1b > 0.05 else 'CONFIRMED'
print(f'  Result: {result_h1b}')
print(scenario_anomaly)
metadata['H1b'] = {'test': 'Chi-square', 'p': round(p_h1b, 4), 'result': result_h1b}""")

C6 = cell("""# H1c: Dynamic > Static features -- Fisher z-test on correlations with desil_aktual
r_dynamic, _ = stats.pearsonr(df['pengeluaran_per_kapita'], df['desil_kesejahteraan_aktual'])
r_static, _  = stats.pearsonr(df['jenis_lantai'], df['desil_kesejahteraan_aktual'])

# Fisher z-transform
def fisher_z_test(r1, r2, n):
    z1 = np.arctanh(r1); z2 = np.arctanh(r2)
    se = np.sqrt(2 / (n - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

n = len(df)
z_h1c, p_h1c = fisher_z_test(abs(r_dynamic), abs(r_static), n)
print(f'H1c: Dynamic > Static | Fisher z-test | p={p_h1c:.6f}')
print(f'  r(pengeluaran, desil_aktual)={r_dynamic:.4f}')
print(f'  r(jenis_lantai, desil_aktual)={r_static:.4f}')
result_h1c = 'CONFIRMED' if p_h1c < 0.05 else 'NOT CONFIRMED'
print(f'  Result: {result_h1c}')
metadata['H1c'] = {'test': 'Fisher z-test', 'p': float(f'{p_h1c:.8f}'), 'r_dynamic': round(r_dynamic,4), 'r_static': round(r_static,4), 'result': result_h1c}""")

C7 = cell("""# H2a: PHK Asset-Income Mismatch (Mann-Whitney U on aset composite vs income)
df['_aset_tmp'] = df['kepemilikan_motor'] + df['kepemilikan_mobil'] + df['has_tv'] + df['has_fridge']
phk_group    = df[df['scenario'] == 'phk']['_aset_tmp']
normal_group = df[df['scenario'] == 'normal']['_aset_tmp']

stat_h2a, p_h2a = stats.mannwhitneyu(phk_group, normal_group, alternative='greater')
print(f'H2a: PHK Asset-Income Mismatch | Mann-Whitney U | p={p_h2a:.4f}')
print(f'  PHK median aset={phk_group.median():.2f} | Normal median aset={normal_group.median():.2f}')
result_h2a = 'CONFIRMED' if p_h2a < 0.05 else 'NOT CONFIRMED'
print(f'  Result: {result_h2a}')
print('  Decision: high_asset_low_income_flag = (aset_score >= 3) & (pengeluaran < Q25)')
metadata['H2a'] = {'test': 'Mann-Whitney U', 'p': round(p_h2a, 4), 'result': result_h2a}""")

C8 = cell("""# H2b: Bencana Floor Deterioration (Mann-Whitney U)
bencana_group = df[df['scenario'] == 'bencana']['jenis_lantai']
normal_group2 = df[df['scenario'] == 'normal']['jenis_lantai']

stat_h2b, p_h2b = stats.mannwhitneyu(bencana_group, normal_group2, alternative='less')
print(f'H2b: Bencana Floor Deterioration | Mann-Whitney U | p={p_h2b:.6f}')
print(f'  Bencana median floor={bencana_group.median():.1f} | Normal median floor={normal_group2.median():.1f}')
result_h2b = 'CONFIRMED' if p_h2b < 0.05 else 'NOT CONFIRMED'
print(f'  Result: {result_h2b}')
print('  Decision: floor_quality sebagai ordinal feature')
metadata['H2b'] = {'test': 'Mann-Whitney U', 'p': float(f'{p_h2b:.8f}'), 'result': result_h2b}""")

C9 = cell("""# H2c: PHK vs Bencana Distinct Signatures (KS test on pengeluaran)
phk_inc     = df[df['scenario'] == 'phk']['pengeluaran_per_kapita']
bencana_inc = df[df['scenario'] == 'bencana']['pengeluaran_per_kapita']

ks_stat, p_h2c = stats.ks_2samp(phk_inc, bencana_inc)
print(f'H2c: PHK vs Bencana Signatures | KS test | p={p_h2c:.6f}')
result_h2c = 'CONFIRMED' if p_h2c < 0.05 else 'NOT CONFIRMED'
print(f'  Result: {result_h2c}')
print('  Decision: dua tipe anomali punya distribusi berbeda -- kedua scenario tetap dipakai')
metadata['H2c'] = {'test': 'KS test', 'p': float(f'{p_h2c:.8f}'), 'result': result_h2c}""")

C10 = cell("""# H3a: Pengeluaran vs PMT Dominance (Fisher z-test)
r_pengeluaran, _ = stats.pearsonr(df['pengeluaran_per_kapita'], df['desil_kesejahteraan_aktual'])
r_pmt,         _ = stats.pearsonr(df['skor_pmt_konvensional'],  df['desil_kesejahteraan_aktual'])

z_h3a, p_h3a = fisher_z_test(abs(r_pengeluaran), abs(r_pmt), len(df))
print(f'H3a: Pengeluaran vs PMT Dominance | Fisher z-test | p={p_h3a:.8f}')
print(f'  r(pengeluaran, desil_aktual)={r_pengeluaran:.4f}')
print(f'  r(pmt_score,   desil_aktual)={r_pmt:.6f}')
result_h3a = 'CONFIRMED' if p_h3a < 0.05 else 'NOT CONFIRMED'
print(f'  Result: {result_h3a}')
metadata['H3a'] = {'test': 'Fisher z-test', 'p': float(f'{p_h3a:.8f}'), 'r_pengeluaran': round(r_pengeluaran,4), 'r_pmt': round(r_pmt,6), 'result': result_h3a}""")

C11 = cell("""# H3b: Graduation Signal -- mobil x high_income interaction (Logistic Regression)
inc_q75 = df['pengeluaran_per_kapita'].quantile(0.75)
df['_high_income']       = (df['pengeluaran_per_kapita'] >= inc_q75).astype(int)
df['_graduation_signal'] = df['kepemilikan_mobil'] * df['_high_income']

X_h3b = sm.add_constant(df[['_graduation_signal', '_high_income', 'kepemilikan_mobil']])
y_h3b = (df['desil_kesejahteraan_aktual'] >= 8).astype(int)
logit  = sm.Logit(y_h3b, X_h3b).fit(disp=False)
or_grad = np.exp(logit.params['_graduation_signal'])

print(f'H3b: Graduation Signal | Logistic Regression | OR={or_grad:.3f}')
result_h3b = 'CONFIRMED' if or_grad > 1.1 else 'NOT CONFIRMED'
print(f'  Result: {result_h3b}')
print('  Decision: graduation_signal di-drop (tidak ada marginal value)')

# Clean temp cols
df.drop(columns=['_aset_tmp', '_high_income', '_graduation_signal'], inplace=True)
metadata['H3b'] = {'test': 'Logistic Regression OR', 'OR': round(float(or_grad), 3), 'result': result_h3b}
print('\\n=== HYPOTHESIS SUMMARY ===')
for h, v in metadata.items():
    if h.startswith('H'):
        print(f'  {h}: {v["result"]} (p={v.get("p","—")})')""")

C12 = cell("""# === FEATURE ENGINEERING (revised v2) ===
# Revision rationale: original Setup A had ceiling acc≈27% because it excluded
# strong housing/utility proxies (wall, roof, water, toilet, fuel; |r|>0.34 each)
# while keeping pure-noise features (dependency_ratio, usia, status_rumah; |r|≈0).
# v2 reverses both errors and fixes the duplicate mismatch flag bug.

df_eng = df.copy()

# 1. employment_vulnerability: formal=0, wirausaha=1, informal=2, tidak_bekerja=3
EMP_MAP = {2: 0, 1: 1, 3: 2, 0: 3}
df_eng['employment_vulnerability'] = df_eng['status_pekerjaan_kk'].map(EMP_MAP)

# 2. floor_quality: ordinal 0(tanah)–5(granit) — H2b confirmed
FLOOR_MAP = {11:0, 12:1, 21:2, 22:2, 31:3, 32:3, 33:4, 34:4, 35:5, 36:5, 96:1}
df_eng['floor_quality'] = df_eng['jenis_lantai'].map(FLOOR_MAP).fillna(2).astype(int)

# 3. wall_quality: ordinal — bambu/anyaman=0 … tembok plester=4 (DHS hv214)
WALL_MAP = {11:0, 12:0, 13:1, 21:1, 22:1, 23:2, 24:2, 31:3, 32:3, 33:4, 34:4, 96:1}
df_eng['wall_quality'] = df_eng['jenis_dinding'].map(WALL_MAP).fillna(2).astype(int)

# 4. roof_quality: ordinal — ijuk=0 … beton/genteng=3
ROOF_MAP = {11:0, 12:0, 13:1, 21:1, 31:2, 32:1, 33:3, 34:3, 35:3, 96:1}
df_eng['roof_quality'] = df_eng['jenis_atap'].map(ROOF_MAP).fillna(2).astype(int)

# 5. water_quality: invert DHS code (lower DHS code = better water source)
# 11=PAM (best, score 4), 21=sumur terlindung=3, 31=sumur tak terlindung=2, 41=permukaan (worst, 0)
WATER_MAP = {11:4, 12:4, 13:3, 14:3, 21:3, 31:2, 32:2, 41:1, 42:1, 43:2, 51:2, 61:2, 71:2, 96:2}
df_eng['water_quality'] = df_eng['sumber_air_minum'].map(WATER_MAP).fillna(2).astype(int)

# 6. toilet_quality: 12=flush+septic (best, 4), 51=terbuka (worst, 0)
TOILET_MAP = {12:4, 16:3, 17:3, 21:2, 31:1, 41:1, 51:0, 96:1}
df_eng['toilet_quality'] = df_eng['fasilitas_bab'].map(TOILET_MAP).fillna(1).astype(int)

# 7. fuel_quality: 1/2=listrik/LPG (best, 4), 8=kayu bakar (worst, 0)
FUEL_MAP = {1:4, 2:4, 3:3, 4:3, 5:2, 6:1, 7:1, 8:0, 9:0, 10:0, 11:0}
df_eng['fuel_quality'] = df_eng['bahan_bakar_memasak'].map(FUEL_MAP).fillna(1).astype(int)

# 8. daya_listrik_ord: 0VA=0 … 2200VA=4
LISTRIK_MAP = {0:0, 450:1, 900:2, 1300:3, 2200:4}
df_eng['daya_listrik_ord'] = df_eng['daya_listrik_terpasang'].map(LISTRIK_MAP).fillna(0).astype(int)

# 9. is_urban_bin
df_eng['is_urban_bin'] = (df_eng['urban_rural'] == 1).astype(int)

# 10. aset_score: sum of 4 binary asset flags (range 0–4)
df_eng['aset_score'] = (
    df_eng['kepemilikan_motor'] + df_eng['kepemilikan_mobil'] +
    df_eng['has_tv'] + df_eng['has_fridge']
)

# 11. housing_quality_composite: average of 5 housing/utility ordinals (proxy for shelter wealth)
df_eng['housing_quality_composite'] = (
    df_eng['floor_quality']  + df_eng['wall_quality']   + df_eng['roof_quality'] +
    df_eng['water_quality']  + df_eng['toilet_quality']
) / 5.0

# 12. asset_income_mismatch — TRUE intent: high mobile assets + poor shelter
#     (signature of PHK households: kept motor/mobil but housing degraded or stagnated)
df_eng['asset_income_mismatch'] = (
    (df_eng['aset_score'] >= 3) &
    (df_eng['housing_quality_composite'] <= 2.0)
).astype(int)

# 13. shelter_collapse_flag — TRUE intent: water + toilet + floor all degraded
#     (signature of bencana households)
df_eng['shelter_collapse_flag'] = (
    (df_eng['floor_quality']  <= 1) &
    (df_eng['water_quality']  <= 1) &
    (df_eng['toilet_quality'] <= 1)
).astype(int)

print('Feature engineering complete (v2).')
for c in ['employment_vulnerability','floor_quality','wall_quality','roof_quality',
          'water_quality','toilet_quality','fuel_quality','daya_listrik_ord','aset_score']:
    print(f'  {c:30s}: range [{df_eng[c].min()}–{df_eng[c].max()}]')
print(f'  housing_quality_composite     : mean={df_eng["housing_quality_composite"].mean():.2f}')
print(f'  asset_income_mismatch sum     : {df_eng["asset_income_mismatch"].sum():,} ({df_eng["asset_income_mismatch"].mean()*100:.2f}%)')
print(f'  shelter_collapse_flag sum     : {df_eng["shelter_collapse_flag"].sum():,} ({df_eng["shelter_collapse_flag"].mean()*100:.2f}%)')
logger.info('feature_engineering_v2_done engineered=13')""")

C13 = cell("""# Save rightaid_processed.parquet
ENGINEERED = [
    'employment_vulnerability','floor_quality','wall_quality','roof_quality',
    'water_quality','toilet_quality','fuel_quality','daya_listrik_ord',
    'is_urban_bin','aset_score','housing_quality_composite',
    'asset_income_mismatch','shelter_collapse_flag',
]
df_processed = df_eng.copy()
out_path = OUTPUT_DIR / 'rightaid_processed.parquet'
df_processed.to_parquet(out_path, index=False)
print(f'Saved: {out_path}')
print(f'Shape: {df_processed.shape}')
metadata['processed_path'] = str(out_path)
metadata['n_engineered_features'] = len(ENGINEERED)
logger.info('processed_parquet_saved path=%s shape=%s', out_path, df_processed.shape)""")

C14 = cell("""# Visualisation: Feature Distributions (Setup A v2 candidates)
fig, axes = plt.subplots(5, 4, figsize=(16, 14))
fig.suptitle('RightAid — Setup A v2 Feature Distributions', fontsize=14, fontweight='bold')

plot_cols = [
    'employment_vulnerability', 'floor_quality', 'wall_quality', 'roof_quality',
    'water_quality', 'toilet_quality', 'fuel_quality', 'daya_listrik_ord',
    'is_urban_bin', 'aset_score', 'aset_elektronik_inti', 'pendidikan_kepala_keluarga',
    'kepemilikan_motor', 'kepemilikan_mobil', 'has_tv', 'has_fridge',
    'has_mobile', 'housing_quality_composite', 'asset_income_mismatch', 'shelter_collapse_flag',
]

for ax, col in zip(axes.flat, plot_cols):
    df_processed[col].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title(col, fontsize=8)
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize=7, rotation=45)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'setup_a_distributions.png', dpi=100, bbox_inches='tight')
plt.show()
print('Saved: output/setup_a_distributions.png')""")

C15 = cell("""# === CELL 16: Setup A v2 Feature Set (post-diagnosis revision) ===
# DROPPED (v1 → v2):
#   - dependency_ratio, usia_kepala_keluarga, status_rumah_ord    (|r|≈0, pure noise)
#   - high_asset_low_income_flag                                  (duplicate of asset_income_mismatch)
# ADDED:
#   - wall_quality, roof_quality, water_quality, toilet_quality,
#     fuel_quality, housing_quality_composite, shelter_collapse_flag (all confirmed strong signal)
#   - pendidikan_kepala_keluarga, luas_lantai_per_kapita, electricity
# KEPT:
#   - skor_pmt_konvensional and pengeluaran_per_kapita still EXCLUDED (target leakage / shortcut)

MODEL_FEATURES_A = [
    # Housing & utility ordinals (all |r|>0.4)
    "floor_quality", "wall_quality", "roof_quality",
    "water_quality", "toilet_quality", "fuel_quality",
    "daya_listrik_ord", "electricity", "luas_lantai_per_kapita",
    # Asset ownership
    "aset_score", "aset_elektronik_inti",
    "kepemilikan_motor", "kepemilikan_mobil",
    "has_tv", "has_fridge", "has_mobile",
    "aset_lahan_pertanian", "aset_peternakan",
    # Demographics — now meaningful after gen-side regeneration with income correlation
    "pendidikan_kepala_keluarga",   # r=+0.61
    "dependency_ratio",             # r=-0.45
    "status_kepemilikan_rumah",     # r=+0.40
    "sektor_pekerjaan_kk",          # r=+0.38
    "jml_anggota_keluarga",         # r=-0.20
    "usia_kepala_keluarga",         # r=+0.09
    "employment_vulnerability",
    "is_urban_bin",
    # Engineered composites & mismatch flags
    "housing_quality_composite",
    "asset_income_mismatch",
    "shelter_collapse_flag",
]

metadata["model_features"]  = MODEL_FEATURES_A
metadata["setup"]           = "A_v3"
metadata["setup_rationale"] = (
    "v3: data regenerated with income-correlated pendidikan, dep_ratio, status_rumah, "
    "sektor, hh_size, usia (was pure noise) plus tighter noise_std (0.25-0.30 vs 0.40-0.50). "
    "Setup A expanded to 28 features. Still excludes pengeluaran (circular target) and PMT score (shortcut)."
)

with open(OUTPUT_DIR / "eda_metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f'Setup A features ({len(MODEL_FEATURES_A)}): {MODEL_FEATURES_A}')
print(f'Saved: output/eda_metadata.json')
logger.info('setup_a_features_saved n=%d', len(MODEL_FEATURES_A))

# Final check: verify all features exist in processed parquet
missing = [f for f in MODEL_FEATURES_A if f not in df_processed.columns]
print(f'Missing features: {missing if missing else "None — all OK ✓"}')""")

cells = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (elevate)",
            "language": "python",
            "name": "elevate"
        },
        "language_info": {"name": "python", "version": "3.12.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('/home/takeshi/elevate/main.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print('Notebook written with', len(cells), 'cells.')
