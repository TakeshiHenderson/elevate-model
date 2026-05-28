"""
Regenerate synthetic_all_provinces.parquet with fixes to the generation-side flaws:
  - dependency_ratio, usia, pendidikan, status_rumah, jml_anggota, status_pekerjaan,
    sektor_pekerjaan now CORRELATE with income rank (was independent → pure noise).
  - noise_std reduced from 0.4-0.5 → 0.25-0.3 for tighter feature-target signal.
  - Floor/wall/roof correlation kept but noise reduced.

Everything else (PMT scoring, anomaly injection, file format) preserved.
"""
import gc, json, logging, math, sys, uuid
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', stream=sys.stdout)
logger = logging.getLogger('rightaid.regen')

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

MASTER_CONFIG_PATH = '/home/takeshi/elevate/province_master_config.json'
OUTPUT_PATH        = Path('/home/takeshi/elevate/synthetic_all_provinces.parquet')

N_HOUSEHOLDS   = 10_000
ANOMALY_RATIOS = {"normal": 0.00, "phk": 0.05, "bencana": 0.10}
MU_URBAN_OFFSET = 0.03
MU_RURAL_OFFSET = -0.18

# Reduced noise — tighter coupling between income rank and proxies.
# Original used 0.4-0.5 which created too much within-decile variance.
NOISE_TIGHT     = 0.25
NOISE_MEDIUM    = 0.30
NOISE_LOOSE     = 0.35   # for things that should retain *some* independence (e.g. gender)

with open(MASTER_CONFIG_PATH, encoding='utf-8') as f:
    MASTER_CONFIG: dict = json.load(f)

# ---------------- PMT (unchanged) ----------------
FLOOR_PMT_SCORE  = {11:0, 12:1, 21:2, 22:2, 31:1, 32:3, 33:4, 34:4, 35:5, 36:3, 96:2}
WATER_PMT_SCORE  = {11:5, 12:5, 13:4, 14:4, 21:3, 31:2, 32:2, 41:1, 42:1, 43:3, 51:4, 61:3, 71:3, 96:2}
TOILET_PMT_SCORE = {12:5, 16:4, 17:4, 21:3, 31:2, 41:1, 51:0, 96:1}
PMT_WEIGHTS = {"floor":0.22, "water":0.15, "toilet":0.15, "electricity":0.10,
               "cooking_fuel":0.08, "motorcycle":0.08, "car":0.10, "tv":0.06, "fridge":0.06}
FUEL_PMT = {1:1.0, 2:1.0, 3:0.9, 4:0.9, 5:0.5, 6:0.4, 7:0.3, 8:0.0, 9:0.0, 10:0.1, 11:0.1}

def compute_pmt_score(df):
    score = pd.Series(0.0, index=df.index)
    score += PMT_WEIGHTS['floor']  * (df['floor_material'].map(FLOOR_PMT_SCORE).fillna(2) / 5.0)
    score += PMT_WEIGHTS['water']  * (df['water_source'].map(WATER_PMT_SCORE).fillna(2)  / 5.0)
    score += PMT_WEIGHTS['toilet'] * (df['toilet_facility'].map(TOILET_PMT_SCORE).fillna(2) / 5.0)
    score += PMT_WEIGHTS['electricity'] * df['electricity'].clip(0, 1)
    score += PMT_WEIGHTS['cooking_fuel'] * df['cooking_fuel'].map(FUEL_PMT).fillna(0.5)
    score += PMT_WEIGHTS['motorcycle'] * df['has_motorcycle'].clip(0, 1)
    score += PMT_WEIGHTS['car']        * df['has_car'].clip(0, 1)
    score += PMT_WEIGHTS['tv']         * df['has_tv'].clip(0, 1)
    score += PMT_WEIGHTS['fridge']     * df['has_fridge'].clip(0, 1)
    return (score * 100).round(2)


def safe_qcut_desil(series, q=10):
    try:
        return pd.qcut(series, q=q, labels=list(range(1, q+1)), duplicates='drop').astype(int)
    except ValueError:
        return series.rank(method='first', pct=True).mul(q).apply(math.ceil).clip(1, q).astype(int)


# ---------------- Correlated samplers (unchanged signature) ----------------
def _sample_correlated_dist(dist, ranks, rng, quality_map=None, noise_std=NOISE_MEDIUM):
    """Sample categorical with quality-rank correlation."""
    if quality_map is not None:
        sorted_keys = sorted(dist.keys(), key=lambda k: (quality_map.get(k, 0), k))
    else:
        sorted_keys = sorted(dist.keys())
    probs = np.array([dist[k] for k in sorted_keys], dtype=float)
    probs /= probs.sum()
    cum = np.cumsum(probs)
    noise = rng.normal(0, noise_std, len(ranks))
    score_pct = pd.Series(ranks + noise).rank(pct=True).values
    assigned = np.zeros(len(ranks), dtype=int)
    for i, c in enumerate(cum):
        m = (score_pct <= c)
        assigned = np.where(m & (assigned == 0), sorted_keys[i], assigned)
    return np.where(assigned == 0, sorted_keys[-1], assigned)


def _sample_correlated_binary(rate, ranks, rng, noise_std=NOISE_MEDIUM):
    n = len(ranks)
    if rate <= 0:  return np.zeros(n, dtype=int)
    if rate >= 1:  return np.ones(n, dtype=int)
    noise = rng.normal(0, noise_std, n)
    score = ranks + noise
    cutoff = np.percentile(score, (1.0 - rate) * 100)
    return (score >= cutoff).astype(int)


def _sample_correlated_ordinal(values, probs, ranks, rng, noise_std=NOISE_MEDIUM, ascending=True):
    """Sample from ordinal categories where order matters w/ income.
    ascending=True: higher rank → larger value (e.g., pendidikan: rank↑ → education↑).
    ascending=False: higher rank → smaller value."""
    values = list(values)
    if not ascending:
        values = values[::-1]
    n = len(ranks)
    cum = np.cumsum(probs)
    noise = rng.normal(0, noise_std, n)
    score_pct = pd.Series(ranks + noise).rank(pct=True).values
    out = np.full(n, values[-1])
    assigned = np.zeros(n, dtype=bool)
    for v, c in zip(values, cum):
        m = (score_pct <= c) & ~assigned
        out[m] = v
        assigned |= m
    return out.astype(int)


def _sample_correlated_continuous(loc, scale, ranks, rng, slope=0.5):
    """Continuous with linear shift along income rank.
       loc + slope * (rank - 0.5) * scale * 2  — rank=0 → loc-scale, rank=1 → loc+scale."""
    base = rng.normal(loc, scale, len(ranks))
    shift = slope * (ranks - 0.5) * scale * 2.0
    return base + shift


# ---------------- Generator ----------------
def generate_households(province, n, scenario, anomaly_pct, rng):
    cfg = MASTER_CONFIG[province]
    is_urban = rng.random(n) < (cfg.get('urban_pct_dhs') or 0.5)
    urban_pct = cfg.get('urban_pct_dhs') or 0.5

    # Income FIRST (drives everything else)
    pen_cfg = cfg['pengeluaran_per_kapita']
    mu_raw  = pen_cfg['lognormal_mu']
    sigma   = pen_cfg['lognormal_sigma']
    dyn_rural = MU_RURAL_OFFSET + 0.05 * (1.0 - urban_pct)
    blending  = urban_pct * math.exp(MU_URBAN_OFFSET) + (1 - urban_pct) * math.exp(dyn_rural)
    mu_base   = mu_raw - math.log(blending)
    mu_arr    = np.where(is_urban, mu_base + MU_URBAN_OFFSET, mu_base + dyn_rural)
    pengeluaran = np.clip(rng.lognormal(mean=mu_arr, sigma=sigma), 200_000, 15_000_000).astype(int)
    ranks = pd.Series(pengeluaran).rank(pct=True).values

    # === FIX 1: pendidikan now correlates with income (was uniform) ===
    # Real-world: pendidikan ↑ correlates strongly with income (textbook poverty indicator)
    pendidikan_kk = _sample_correlated_ordinal(
        values=[0, 1, 2, 3, 4],
        probs=[0.153, 0.345, 0.222, 0.176, 0.104],
        ranks=ranks, rng=rng, noise_std=NOISE_MEDIUM, ascending=True,
    )

    # === FIX 2: dep_ratio now negatively correlates with income (poorer → larger dep) ===
    hh_size_mean = cfg.get('hh_size_mean') or 4.0
    # Tighter family size → richer households tend smaller (Engel curve).
    base_size = rng.poisson(max(hh_size_mean - 1, 1.0), n)
    size_shift = (0.5 - ranks) * 2.0  # rank=0 → +2, rank=1 → -2 (members)
    jml_anggota = np.clip((base_size + size_shift + 1).astype(int), 1, 12)
    # dep_ratio: poorer → more dependants
    dep_base = rng.normal(0.8, 0.4, n)
    dep_ratio = np.clip(dep_base + (0.5 - ranks) * 0.8, 0.0, 3.0).round(2)

    # === FIX 3: usia weakly positive with income (more experience → more income) ===
    usia_base = rng.normal(45, 12, n)
    usia_kk = np.clip((usia_base + (ranks - 0.5) * 4).astype(int), 18, 90)

    gender_kk = rng.choice([1, 2], size=n, p=[0.89, 0.11])

    # === FIX 4: status_rumah correlates (richer → milik sendiri) ===
    status_rumah = _sample_correlated_ordinal(
        values=[1, 2, 3, 4],  # 1=milik sendiri (best), 4=dinas
        probs=[0.80, 0.12, 0.05, 0.03],
        ranks=ranks, rng=rng, noise_std=NOISE_LOOSE, ascending=True,
    )

    # === FIX 5: status_pekerjaan correlates (richer → formal sector) ===
    # 2=formal (best), 1=informal, 3=wirausaha, 0=tidak bekerja (worst)
    status_pekerjaan = np.where(
        is_urban,
        _sample_correlated_ordinal(
            values=[0, 3, 1, 2],   # tidak bekerja → wirausaha → informal → formal
            probs=[0.05, 0.15, 0.35, 0.45],
            ranks=ranks, rng=rng, noise_std=NOISE_LOOSE, ascending=True,
        ),
        _sample_correlated_ordinal(
            values=[0, 3, 2, 1],   # rural: formal is rare so ordering is different
            probs=[0.08, 0.17, 0.10, 0.65],
            ranks=ranks, rng=rng, noise_std=NOISE_LOOSE, ascending=True,
        ),
    )

    # === FIX 6: sektor_pekerjaan correlates (richer urban → manufaktur/jasa; rural → pertanian) ===
    # 1=pertanian (poorest), 2=jasa, 3=manufaktur, 4=konstruksi (ordering: pertanian < others)
    sektor_pekerjaan = np.where(
        is_urban,
        _sample_correlated_ordinal(
            values=[1, 4, 2, 3], probs=[0.05, 0.12, 0.55, 0.28],
            ranks=ranks, rng=rng, noise_std=NOISE_LOOSE, ascending=True,
        ),
        _sample_correlated_ordinal(
            values=[1, 4, 3, 2], probs=[0.55, 0.10, 0.10, 0.25],
            ranks=ranks, rng=rng, noise_std=NOISE_LOOSE, ascending=True,
        ),
    )

    # Kondisi fisik — tighter noise (was 0.4-0.5)
    floor_dist = cfg.get('floor_material_dist') or {33:0.5, 32:0.3, 21:0.2}
    wall_dist  = cfg.get('wall_material_dist')  or {34:0.6, 31:0.2, 23:0.2}
    roof_natl  = {31:0.491, 33:0.362, 32:0.091, 34:0.011, 35:0.019, 12:0.017, 13:0.003}
    floor_codes = _sample_correlated_dist({int(k):v for k,v in floor_dist.items()},
                                          ranks, rng, FLOOR_PMT_SCORE, noise_std=NOISE_TIGHT)
    wall_codes  = _sample_correlated_dist({int(k):v for k,v in wall_dist.items()},
                                          ranks, rng, noise_std=NOISE_TIGHT)
    roof_codes  = _sample_correlated_dist(roof_natl, ranks, rng, noise_std=NOISE_MEDIUM)

    z = (np.log(pengeluaran) - np.log(pengeluaran).mean()) / (np.log(pengeluaran).std() + 1e-9)
    luas_lantai = np.where(
        is_urban,
        np.clip(rng.lognormal(2.3 + 0.20 * z, 0.35, n), 3, 80),  # stronger slope, less sigma
        np.clip(rng.lognormal(2.1 + 0.20 * z, 0.40, n), 2, 60),
    ).round(1)

    # Utilitas — tighter noise
    water_dist  = {int(k):v for k,v in (cfg.get('water_source_dist') or {11:0.3, 21:0.3, 31:0.2, 41:0.2}).items()}
    toilet_dist = {int(k):v for k,v in (cfg.get('toilet_dist')        or {12:0.7, 16:0.1, 17:0.1, 31:0.1}).items()}
    fuel_dist   = {int(k):v for k,v in (cfg.get('cooking_fuel_dist')  or {2:0.65, 8:0.25, 5:0.08, 1:0.02}).items()}
    water_codes  = _sample_correlated_dist(water_dist, ranks, rng, WATER_PMT_SCORE, noise_std=NOISE_TIGHT)
    toilet_codes = _sample_correlated_dist(toilet_dist, ranks, rng, TOILET_PMT_SCORE, noise_std=NOISE_TIGHT)
    fuel_codes   = _sample_correlated_dist(fuel_dist, ranks, rng, FUEL_PMT, noise_std=NOISE_TIGHT)

    elec_rate   = cfg.get('pct_electricity') or 0.96
    electricity = _sample_correlated_binary(elec_rate, ranks, rng, noise_std=NOISE_TIGHT)
    daya_listrik = np.zeros(n, dtype=int)
    has_elec = np.where(electricity == 1)[0]
    if len(has_elec) > 0:
        er = pd.Series(ranks[has_elec]).rank(pct=True).values
        daya_listrik[has_elec] = _sample_correlated_dist(
            {450:0.12, 900:0.28, 1300:0.38, 2200:0.22}, er, rng, noise_std=NOISE_TIGHT,
        )

    # Aset — tighter noise (was 0.4-0.5)
    has_motorcycle = _sample_correlated_binary(cfg.get('pct_motorcycle') or 0.765, ranks, rng, NOISE_MEDIUM)
    has_car        = _sample_correlated_binary(cfg.get('pct_car')        or 0.129, ranks, rng, NOISE_TIGHT)
    has_tv         = _sample_correlated_binary(cfg.get('pct_tv')         or 0.852, ranks, rng, NOISE_TIGHT)
    has_fridge     = _sample_correlated_binary(cfg.get('pct_fridge')     or 0.573, ranks, rng, NOISE_TIGHT)
    has_mobile     = _sample_correlated_binary(cfg.get('pct_mobile')     or 0.896, ranks, rng, NOISE_MEDIUM)

    wealth_dist = {int(k):v for k,v in (cfg.get('wealth_index_dist') or {1:0.2,2:0.2,3:0.2,4:0.2,5:0.2}).items()}
    wealth_sample = _sample_correlated_dist(wealth_dist, ranks, rng, noise_std=NOISE_MEDIUM)
    has_pc = (rng.random(n) < np.where(wealth_sample >= 4, 0.45, 0.10)).astype(int)
    has_ac = (rng.random(n) < np.where(wealth_sample >= 5, 0.40, 0.05)).astype(int)
    aset_elektronik = has_tv + has_fridge + has_pc + has_ac

    is_farmer = (sektor_pekerjaan == 1)
    aset_lahan  = np.where(~is_farmer, 0, rng.choice([0, 1, 2], n, p=[0.20, 0.50, 0.30]))
    aset_ternak = np.where(~is_farmer, 0, rng.choice([0, 1, 2, 3], n, p=[0.40, 0.30, 0.20, 0.10]))

    df = pd.DataFrame({
        'record_id':                  [str(uuid.uuid4())[:8] for _ in range(n)],
        'province':                   province,
        'urban_rural':                np.where(is_urban, 1, 2),
        'jml_anggota_keluarga':       jml_anggota,
        'usia_kepala_keluarga':       usia_kk,
        'gender_kepala_keluarga':     gender_kk,
        'dependency_ratio':           dep_ratio,
        'status_kepemilikan_rumah':   status_rumah,
        'luas_lantai_per_kapita':     luas_lantai,
        'jenis_lantai':               floor_codes,
        'jenis_dinding':              wall_codes,
        'jenis_atap':                 roof_codes,
        'sumber_air_minum':           water_codes,
        'fasilitas_bab':              toilet_codes,
        'daya_listrik_terpasang':     daya_listrik,
        'bahan_bakar_memasak':        fuel_codes,
        'electricity':                electricity,
        'pendidikan_kepala_keluarga': pendidikan_kk,
        'status_pekerjaan_kk':        status_pekerjaan,
        'sektor_pekerjaan_kk':        sektor_pekerjaan,
        'pengeluaran_per_kapita':     pengeluaran,
        'kepemilikan_motor':          has_motorcycle,
        'kepemilikan_mobil':          has_car,
        'has_tv':                     has_tv,
        'has_fridge':                 has_fridge,
        'has_mobile':                 has_mobile,
        'aset_elektronik_inti':       aset_elektronik,
        'aset_lahan_pertanian':       aset_lahan,
        'aset_peternakan':            aset_ternak,
        'is_anomaly':                 np.zeros(n, dtype=int),
        'anomaly_type':               'none',
    })

    df_pmt = df.rename(columns={
        'jenis_lantai': 'floor_material', 'sumber_air_minum': 'water_source',
        'fasilitas_bab': 'toilet_facility', 'bahan_bakar_memasak': 'cooking_fuel',
        'kepemilikan_motor': 'has_motorcycle', 'kepemilikan_mobil': 'has_car',
    })
    df['skor_pmt_konvensional'] = compute_pmt_score(df_pmt)

    if scenario != 'normal' and anomaly_pct > 0:
        df = inject_anomaly(df, scenario, anomaly_pct, rng)

    df['desil_kesejahteraan_aktual'] = safe_qcut_desil(df['pengeluaran_per_kapita'], 10)
    df['desil_pmt_konvensional']     = safe_qcut_desil(df['skor_pmt_konvensional'], 10)
    df['scenario'] = scenario
    df['anomaly_pct'] = anomaly_pct
    df['dataset_id'] = str(uuid.uuid4())[:12]
    return df


def inject_anomaly(df, scenario, anomaly_pct, rng):
    df = df.copy()
    n = len(df)
    n_inj = int(n * anomaly_pct)
    if scenario == 'phk':
        eligible = df.index[(df['sektor_pekerjaan_kk'] == 3) & (df['status_pekerjaan_kk'] == 2)].tolist()
        if len(eligible) < n_inj:
            supp = df.index[(df['status_pekerjaan_kk'] == 2) & (~df.index.isin(eligible))].tolist()
            eligible = eligible + supp
        if not eligible:
            eligible = df.index.tolist()
        idx = rng.choice(eligible, size=min(n_inj, len(eligible)), replace=False)
        drop = rng.uniform(0.30, 0.60, len(idx))
        df.loc[idx, 'pengeluaran_per_kapita'] = (df.loc[idx, 'pengeluaran_per_kapita'] * drop).astype(int)
        df.loc[idx, 'status_pekerjaan_kk'] = 0
        df.loc[idx, 'is_anomaly'] = 1
        df.loc[idx, 'anomaly_type'] = 'phk'
    elif scenario == 'bencana':
        idx = rng.choice(n, size=n_inj, replace=False)
        df.loc[idx, 'jenis_lantai']     = rng.choice([11, 12, 21], n_inj)
        df.loc[idx, 'sumber_air_minum'] = rng.choice([31, 41, 42], n_inj)
        df.loc[idx, 'fasilitas_bab']    = rng.choice([41, 51, 31], n_inj)
        drop = rng.uniform(0.50, 0.70, n_inj)
        df.loc[idx, 'pengeluaran_per_kapita'] = (df.loc[idx, 'pengeluaran_per_kapita'] * drop).astype(int)
        df.loc[idx, 'is_anomaly'] = 1
        df.loc[idx, 'anomaly_type'] = 'bencana'
    return df


# ---------------- Drive ----------------
all_dfs = []
provinces = sorted(MASTER_CONFIG.keys())
for i, prov in enumerate(provinces, 1):
    for scn in ('normal', 'phk', 'bencana'):
        d = generate_households(prov, N_HOUSEHOLDS, scn, ANOMALY_RATIOS[scn], rng)
        all_dfs.append(d)
    logger.info('progress %d/%d done province=%s', i, len(provinces), prov)

df_all = pd.concat(all_dfs, ignore_index=True)
df_all.to_parquet(OUTPUT_PATH, index=False)
logger.info('saved path=%s rows=%d size_mb=%.1f',
            OUTPUT_PATH, len(df_all), OUTPUT_PATH.stat().st_size / 1e6)

print(f'\nGenerated: {len(df_all):,} rows, {df_all.shape[1]} cols')
print(f'Anomaly rate per scenario:')
for sc, g in df_all.groupby('scenario'):
    print(f'  {sc:<10}: {g["is_anomaly"].mean()*100:.1f}% (n={len(g):,})')
