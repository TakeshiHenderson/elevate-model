"""Wraps regenerate_data.generate_households for per-province on-demand generation."""
import os
import sys
import numpy as np
import pandas as pd

# Locate regenerate_data.py — flattened into /app in Docker, repo root in local dev.
_candidates = [
    os.environ.get("REPO_ROOT"),
    "/app",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
]
for _c in _candidates:
    if _c and os.path.exists(os.path.join(_c, "regenerate_data.py")):
        if _c not in sys.path:
            sys.path.insert(0, _c)
        break

from regenerate_data import generate_households, MASTER_CONFIG  # noqa: E402

PROVINCE_CODE: dict[str, str] = {
    "Aceh": "AC", "Sumatera Utara": "SU", "Sumatera Barat": "SB",
    "Riau": "RI", "Jambi": "JA", "Sumatera Selatan": "SS",
    "Bengkulu": "BE", "Lampung": "LA", "Kepulauan Bangka Belitung": "BB",
    "Kepulauan Riau": "KR", "DKI Jakarta": "JK", "Jawa Barat": "JB",
    "Jawa Tengah": "JT", "DI Yogyakarta": "YO", "Jawa Timur": "JI",
    "Banten": "BT", "Bali": "BA", "Nusa Tenggara Barat": "NB",
    "Nusa Tenggara Timur": "NT", "Kalimantan Barat": "KB",
    "Kalimantan Tengah": "KT", "Kalimantan Selatan": "KS",
    "Kalimantan Timur": "KI", "Kalimantan Utara": "KU",
    "Sulawesi Utara": "SA", "Sulawesi Tengah": "ST", "Sulawesi Selatan": "SN",
    "Sulawesi Tenggara": "SG", "Gorontalo": "GO", "Sulawesi Barat": "SR",
    "Maluku": "MA", "Maluku Utara": "MU", "Papua Barat": "PB",
    "Papua": "PA", "Papua Selatan": "PS", "Papua Tengah": "PT",
    "Papua Pegunungan": "PP", "Papua Barat Daya": "PD",
}

KECAMATAN_POOL: dict[str, list[str]] = {
    "JB": ["Bandung Barat","Cimahi Utara","Cianjur","Garut Kota","Sukabumi"],
    "JK": ["Penjaringan","Kebayoran Baru","Cengkareng","Pasar Minggu","Cilincing"],
    "JT": ["Semarang Utara","Cilacap","Solo Baru","Purwokerto","Kudus"],
    "JI": ["Surabaya Barat","Malang Kota","Kediri","Pasuruan","Jember"],
    "SU": ["Medan Baru","Deli Serdang","Langkat","Simalungun","Asahan"],
    "AC": ["Banda Aceh","Aceh Besar","Bireuen","Pidie","Aceh Utara"],
    "SB": ["Padang","Bukittinggi","Payakumbuh","Solok","Tanah Datar"],
    "RI": ["Pekanbaru","Dumai","Bengkalis","Rokan Hilir","Kampar"],
    "JA": ["Jambi Kota","Batanghari","Muaro Jambi","Tanjung Jabung","Sarolangun"],
    "SS": ["Palembang","Ogan Ilir","Banyuasin","Muara Enim","Lahat"],
    "BE": ["Bengkulu Kota","Seluma","Kepahiang","Bengkulu Tengah","Kaur"],
    "LA": ["Bandar Lampung","Lampung Selatan","Lampung Tengah","Pringsewu","Metro"],
    "BB": ["Pangkalpinang","Bangka","Belitung","Bangka Tengah","Bangka Barat"],
    "KR": ["Batam","Tanjungpinang","Bintan","Karimun","Lingga"],
    "YO": ["Sleman","Bantul","Gunung Kidul","Kulon Progo","Yogyakarta"],
    "BT": ["Serang","Tangerang","Cilegon","Lebak","Pandeglang"],
    "BA": ["Denpasar","Badung","Gianyar","Tabanan","Buleleng"],
    "NB": ["Mataram","Lombok Tengah","Sumbawa","Lombok Timur","Bima"],
    "NT": ["Kupang","Ende","Flores Timur","Sikka","Timor Tengah Selatan"],
    "KB": ["Pontianak","Mempawah","Sanggau","Ketapang","Sintang"],
    "KT": ["Palangkaraya","Kotawaringin","Barito Selatan","Kapuas","Katingan"],
    "KS": ["Banjarmasin","Banjarbaru","Tanah Laut","Barito Kuala","Kotabaru"],
    "KI": ["Samarinda","Balikpapan","Kutai Kartanegara","Bontang","Berau"],
    "KU": ["Tarakan","Bulungan","Malinau","Nunukan","Tana Tidung"],
    "SA": ["Manado","Bitung","Minahasa","Tomohon","Bolaang Mongondow"],
    "ST": ["Palu","Donggala","Sigi","Morowali","Tojo Una-Una"],
    "SN": ["Makassar","Gowa","Bone","Jeneponto","Maros"],
    "SG": ["Kendari","Konawe","Kolaka","Buton","Muna"],
    "GO": ["Gorontalo Kota","Bone Bolango","Pohuwato","Boalemo","Gorontalo"],
    "SR": ["Mamuju","Majene","Polewali Mandar","Mamasa","Pasangkayu"],
    "MA": ["Ambon","Maluku Tengah","Seram Bagian Barat","Buru","Maluku Tenggara"],
    "MU": ["Ternate","Halmahera Utara","Tidore","Halmahera Barat","Morotai"],
    "PB": ["Manokwari","Sorong","Fakfak","Kaimana","Teluk Wondama"],
    "PA": ["Jayapura","Mimika","Merauke","Jayawijaya","Biak"],
    "PS": ["Merauke","Mappi","Asmat","Boven Digoel","Musi"],
    "PT": ["Nabire","Puncak","Puncak Jaya","Paniai","Dogiyai"],
    "PP": ["Jayawijaya","Pegunungan Bintang","Yahukimo","Tolikara","Nduga"],
    "PD": ["Sorong","Raja Ampat","Maybrat","Tambrauw","South Sorong"],
}
_DEFAULT_KEC = ["Kecamatan A","Kecamatan B","Kecamatan C","Kecamatan D","Kecamatan E"]

EDU_LABELS   = {0:"Tidak Sekolah", 1:"SD", 2:"SMP", 3:"SMA", 4:"Perguruan Tinggi"}
SECTOR_LABELS = {1:"Pertanian", 2:"Jasa", 3:"Manufaktur", 4:"Konstruksi", 0:"Tidak Bekerja"}
GENDER_LABELS = {1:"L", 2:"P"}
FLOOR_LABELS  = {
    11:"Tanah", 12:"Bambu", 21:"Kayu", 22:"Kayu", 31:"Semen", 32:"Semen",
    33:"Keramik", 34:"Keramik", 35:"Marmer", 36:"Marmer", 96:"Lainnya",
}
WATER_LABELS  = {
    11:"Air Kemasan", 12:"Air Kemasan", 13:"Ledeng", 14:"Ledeng", 21:"Sumur Bor",
    31:"Sumur Galian", 32:"Sumur Galian", 41:"Mata Air", 42:"Mata Air",
    43:"Sumur", 51:"Air Hujan", 61:"Mata Air Tak Terlindung",
    62:"Air Permukaan", 71:"Air Sungai", 72:"Air Danau/Kolam", 96:"Lainnya",
}
STATUS_LABELS = {
    "none": "normal", "phk": "phk_risk", "bencana": "disaster_risk",
}


def generate(province_id: str, scenario: str, anomaly_pct: float, n: int = 10_000) -> pd.DataFrame:
    if province_id not in MASTER_CONFIG:
        raise ValueError(f"Unknown province: {province_id}")

    rng = np.random.default_rng()
    df = generate_households(province_id, n, scenario, anomaly_pct, rng)

    code = PROVINCE_CODE.get(province_id, province_id[:2].upper())
    pool = KECAMATAN_POOL.get(code, _DEFAULT_KEC)
    df['kecamatan'] = rng.choice(pool, n)
    df['id'] = [f"KK-{code}-{i:04d}" for i in range(n)]
    df['expenditure'] = (df['pengeluaran_per_kapita'] * df['jml_anggota_keluarga']).astype(int)
    df['edu_label']    = df['pendidikan_kepala_keluarga'].map(EDU_LABELS).fillna('Lainnya')
    df['sector_label'] = df['sektor_pekerjaan_kk'].map(SECTOR_LABELS).fillna('Lainnya')
    df['gender_label'] = df['gender_kepala_keluarga'].map(GENDER_LABELS).fillna('L')
    df['floor_label']  = df['jenis_lantai'].map(FLOOR_LABELS).fillna('Lainnya')
    df['water_label']  = df['sumber_air_minum'].map(WATER_LABELS).fillna('Lainnya')
    df['status_label'] = df['anomaly_type'].map(STATUS_LABELS).fillna('normal')

    return df


def to_records(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame rows to the shape expected by the frontend."""
    out = []
    for _, r in df.iterrows():
        out.append({
            "id": r["id"],
            "kecamatan": r["kecamatan"],
            "hhSize": int(r["jml_anggota_keluarga"]),
            "headAge": int(r["usia_kepala_keluarga"]),
            "headGender": r["gender_label"],
            "edu": r["edu_label"],
            "sector": r["sector_label"],
            "expenditure": int(r["expenditure"]),
            "floor": r["floor_label"],
            "water": r["water_label"],
            "ownCar": bool(r["kepemilikan_mobil"]),
            "ownMotor": bool(r["kepemilikan_motor"]),
            "pmtScore": float(r["skor_pmt_konvensional"]),
            "pmtDecile": int(r["desil_pmt_konvensional"]),
            "actualDecile": int(r["desil_kesejahteraan_aktual"]),
            "isAnomaly": bool(r["is_anomaly"]),
            "status": r["status_label"],
        })
    return out
