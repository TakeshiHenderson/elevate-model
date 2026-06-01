"""Generates a 2-page policy brief in Bahasa Indonesia via Azure OpenAI."""
from openai import AzureOpenAI

from config import settings

_client = AzureOpenAI(
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_KEY,
    api_version="2024-02-01",
)

SCENARIO_LABELS = {
    "normal": "Kondisi Normal (Baseline)",
    "phk": "PHK Massal (Sektor Manufaktur)",
    "bencana": "Pasca-Bencana",
}


def generate(province: str, scenario: str, stats: dict) -> dict:
    """
    stats keys expected:
      total_records, eligible_pct, pmt_exclusion_err, ml_exclusion_err,
      pmt_inclusion_err, ml_inclusion_err, anomaly_pct,
      ml_f1, ml_auc, pmt_f1
    """
    scenario_label = SCENARIO_LABELS.get(scenario, scenario)
    prompt = f"""Anda adalah analis kebijakan sosial senior di Indonesia.
Buatkan policy brief dua halaman dalam Bahasa Indonesia yang profesional berdasarkan
hasil simulasi sistem RightAid ML-PMT Refresher untuk {province} dalam skenario {scenario_label}.

Data hasil simulasi:
- Total rumah tangga dianalisis: {stats.get('total_records', 10000):,}
- Persentase rumah tangga layak bansos (desil ≤ 4): {stats.get('eligible_pct', 40):.1f}%
- Exclusion error PMT konvensional (yang layak tapi terlewat): {stats.get('pmt_exclusion_err', 0):.2f}%
- Exclusion error model ML: {stats.get('ml_exclusion_err', 0):.2f}%
- Inclusion error PMT konvensional: {stats.get('pmt_inclusion_err', 0):.2f}%
- Inclusion error model ML: {stats.get('ml_inclusion_err', 0):.2f}%
- Persentase anomali/kasus mis-targeting terdeteksi: {stats.get('anomaly_pct', 0)*100:.1f}%
- F1-score model ML: {stats.get('ml_f1', 0):.4f} vs PMT: {stats.get('pmt_f1', 0):.4f}
- AUC model ML: {stats.get('ml_auc', 0):.4f}

Susun policy brief dengan struktur berikut (gunakan heading yang jelas):

1. RINGKASAN TEMUAN UTAMA
   Deskripsikan tingkat mis-targeting PMT konvensional vs model ML untuk {province}
   dalam skenario {scenario_label}. Sebutkan angka konkret.

2. PROFIL DEMOGRAFIS TERDAMPAK
   Jelaskan karakteristik kelompok yang paling sering terlewat (excluded) oleh PMT
   berdasarkan skenario ini (pekerja formal yang di-PHK / korban bencana / dll).

3. REKOMENDASI TINDAK LANJUT
   Berikan 3-4 rekomendasi kebijakan yang dapat segera ditindaklanjuti oleh
   Kemensos/Dinsos dalam 6 bulan ke depan.

Gunakan bahasa yang dapat dipahami oleh pejabat non-teknis. Maksimal 600 kata."""

    response = _client.chat.completions.create(
        model=settings.AZURE_OPENAI_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.4,
    )

    content = response.choices[0].message.content
    title = f"Policy Brief: Analisis Mis-Targeting Bansos — {province} ({scenario_label})"
    return {"title": title, "content": content}
