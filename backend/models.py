from typing import Optional
from pydantic import BaseModel


# ── Auth ──────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    token: str
    user: dict


# ── Generate ──────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    province_id: str           # province name as in province_master_config.json
    scenario: str              # "normal" | "phk" | "bencana"
    anomaly_pct: float = 0.1   # 0.0 – 0.30
    n: int = 10_000


class GenerateResponse(BaseModel):
    session_id: str
    province_id: str
    scenario: str
    total_records: int
    preview: list[dict]        # first 5 rows


# ── Data ──────────────────────────────────────────────────────────────────────
class HouseholdRecord(BaseModel):
    id: str
    kecamatan: str
    hhSize: int
    headAge: int
    headGender: str
    edu: str
    sector: str
    expenditure: int
    floor: str
    water: str
    ownCar: bool
    ownMotor: bool
    pmtScore: float
    pmtDecile: int
    actualDecile: int
    mlScore: Optional[float] = None
    mlEligible: Optional[int] = None
    isAnomaly: Optional[bool] = None
    status: Optional[str] = None
    confidence: Optional[float] = None


class DataResponse(BaseModel):
    records: list[dict]
    total: int
    page: int
    pages: int


# ── Predict ───────────────────────────────────────────────────────────────────
class ConfusionMatrix(BaseModel):
    tp: int
    fp: int
    fn: int
    tn: int


class Metrics(BaseModel):
    exclusionErr: float
    inclusionErr: float
    f1: float
    auc: float


class DecileDist(BaseModel):
    actual: list[int]
    pmt: list[int]
    ml: list[int]


class MistargetingCandidate(BaseModel):
    id: str
    kecamatan: str
    actualDecile: int
    pmtDecile: int
    mlEligible: int
    confidence: float
    isAnomaly: bool
    status: str


class PredictResponse(BaseModel):
    confusion_matrix: ConfusionMatrix
    confusion_matrix_pmt: ConfusionMatrix
    metrics: Metrics
    metrics_pmt: Metrics
    decile_dist: DecileDist
    mistargeting_candidates: list[dict]


# ── SHAP ──────────────────────────────────────────────────────────────────────
class ShapFeature(BaseModel):
    feature: str
    value: float


class ShapResponse(BaseModel):
    record_id: str
    features: list[ShapFeature]
    prediction: float
    eligible: int


# ── Policy Brief ──────────────────────────────────────────────────────────────
class PolicyBriefRequest(BaseModel):
    province_id: str
    scenario: str
    session_id: str


class PolicyBriefResponse(BaseModel):
    title: str
    content: str               # full Bahasa Indonesia policy brief text
