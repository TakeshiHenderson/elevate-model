"""In-memory session store. Each session holds the generated DataFrame and predictions."""
import uuid
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

_store: dict[str, dict] = {}
SESSION_TTL_HOURS = 2


def create(province_id: str, scenario: str, anomaly_pct: float, df: pd.DataFrame) -> str:
    sid = str(uuid.uuid4())
    _store[sid] = {
        "province_id": province_id,
        "scenario": scenario,
        "anomaly_pct": anomaly_pct,
        "df": df,
        "predictions": None,
        "created_at": datetime.utcnow(),
    }
    _cleanup()
    return sid


def get(sid: str) -> Optional[dict]:
    return _store.get(sid)


def set_predictions(sid: str, predictions: dict) -> None:
    if sid in _store:
        _store[sid]["predictions"] = predictions


def _cleanup() -> None:
    cutoff = datetime.utcnow() - timedelta(hours=SESSION_TTL_HOURS)
    expired = [k for k, v in _store.items() if v["created_at"] < cutoff]
    for k in expired:
        del _store[k]
