"""Model loading and inference logic for the Customer360 prediction API.

The model and preprocessor are loaded exactly once at API startup into
module-level singletons so every request reuses the same in-memory objects.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("Customer360.API.Predictor")

_PROJECT_ROOT      = Path(__file__).resolve().parents[2]
_MODEL_PATH        = _PROJECT_ROOT / "models" / "best_model.joblib"
_PREPROCESSOR_PATH = _PROJECT_ROOT / "models" / "preprocessor.joblib"

# Illustrative risk thresholds (business categories, not calibrated)
HIGH_RISK_THRESHOLD   = 0.70
MEDIUM_RISK_THRESHOLD = 0.40

# Columns that must NEVER enter the feature matrix
_EXCLUDED_COLS = {"customerID", "Churn"}

_model: Any | None = None
_preprocessor: Any | None = None


def load_artifacts() -> None:
    """Load model and preprocessor from disk. Call once at API startup."""
    global _model, _preprocessor
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found: {_MODEL_PATH}")
    if not _PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {_PREPROCESSOR_PATH}")
    _model = joblib.load(_MODEL_PATH)
    _preprocessor = joblib.load(_PREPROCESSOR_PATH)
    logger.info("Loaded %s + %s.", type(_model).__name__, type(_preprocessor).__name__)


def is_loaded() -> bool:
    """Return True when artifacts are resident in memory."""
    return _model is not None and _preprocessor is not None


def get_model_name() -> str:
    """Return the class name of the loaded estimator."""
    return type(_model).__name__ if _model is not None else "Not loaded"


def assign_risk_category(prob: float) -> str:
    """Map a churn probability to an illustrative risk tier."""
    if prob >= HIGH_RISK_THRESHOLD:
        return "High Risk"
    if prob >= MEDIUM_RISK_THRESHOLD:
        return "Medium Risk"
    return "Low Risk"


_DISCLAIMER = (
    "This probability is a model estimate and does not represent certainty. "
    "Risk categories are illustrative business segments intended to support "
    "prioritisation, not calibrated intervention thresholds."
)


def predict_single(features: dict) -> dict:
    """Score a single customer record and return a prediction dict.

    customerID is extracted for traceability and excluded from the feature matrix.
    No Churn field should be present in features.
    """
    if not is_loaded():
        raise RuntimeError("Model artifacts are not loaded.")
    customer_id = features.pop("customerID", None)
    for col in _EXCLUDED_COLS:
        features.pop(col, None)
    df = pd.DataFrame([features])
    X = _preprocessor.transform(df)
    prob = float(np.clip(_model.predict_proba(X)[0, 1], 0.0, 1.0))
    return {
        "customerID": customer_id,
        "churn_probability": round(prob, 4),
        "predicted_churn": "Yes" if prob >= 0.5 else "No",
        "risk_category": assign_risk_category(prob),
        "message": _DISCLAIMER,
    }


def predict_batch(records: list[dict]) -> list[dict]:
    """Score a list of customer records efficiently in a single batch.

    customerID values are extracted and re-attached to results; they never
    enter the model feature matrix.
    """
    if not is_loaded():
        raise RuntimeError("Model artifacts are not loaded.")
    customer_ids = [r.pop("customerID", None) for r in records]
    for r in records:
        for col in _EXCLUDED_COLS:
            r.pop(col, None)
    df = pd.DataFrame(records)
    X = _preprocessor.transform(df)
    probs = np.clip(_model.predict_proba(X)[:, 1], 0.0, 1.0)
    return [
        {
            "customerID": cid,
            "churn_probability": round(float(prob), 4),
            "predicted_churn": "Yes" if prob >= 0.5 else "No",
            "risk_category": assign_risk_category(float(prob)),
            "message": _DISCLAIMER,
        }
        for cid, prob in zip(customer_ids, probs)
    ]
