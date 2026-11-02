"""Customer360 – FastAPI endpoint tests.

Uses FastAPI's TestClient (synchronous HTTPX transport) so no live server
is required.  The lifespan hook loads the model artifacts from models/.

Run with:
    .\\venv\\Scripts\\python -m pytest tests/test_api.py -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi.testclient import TestClient
from src.api.main import app

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    """Start the app (including lifespan / model loading) once per module."""
    with TestClient(app) as c:
        yield c


# A minimal valid customer payload used by multiple tests
_VALID_CUSTOMER = {
    "customerID": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.75,
    "TotalCharges": 478.75,
}

_STABLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 48,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "DSL",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0,
    "TotalCharges": 2640.0,
}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_model_loaded(self, client):
        body = client.get("/health").json()
        assert body["model_loaded"] is True

    def test_health_model_name(self, client):
        body = client.get("/health").json()
        assert body["model_name"] == "XGBClassifier"

    def test_health_status_ok(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

class TestSinglePredict:
    def test_valid_customer_returns_200(self, client):
        resp = client.post("/predict", json=_VALID_CUSTOMER)
        assert resp.status_code == 200

    def test_probability_in_unit_interval(self, client):
        body = client.post("/predict", json=_VALID_CUSTOMER).json()
        prob = body["churn_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predicted_churn_is_yes_or_no(self, client):
        body = client.post("/predict", json=_VALID_CUSTOMER).json()
        assert body["predicted_churn"] in ("Yes", "No")

    def test_risk_category_valid(self, client):
        body = client.post("/predict", json=_VALID_CUSTOMER).json()
        assert body["risk_category"] in ("High Risk", "Medium Risk", "Low Risk")

    def test_customer_id_echoed_in_response(self, client):
        body = client.post("/predict", json=_VALID_CUSTOMER).json()
        assert body["customerID"] == "TEST-001"

    def test_high_risk_threshold_matches(self, client):
        body = client.post("/predict", json=_VALID_CUSTOMER).json()
        prob = body["churn_probability"]
        category = body["risk_category"]
        if category == "High Risk":
            assert prob >= 0.70
        elif category == "Medium Risk":
            assert 0.40 <= prob < 0.70
        else:
            assert prob < 0.40

    def test_churn_label_not_accepted_as_feature(self, client):
        """Sending a Churn field must not alter prediction or cause 500."""
        payload = dict(_VALID_CUSTOMER, Churn="No")
        resp = client.post("/predict", json=payload)
        # Pydantic v2 by default ignores extra fields; result must still be valid
        assert resp.status_code == 200
        body = resp.json()
        assert 0.0 <= body["churn_probability"] <= 1.0

    def test_customer_id_not_in_features(self, client):
        """Sending customerID must not change the probability."""
        with_id = dict(_VALID_CUSTOMER, customerID="ANY-ID")
        without_id = {k: v for k, v in _VALID_CUSTOMER.items() if k != "customerID"}
        p1 = client.post("/predict", json=with_id).json()["churn_probability"]
        p2 = client.post("/predict", json=without_id).json()["churn_probability"]
        assert abs(p1 - p2) < 1e-6

    def test_missing_required_field_rejected(self, client):
        incomplete = {k: v for k, v in _VALID_CUSTOMER.items() if k != "tenure"}
        resp = client.post("/predict", json=incomplete)
        assert resp.status_code == 422

    def test_invalid_gender_rejected(self, client):
        payload = dict(_VALID_CUSTOMER, gender="Unknown")
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_negative_tenure_rejected(self, client):
        payload = dict(_VALID_CUSTOMER, tenure=-1)
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_contract_rejected(self, client):
        payload = dict(_VALID_CUSTOMER, Contract="Weekly")
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

class TestBatchPredict:
    def test_batch_returns_200(self, client):
        payload = {"customers": [
            {"customerID": "B-001", **_STABLE_CUSTOMER},
            {"customerID": "B-002", **_VALID_CUSTOMER},
        ]}
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == 200

    def test_batch_count_matches_input(self, client):
        customers = [{"customerID": f"B-{i:03d}", **_STABLE_CUSTOMER} for i in range(5)]
        body = client.post("/predict/batch", json={"customers": customers}).json()
        assert body["count"] == 5
        assert len(body["predictions"]) == 5

    def test_batch_all_probs_in_unit_interval(self, client):
        customers = [{"customerID": "B-001", **_STABLE_CUSTOMER}, {"customerID": "B-002", **_VALID_CUSTOMER}]
        preds = client.post("/predict/batch", json={"customers": customers}).json()["predictions"]
        for p in preds:
            assert 0.0 <= p["churn_probability"] <= 1.0

    def test_batch_all_risk_categories_valid(self, client):
        customers = [{"customerID": "B-001", **_STABLE_CUSTOMER}, {"customerID": "B-002", **_VALID_CUSTOMER}]
        preds = client.post("/predict/batch", json={"customers": customers}).json()["predictions"]
        valid = {"High Risk", "Medium Risk", "Low Risk"}
        for p in preds:
            assert p["risk_category"] in valid

    def test_batch_customer_ids_preserved(self, client):
        # Strip customerID from _VALID_CUSTOMER so our explicit key wins
        base = {k: v for k, v in _VALID_CUSTOMER.items() if k != "customerID"}
        customers = [
            {"customerID": "ID-ALPHA", **_STABLE_CUSTOMER},
            {"customerID": "ID-BETA", **base},
        ]
        preds = client.post("/predict/batch", json={"customers": customers}).json()["predictions"]
        ids = [p["customerID"] for p in preds]
        assert ids == ["ID-ALPHA", "ID-BETA"]

    def test_empty_batch_rejected(self, client):
        resp = client.post("/predict/batch", json={"customers": []})
        assert resp.status_code == 422

    def test_invalid_customer_in_batch_rejected(self, client):
        bad = [{"customerID": "BAD", "gender": "UNKNOWN", "SeniorCitizen": 99}]
        resp = client.post("/predict/batch", json={"customers": bad})
        assert resp.status_code == 422

    def test_churn_label_excluded_from_batch(self, client):
        """Historical Churn labels must not influence batch predictions."""
        base = dict(_STABLE_CUSTOMER)
        with_churn = {"customerID": "B-CHURN", **base, "Churn": "Yes"}
        without_churn = {"customerID": "B-NO-CHURN", **base}
        preds = client.post(
            "/predict/batch",
            json={"customers": [with_churn, without_churn]},
        ).json()["predictions"]
        p1 = preds[0]["churn_probability"]
        p2 = preds[1]["churn_probability"]
        # Both should produce the same score since Churn is ignored by schema
        assert abs(p1 - p2) < 1e-6

    def test_batch_exceeding_schema_limit_rejected(self, client):
        """FastAPI schema enforces max_length=5000; 5001 records must return HTTP 422."""
        too_many = [{"customerID": f"C-{i}", **_STABLE_CUSTOMER} for i in range(5001)]
        resp = client.post("/predict/batch", json={"customers": too_many})
        assert resp.status_code == 422
        assert "5000" in resp.text


# ---------------------------------------------------------------------------
# Streamlit Chunked Scoring Integration Tests (7,043 customer verification)
# ---------------------------------------------------------------------------

class TestStreamlitChunkedScoring:
    def test_7043_records_processed_in_two_batches(self, monkeypatch, client):
        """Verify that 7,043 records are chunked into two batches (5,000 + 2,043),
        sent sequentially to /predict/batch, and returned in the exact customer order.
        """
        import pandas as pd
        from dashboard.app import score_customers_via_api

        # Create 7,043 synthetic customer records
        n_total = 7043
        records = []
        for i in range(n_total):
            rec = dict(_STABLE_CUSTOMER)
            rec["customerID"] = f"CUST-{i:05d}"
            # Alternate tenure to ensure distinct, predictable features
            rec["tenure"] = (i % 72) + 1
            records.append(rec)

        test_df = pd.DataFrame(records)

        # Track batch requests made to client
        batch_sizes = []

        def mock_post(url, json=None, timeout=None):
            assert url.endswith("/predict/batch")
            customers = json["customers"]
            batch_sizes.append(len(customers))
            # Route to our in-memory TestClient app
            return client.post("/predict/batch", json=json)

        monkeypatch.setattr("dashboard.app.requests.post", mock_post)

        scored_df = score_customers_via_api(test_df, chunk_size=5000)

        # 1. Exactly two batches: 5,000 and 2,043
        assert batch_sizes == [5000, 2043], f"Expected [5000, 2043], got {batch_sizes}"

        # 2. All 7,043 records receive predictions exactly once
        assert len(scored_df) == n_total

        # 3. Customer order is strictly preserved
        assert list(scored_df["customerID"]) == list(test_df["customerID"])

        # 4. Required output columns exist and are fully populated
        assert "churn_probability" in scored_df.columns
        assert "churn_probability_pct" in scored_df.columns
        assert "predicted_churn" in scored_df.columns
        assert "risk_category" in scored_df.columns

        assert not scored_df["churn_probability"].isna().any()
        assert ((scored_df["churn_probability"] >= 0.0) & (scored_df["churn_probability"] <= 1.0)).all()
        assert set(scored_df["predicted_churn"].unique()).issubset({"Yes", "No"})
        assert set(scored_df["risk_category"].unique()).issubset({"High Risk", "Medium Risk", "Low Risk"})


class TestPredictSingleCustomerViaApi:
    def test_predict_single_customer_success(self, monkeypatch, client):
        """Verify predict_single_customer_via_api formats payload and parses API response."""
        from dashboard.app import predict_single_customer_via_api

        # Mock requests.post to route directly to TestClient
        def mock_post(url, json=None, timeout=None):
            assert url.endswith("/predict")
            return client.post("/predict", json=json)

        monkeypatch.setattr("dashboard.app.requests.post", mock_post)

        new_customer_payload = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 85.50,
            "TotalCharges": 1026.00,
        }

        res = predict_single_customer_via_api(new_customer_payload)
        assert "churn_probability" in res
        assert 0.0 <= res["churn_probability"] <= 1.0
        assert res["predicted_churn"] in ("Yes", "No")
        assert res["risk_category"] in ("High Risk", "Medium Risk", "Low Risk")
        assert "message" in res

    def test_predict_single_customer_api_error_handling(self, monkeypatch):
        """Verify predict_single_customer_via_api raises clean RuntimeError on failure."""
        import requests
        from dashboard.app import predict_single_customer_via_api

        def mock_post_fail(url, json=None, timeout=None):
            raise requests.exceptions.ConnectionError("Connection refused")

        monkeypatch.setattr("dashboard.app.requests.post", mock_post_fail)

        with pytest.raises(RuntimeError) as exc_info:
            predict_single_customer_via_api({"gender": "Female"})
        assert "Cannot reach FastAPI service" in str(exc_info.value)
