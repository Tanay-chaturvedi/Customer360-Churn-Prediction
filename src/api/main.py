"""Customer360 FastAPI application.

Endpoints
---------
GET  /health          Service and model status.
POST /predict         Single-customer churn probability prediction.
POST /predict/batch   Batch prediction (1-5,000 records).
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api import predictor
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    PredictionResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("Customer360.API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts exactly once at startup."""
    logger.info("Loading model artifacts...")
    predictor.load_artifacts()
    logger.info("Model artifacts loaded successfully.")
    yield
    logger.info("Customer360 API shutting down.")


app = FastAPI(
    title="Customer360 Churn Prediction API",
    description=(
        "FastAPI backend for the Customer360 Retention Intelligence Platform. "
        "Serves the champion XGBoost churn prediction model via REST endpoints. "
        "All predictions are probabilistic estimates, not certainties."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Monitoring"],
)
def health_check() -> HealthResponse:
    """Return service status and model availability."""
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded(),
        model_name=predictor.get_model_name(),
    )


@app.post(
    "/predict",
    response_model=PredictionResult,
    summary="Single-customer churn prediction",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
)
def predict_single(customer: CustomerFeatures) -> PredictionResult:
    """Predict churn probability for one customer.

    customerID is accepted for reference only and is never used as a model feature.
    Risk thresholds: High Risk >= 0.70, Medium Risk 0.40-0.69, Low Risk < 0.40.
    """
    try:
        result = predictor.predict_single(customer.model_dump())
        return PredictionResult(**result)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction error for customer %s", customer.customerID)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred during prediction. Check API logs.") from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch customer churn prediction",
    tags=["Prediction"],
    status_code=status.HTTP_200_OK,
)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Predict churn probabilities for 1-5,000 customers in one call.

    Records are validated before inference. Results are returned in the same order.
    """
    try:
        records = [c.model_dump() for c in request.customers]
        results = predictor.predict_batch(records)
        return BatchPredictionResponse(
            count=len(results),
            predictions=[PredictionResult(**r) for r in results],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An error occurred during batch prediction. Check API logs.") from exc
