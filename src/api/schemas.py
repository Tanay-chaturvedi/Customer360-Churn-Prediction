"""Pydantic request/response schemas for the Customer360 prediction API.

Field values match the IBM Telco Customer Churn training data exactly.
customerID and Churn are excluded from all inference inputs.
"""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Raw customer attributes accepted by the prediction endpoints.

    customerID is stored for traceability only and never enters the model.
    The historical Churn label is intentionally absent from this schema.
    """

    customerID: Optional[str] = Field(default=None, description="Reference ID (not used in inference).")
    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=200, description="Months with the company.")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly bill amount (USD).")
    TotalCharges: float = Field(..., ge=0.0, description="Cumulative charges (USD).")

    model_config = {"str_strip_whitespace": True}


class PredictionResult(BaseModel):
    """Churn prediction result for a single customer."""

    customerID: Optional[str] = None
    churn_probability: float = Field(..., description="Estimated churn probability [0, 1].")
    predicted_churn: Literal["Yes", "No"]
    risk_category: Literal["High Risk", "Medium Risk", "Low Risk"]
    message: str = Field(..., description="Disclaimer about the probabilistic nature of predictions.")


class BatchPredictionRequest(BaseModel):
    """Batch of customer records to score in a single call."""

    customers: list[CustomerFeatures] = Field(
        ..., min_length=1, max_length=5000,
        description="Between 1 and 5,000 customer records per request.",
    )


class BatchPredictionResponse(BaseModel):
    """Response for a batch prediction call."""

    count: int
    predictions: list[PredictionResult]


class HealthResponse(BaseModel):
    """Health-check response payload."""

    status: str
    model_loaded: bool
    model_name: str
