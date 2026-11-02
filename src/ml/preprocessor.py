"""Feature preprocessing and transformation pipeline for Customer360.

Provides leak-free, reproducible transformation for customer churn modeling:
- TotalCharges sanitization (whitespace to numeric float)
- Numerical standardization (StandardScaler)
- Categorical one-hot encoding (OneHotEncoder with handle_unknown='ignore')
- Serialization to models/preprocessor.joblib
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

logger = logging.getLogger("Customer360.Preprocessor")

# Canonical column definitions
NUMERICAL_FEATURES: List[str] = ['tenure', 'MonthlyCharges', 'TotalCharges']
CATEGORICAL_FEATURES: List[str] = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]


class DataCleaner(BaseEstimator, TransformerMixin):
    """Custom transformer to sanitize anomalies prior to scaling/encoding.

    1. Converts TotalCharges to numeric, replacing whitespace strings with 0.00.
    2. Casts SeniorCitizen to string to guarantee categorical treatment.
    """
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        if 'TotalCharges' in X_out.columns:
            # Handle blank strings (' ') and non-numeric characters safely
            X_out['TotalCharges'] = pd.to_numeric(X_out['TotalCharges'], errors='coerce').fillna(0.0)

        if 'SeniorCitizen' in X_out.columns:
            X_out['SeniorCitizen'] = X_out['SeniorCitizen'].astype(str)

        return X_out


def build_preprocessor() -> Pipeline:
    """Construct the scikit-learn preprocessing pipeline."""
    column_transformer = ColumnTransformer(
        transformers=[
            (
                'num',
                StandardScaler(),
                NUMERICAL_FEATURES
            ),
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                CATEGORICAL_FEATURES
            )
        ],
        remainder='drop'
    )

    pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('transformer', column_transformer)
    ])

    return pipeline


def fit_and_transform_splits(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    save_path: Path = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline, List[str]]:
    """Fit the preprocessor strictly on X_train to prevent data leakage,
    then transform X_train, X_val, and X_test.

    Parameters:
        X_train: Training features DataFrame
        X_val: Validation features DataFrame
        X_test: Holdout test features DataFrame
        save_path: Optional path to save preprocessor artifact (default: models/preprocessor.joblib)

    Returns:
        (X_train_proc, X_val_proc, X_test_proc, preprocessor, feature_names)
    """
    logger.info("Instantiating preprocessing pipeline...")
    preprocessor = build_preprocessor()

    # Fit strictly on training data
    logger.info("Fitting preprocessor strictly on X_train (preventing data leakage)...")
    X_train_proc = preprocessor.fit_transform(X_train)

    # Transform validation and test data
    logger.info("Transforming validation and holdout test feature matrices...")
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    # Extract human-interpretable feature names
    raw_feature_names = preprocessor.named_steps['transformer'].get_feature_names_out()
    # Clean prefixes (num__tenure -> tenure, cat__Contract_One year -> Contract_One year)
    feature_names = [f.replace('num__', '').replace('cat__', '') for f in raw_feature_names]

    logger.info(f"Generated {len(feature_names)} processed features from {X_train.shape[1]} raw attributes.")

    # Save artifact
    if save_path is None:
        save_path = Path("models/preprocessor.joblib")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, save_path)
    logger.info(f"Saved fitted preprocessor artifact to: {save_path}")

    # Also save feature names list for downstream explainability
    names_path = save_path.parent / "feature_names.joblib"
    joblib.dump(feature_names, names_path)
    logger.info(f"Saved feature names list to: {names_path}")

    return X_train_proc, X_val_proc, X_test_proc, preprocessor, feature_names


if __name__ == "__main__":
    from src.ml.data_loader import load_raw_dataset, split_data

    df = load_raw_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_p, X_val_p, X_test_p, preprocessor, feat_names = fit_and_transform_splits(X_train, X_val, X_test)
    print("Preprocessor test complete. Output shape:", X_train_p.shape)
