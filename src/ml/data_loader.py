"""Data loading and stratified splitting for Customer360 ML pipeline.

Loads the customer churn dataset, isolates the target variable, excludes
uninformative identifiers, and performs stratified 70/15/15 train/val/test splits.
"""

import sys
import logging
from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Customer360.DataLoader")


def get_data_path() -> Path:
    """Locate customer_churn.csv across known project paths."""
    candidate_paths = [
        Path("data/raw/customer_churn.csv"),
        Path("../data/raw/customer_churn.csv"),
        Path(__file__).resolve().parents[2] / "data" / "raw" / "customer_churn.csv"
    ]
    for p in candidate_paths:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Could not find 'data/raw/customer_churn.csv'. Please check directory structure.")


def load_raw_dataset(csv_path: Path = None) -> pd.DataFrame:
    """Load the raw customer churn dataset without modifying original file."""
    if csv_path is None:
        csv_path = get_data_path()

    logger.info(f"Loading raw dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Perform leak-free stratified Train/Validation/Test splitting.

    Parameters:
        df: Input DataFrame containing all features, customerID, and Churn.
        test_size: Proportion of dataset allocated to the final holdout test set (default: 0.15).
        val_size: Proportion of dataset allocated to the validation set (default: 0.15).
        random_state: Fixed random seed for complete reproducibility (default: 42).

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"Initiating stratified split with random_state={random_state}...")

    # 1. Encode Target: 'Yes' -> 1 (Churned), 'No' -> 0 (Retained)
    if 'Churn' not in df.columns:
        raise KeyError("Target column 'Churn' not found in dataset.")

    y = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # 2. Separate Features: Strictly exclude customerID and target Churn
    drop_cols = ['Churn']
    if 'customerID' in df.columns:
        drop_cols.append('customerID')
        logger.info("Excluded 'customerID' from predictive feature matrix to prevent identity leakage.")

    X = df.drop(columns=drop_cols).copy()

    # 3. First Split: Separate holdout Test set (e.g. 15%)
    # Remaining goes to train + validation (85%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 4. Second Split: From the 85% train_val, split into Train and Validation
    # To achieve exactly val_size relative to the original dataset:
    # val_ratio_relative = val_size / (1.0 - test_size)
    val_ratio_adjusted = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_train_val
    )

    # 5. Log verification metrics
    total_samples = len(df)
    logger.info("--- Stratified Split Summary ---")
    logger.info(f"Train Set     : {len(X_train):,} samples ({len(X_train)/total_samples*100:.1f}%) | Churn Rate: {y_train.mean()*100:.2f}%")
    logger.info(f"Validation Set: {len(X_val):,} samples ({len(X_val)/total_samples*100:.1f}%) | Churn Rate: {y_val.mean()*100:.2f}%")
    logger.info(f"Test Set      : {len(X_test):,} samples ({len(X_test)/total_samples*100:.1f}%) | Churn Rate: {y_test.mean()*100:.2f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df = load_raw_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print("DataLoader verification complete.")
