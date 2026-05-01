# src/data/silver_to_gold.py
"""
Silver → Gold Pipeline
Reads cleaned Parquet from S3 Silver,
encodes categorical columns,
saves ML-ready data to S3 Gold.

Run locally:
    python -m src.data.silver_to_gold
"""

import io
import pickle

import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config.settings import Settings
from utils.custom_exceptions import (
    CustomException,
    DataCleaningException,
    DataQualityException,
    S3UploadException,
)
from utils.logger_exceptions import get_logger

# remove the outliars in the price column and save the cleaned data to silver layer in s3

# ─────────────────────────────────────────────────────────────
# Silver SILVER_COLUMNS (from data_cleaner.py):
#   source_city, OEM, Car_Model, Variant_Name,
#   Model_Year, Age_of_Car,
#   Fuel_Type, Body_Type, Transmission_Type,
#   Kms_Driven, Number_of_Owners,
#   Engine, Max_Power, Torque, Mileage, Seats,
#   Price_INR
# ─────────────────────────────────────────────────────────────

# High-cardinality cols → Frequency Encoding
HIGH_CARDINALITY_GOLD = ["OEM", "Car_Model", "Variant_Name"]

# Low-cardinality cols → Label Encoding
LOW_CARDINALITY_GOLD = ["Fuel_Type", "Body_Type", "Transmission_Type",]

# All categorical cols
CATEGORICAL_COLS_GOLD = LOW_CARDINALITY_GOLD + HIGH_CARDINALITY_GOLD


# ─────────────────────────────────────────────────────────────
# STEP 1 — Load from S3 Silver
# ─────────────────────────────────────────────────────────────

def load_silver(s3_client, bucket: str, silver_key: str, logger) -> pd.DataFrame:
    """Read cleaned Parquet from S3 Silver layer."""
    logger.info(f"Loading Silver data from s3://{bucket}/{silver_key} ...")
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=silver_key)
        df  = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        logger.info(f"  Loaded shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"Error loading Silver data: {e}")
        raise S3UploadException(f"Error loading Silver data: {e}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — Fill nulls in categorical cols before encoding
# ─────────────────────────────────────────────────────────────

def fill_categorical_nulls(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Fill nulls in categorical columns with 'Unknown'."""
    logger.info("Filling nulls in categorical columns...")
    for col in CATEGORICAL_COLS_GOLD:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                df[col] = df[col].fillna("Unknown")
                logger.info(f"  {col}: filled {n:,} nulls with 'Unknown'")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 3 — Label Encoding  (low-cardinality cols)
#          Fuel_Type, Body_Type, Transmission_Type, source_city
# ─────────────────────────────────────────────────────────────

def label_encode(df: pd.DataFrame, logger) -> tuple[pd.DataFrame, dict]:
    """
    Apply LabelEncoder to low-cardinality columns.
    Returns df with new _encoded columns + dict of fitted encoders.
    Encoded column name: <col>_encoded  e.g. Fuel_Type_encoded
    """
    logger.info("Applying Label Encoding to low-cardinality columns...")
    encoders = {}

    for col in LOW_CARDINALITY_GOLD:
        if col not in df.columns:
            logger.warning(f"  Skipping {col} — not found in DataFrame")
            continue

        le      = LabelEncoder()
        enc_col = f"{col}_encoded"
        df[enc_col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        logger.info(f"  {col} -> {enc_col} | unique={len(le.classes_)} | mapping={mapping}")

    return df, encoders


# ─────────────────────────────────────────────────────────────
# STEP 4 — Frequency Encoding  (high-cardinality cols)
#          OEM, Car_Model, Variant_Name
#
#          Why frequency encoding for high-cardinality?
#          OEM alone can have 50+ unique values.
#          Label encoding creates arbitrary ordinal ranks.
#          Frequency encoding replaces each category with how
#          often it appears in the data — preserves real signal.
# ─────────────────────────────────────────────────────────────

def frequency_encode(df: pd.DataFrame, logger) -> tuple[pd.DataFrame, dict]:
    """
    Replace each category with its frequency (proportion) in the dataset.
    Encoded column name: <col>_freq  e.g. OEM_freq
    Returns df + dict of frequency maps (needed at inference time).
    """
    logger.info("Applying Frequency Encoding to high-cardinality columns...")
    freq_maps = {}

    for col in HIGH_CARDINALITY_GOLD:
        if col not in df.columns:
            logger.warning(f"  Skipping {col} — not found in DataFrame")
            continue

        freq_map  = df[col].value_counts(normalize=True).to_dict()
        freq_col  = f"{col}_freq"
        df[freq_col]   = df[col].map(freq_map).fillna(0)
        freq_maps[col] = freq_map

        logger.info(
            f"  {col} -> {freq_col} | unique={df[col].nunique()} | "
            f"min_freq={df[freq_col].min():.4f} max_freq={df[freq_col].max():.4f}"
        )

    return df, freq_maps


# ─────────────────────────────────────────────────────────────
# STEP 5 — Select final Gold columns
#
#   Numeric names match Silver exactly:
#     Engine, Max_Power, Torque, Mileage, Kms_Driven,
#     Number_of_Owners, Age_of_Car, Seats, Model_Year
# ─────────────────────────────────────────────────────────────

GOLD_COLUMNS = [
    # Numeric features (exact Silver column names)
    "Age_of_Car",
    "Model_Year",
    "Kms_Driven",
    "Number_of_Owners",
    "Engine",
    "Max_Power",
    "Torque",
    "Mileage",
    "Seats",

    # Label-encoded columns
    "Fuel_Type_encoded",
    "Body_Type_encoded",
    "Transmission_Type_encoded",
    "source_city_encoded",

    # Frequency-encoded columns
    "OEM_freq",
    "Car_Model_freq",
    "Variant_Name_freq",

    # Original categorical (kept for reference / explainability)
    "Fuel_Type", "Body_Type", "Transmission_Type",
    "OEM", "Car_Model", "Variant_Name", "source_city",

    # Target
    "Price_INR",
]


def select_gold_columns(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Keep only the final Gold columns."""
    logger.info("Selecting final Gold columns...")
    present = [c for c in GOLD_COLUMNS if c in df.columns]
    missing = [c for c in GOLD_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(f"  Columns not found, skipped: {missing}")
    df = df[present].copy()
    logger.info(f"  {len(present)} columns selected. Final shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 6 — Save encoders to S3  (needed at inference time)
# ─────────────────────────────────────────────────────────────

def save_encoders_to_s3(s3_client, bucket: str, encoders: dict,
                         freq_maps: dict, logger) -> None:
    """
    Pickle and upload both encoder dicts to S3 Gold so the
    prediction API can apply the same encoding at inference time.
    """
    logger.info("Saving encoders to S3 Gold...")

    artifact = {"label_encoders": encoders, "frequency_maps": freq_maps}
    buf = io.BytesIO()
    pickle.dump(artifact, buf)
    buf.seek(0)

    key = "gold/encoders/categorical_encoders.pkl"
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.info(f"  Saved encoders -> s3://{bucket}/{key}")


# ─────────────────────────────────────────────────────────────
# STEP 7 — Save Gold data to S3
# ─────────────────────────────────────────────────────────────

def save_to_gold(s3_client, bucket: str, df: pd.DataFrame, logger) -> None:
    """Save Gold DataFrame as Parquet + CSV to S3 Gold layer."""
    logger.info(f"Saving Gold data to s3://{bucket}/gold/ ...")

    # Parquet
    parquet_buf = io.BytesIO()
    df.to_parquet(parquet_buf, index=False, engine="pyarrow")
    parquet_buf.seek(0)
    s3_client.put_object(
        Bucket=bucket,
        Key="gold/cars_gold.parquet",
        Body=parquet_buf.getvalue(),
    )
    logger.info(f"  Saved Parquet -> s3://{bucket}/gold/cars_gold.parquet")

    # CSV
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    s3_client.put_object(
        Bucket=bucket,
        Key="gold/cars_gold.csv",
        Body=csv_buf.getvalue(),
    )
    logger.info(f"  Saved CSV     -> s3://{bucket}/gold/cars_gold.csv")


# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE — called by script & Airflow task
# ─────────────────────────────────────────────────────────────

def run_silver_to_gold(logger=None) -> pd.DataFrame:
    """
    Full Silver -> Gold encoding pipeline.
    Reads from S3 Silver, encodes categoricals, saves to S3 Gold.

    Usage:
        from src.data.silver_to_gold import run_silver_to_gold
        gold_df = run_silver_to_gold()
    """
    if logger is None:
        logger = get_logger(__name__)

    settings = Settings()

    # Settings paths match data_cleaner.py usage exactly
    bucket     = settings.s3_bucket
    silver_key = settings.silver_parquet   # e.g. "silver/cars_cleaned.parquet"

    s3_client = boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
    )

    logger.info("=== Silver -> Gold Pipeline START ===")

    # 1. Load Silver
    df = load_silver(s3_client, bucket, silver_key, logger)

    # 2. Fill categorical nulls
    df = fill_categorical_nulls(df, logger)

    # 3. Label encode low-cardinality cols
    df, label_encoders = label_encode(df, logger)

    # 4. Frequency encode high-cardinality cols
    df, freq_maps = frequency_encode(df, logger)

    # 5. Select final Gold columns
    df = select_gold_columns(df, logger)

    # 6. Save encoders (for inference)
    save_encoders_to_s3(s3_client, bucket, label_encoders, freq_maps, logger)

    # 7. Save Gold data
    save_to_gold(s3_client, bucket, df, logger)

    logger.info(f"=== Silver -> Gold Pipeline DONE | Final shape: {df.shape} ===")
    return df


# ─────────────────────────────────────────────────────────────
# Run as standalone script
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger  = get_logger(__name__)
    gold_df = run_silver_to_gold(logger)

    logger.info("\n── Gold DataFrame Sample ──")
    logger.info(f"\n{gold_df.head(3).to_string()}")

    logger.info("\n── Encoded Column Value Counts ──")
    for col in ["Fuel_Type_encoded", "Body_Type_encoded",
                "Transmission_Type_encoded", "source_city_encoded"]:
        if col in gold_df.columns:
            logger.info(f"\n{col}:\n{gold_df[col].value_counts().to_string()}")

    logger.info("\n── Frequency Encoded Stats ──")
    for col in ["OEM_freq", "Car_Model_freq", "Variant_Name_freq"]:
        if col in gold_df.columns:
            logger.info(
                f"{col}: min={gold_df[col].min():.4f}  "
                f"mean={gold_df[col].mean():.4f}  "
                f"max={gold_df[col].max():.4f}"
            )