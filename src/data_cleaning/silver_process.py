# src/data/data_cleaner.py
import ast
from io import BytesIO, StringIO
from datetime import datetime

import boto3
import numpy as np
import pandas as pd

from config.settings import Settings
from utils.custom_exceptions import (
    CustomException,
    DataCleaningException,
    DataQualityException,
    S3UploadException,
)
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = Settings()


# ══════════════════════════════════════════════════════════════
# STEP 1 — Parse nested / stringified-dict columns
# ══════════════════════════════════════════════════════════════

def _safe_eval(val):
    if isinstance(val, dict):
        return val
    try:
        return ast.literal_eval(str(val))
    except Exception:
        return {}


def parse_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely evaluate the 4 stringified-dict columns that come
    straight out of the raw Excel files.
    """
    try:
        nested_cols = [
            "new_car_detail",
            "new_car_overview",
            "new_car_feature",
            "new_car_specs",
        ]
        for col in nested_cols:
            if col in df.columns:
                df[col] = df[col].apply(_safe_eval)
                logger.info(f"Parsed nested column: {col}")
            else:
                logger.warning(f"Column not found, skipping: {col}")
        return df
    except Exception as e:
        logger.error(f"Error parsing nested columns: {str(e)}")
        raise DataCleaningException(f"Error parsing nested columns: {str(e)}")


# ══════════════════════════════════════════════════════════════
# STEP 2 — Extract flat features from nested dicts
# ══════════════════════════════════════════════════════════════

def extract_car_detail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten new_car_detail dict → individual feature columns.
    """
    try:
        detail = df["new_car_detail"]

        df["Fuel_Type"]         = detail.apply(lambda x: x.get("ft",           np.nan))
        df["Body_Type"]         = detail.apply(lambda x: x.get("bt",           np.nan))
        df["Kms_Driven"]        = detail.apply(lambda x: x.get("km",           np.nan))
        df["Transmission_Type"] = detail.apply(lambda x: x.get("transmission", np.nan))
        df["Number_of_Owners"]  = detail.apply(lambda x: x.get("ownerNo",      np.nan))
        df["OEM"]               = detail.apply(lambda x: x.get("oem",          np.nan))
        df["Car_Model"]         = detail.apply(lambda x: x.get("model",        np.nan))
        df["Model_Year"]        = detail.apply(lambda x: x.get("modelYear",    np.nan))
        df["Variant_Name"]      = detail.apply(lambda x: x.get("variantName",  np.nan))
        df["Price_Raw"]         = detail.apply(lambda x: x.get("price",        np.nan))

        logger.info(f"Extracted car detail fields. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error extracting car detail: {str(e)}")
        raise DataCleaningException(f"Error extracting car detail: {str(e)}")


def extract_specs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Mileage / Engine / Max Power / Torque / Seats
    from the new_car_specs.top list.
    """
    try:
        TARGET_KEYS = {"Mileage", "Engine", "Max Power", "Torque", "Seats"}

        def _parse_specs(specs_dict):
            out = {k: np.nan for k in TARGET_KEYS}
            top = specs_dict.get("top", []) if isinstance(specs_dict, dict) else []
            if isinstance(top, list):
                for item in top:
                    if isinstance(item, dict) and item.get("key") in TARGET_KEYS:
                        out[item["key"]] = item.get("value", np.nan)
            return out

        specs_df = df["new_car_specs"].apply(_parse_specs).apply(pd.Series)
        specs_df = specs_df.rename(columns={"Max Power": "Max_Power"})
        df = pd.concat([df, specs_df], axis=1)

        logger.info(f"Extracted spec fields. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error extracting specs: {str(e)}")
        raise DataCleaningException(f"Error extracting specs: {str(e)}")


def drop_raw_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the original nested dict columns after extraction."""
    try:
        drop_cols = [
            "new_car_detail", "new_car_overview",
            "new_car_feature", "new_car_specs", "car_links",
        ]
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)
        logger.info(f"Dropped raw nested columns: {existing}")
        return df
    except Exception as e:
        logger.error(f"Error dropping nested columns: {str(e)}")
        raise DataCleaningException(f"Error dropping nested columns: {str(e)}")


# ══════════════════════════════════════════════════════════════
# STEP 3 — Convert & clean individual columns
# ══════════════════════════════════════════════════════════════

def convert_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price string  →  float in INR.
    '₹ 4.50 Lakh'  →  450000.0
    '₹ 1.2 Crore'  →  12000000.0
    """
    try:
        def _parse_price(val):
            if pd.isna(val):
                return np.nan
            parts = str(val).replace(",", "").strip().split()
            try:
                amount = float(parts[1])
                unit   = parts[2].lower() if len(parts) > 2 else "lakh"
                return amount * 1_00_00_000 if "crore" in unit else amount * 1_00_000
            except (IndexError, ValueError):
                return np.nan

        df["Price_INR"] = df["Price_Raw"].apply(_parse_price)
        logger.info(f"Converted price. Null count: {df['Price_INR'].isna().sum()}")
        return df
    except Exception as e:
        logger.error(f"Error converting price: {str(e)}")
        raise DataCleaningException(f"Error converting price: {str(e)}")


def strip_units(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip unit suffixes and cast to float.
    '1197 CC'    →  1197.0   (Engine)
    '82 BHP'     →  82.0     (Max_Power)
    '113 Nm'     →  113.0    (Torque)
    '21.5 kmpl'  →  21.5     (Mileage)
    '45,000 km'  →  45000.0  (Kms_Driven)
    '5 Seats'    →  5.0      (Seats)
    """
    try:
        unit_cols = ["Engine", "Max_Power", "Torque", "Mileage", "Kms_Driven", "Seats"]
        for col in unit_cols:
            if col in df.columns:
                before_nulls = df[col].isna().sum()
                df[col] = (
                    df[col].astype(str)
                           .str.replace(r"[^\d.]", "", regex=True)
                           .replace("", np.nan)
                           .pipe(pd.to_numeric, errors="coerce")
                )
                after_nulls = df[col].isna().sum()
                logger.info(
                    f"Stripped units: {col} | "
                    f"nulls before={before_nulls}, after={after_nulls}"
                )
        return df
    except Exception as e:
        logger.error(f"Error stripping units: {str(e)}")
        raise DataCleaningException(f"Error stripping units: {str(e)}")


def compute_age_of_car(df: pd.DataFrame) -> pd.DataFrame:
    """
    Model_Year  →  Age_of_Car  (current year − model year).
    Values outside 0–40 are set to NaN.
    """
    try:
        current_year     = datetime.now().year
        df["Model_Year"] = pd.to_numeric(df["Model_Year"], errors="coerce")
        df["Age_of_Car"] = current_year - df["Model_Year"]
        invalid          = ~df["Age_of_Car"].between(0, 40)
        df.loc[invalid, "Age_of_Car"] = np.nan
        logger.info(
            f"Computed Age_of_Car. "
            f"Range: {df['Age_of_Car'].min():.0f}–{df['Age_of_Car'].max():.0f} yrs | "
            f"Nulls: {df['Age_of_Car'].isna().sum()}"
        )
        return df
    except Exception as e:
        logger.error(f"Error computing age: {str(e)}")
        raise DataCleaningException(f"Error computing age: {str(e)}")


def cast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast Number_of_Owners and Seats to numeric."""
    try:
        for col in ["Number_of_Owners", "Seats"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                logger.info(f"Cast to numeric: {col}")
        return df
    except Exception as e:
        logger.error(f"Error casting numeric columns: {str(e)}")
        raise DataCleaningException(f"Error casting numeric columns: {str(e)}")


# ══════════════════════════════════════════════════════════════
# STEP 4 — Null handling & deduplication
# ══════════════════════════════════════════════════════════════

REQUIRED_COLS  = ["Price_INR", "Fuel_Type", "Body_Type", "Transmission_Type"]

NUMERIC_FILL   = [
    "Engine", "Max_Power", "Torque", "Mileage",
    "Kms_Driven", "Seats", "Number_of_Owners", "Age_of_Car",
]
CATEGORIC_FILL = ["Fuel_Type", "Body_Type", "Transmission_Type"]


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Drop rows where any REQUIRED column is null.
    2. Fill remaining numeric nulls with column median.
    3. Fill remaining categorical nulls with column mode.
    """
    try:
        before = len(df)
        df = df.dropna(subset=REQUIRED_COLS)
        logger.info(
            f"Dropped rows missing required cols: "
            f"{before - len(df):,} removed | {len(df):,} remaining"
        )

        for col in NUMERIC_FILL:
            if col in df.columns:
                n_null  = df[col].isna().sum()
                median  = df[col].median()
                df[col] = df[col].fillna(median)
                if n_null:
                    logger.info(f"Filled {n_null:,} nulls in '{col}' with median={median:.2f}")

        for col in CATEGORIC_FILL:
            if col in df.columns:
                n_null  = df[col].isna().sum()
                mode    = df[col].mode()
                if len(mode):
                    df[col] = df[col].fillna(mode[0])
                    if n_null:
                        logger.info(f"Filled {n_null:,} nulls in '{col}' with mode='{mode[0]}'")

        return df
    except Exception as e:
        logger.error(f"Error handling nulls: {str(e)}")
        raise DataCleaningException(f"Error handling nulls: {str(e)}")


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove fully duplicate rows."""
    try:
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {before - len(df):,} duplicate rows | {len(df):,} remaining")
        return df
    except Exception as e:
        logger.error(f"Error removing duplicates: {str(e)}")
        raise DataCleaningException(f"Error removing duplicates: {str(e)}")


# ══════════════════════════════════════════════════════════════
# STEP 5 — Select final Silver columns
# ══════════════════════════════════════════════════════════════

SILVER_COLUMNS = [
    "source_city",
    "OEM", "Car_Model", "Variant_Name",
    "Model_Year", "Age_of_Car",
    "Fuel_Type", "Body_Type", "Transmission_Type",
    "Kms_Driven", "Number_of_Owners",
    "Engine", "Max_Power", "Torque", "Mileage", "Seats",
    "Price_INR",
]


def select_silver_columns(df: pd.DataFrame) -> pd.DataFrame:
    try:
        present = [c for c in SILVER_COLUMNS if c in df.columns]
        missing = [c for c in SILVER_COLUMNS if c not in df.columns]
        if missing:
            logger.warning(f"Silver columns not found, skipped: {missing}")
        df = df[present].copy()
        logger.info(f"Selected {len(present)} Silver columns. Final shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error selecting Silver columns: {str(e)}")
        raise DataCleaningException(f"Error selecting Silver columns: {str(e)}")


# ══════════════════════════════════════════════════════════════
# STEP 6 — Data quality check
# ══════════════════════════════════════════════════════════════

def run_quality_checks(df: pd.DataFrame, raw_row_count: int) -> None:
    """
    Assert Silver data meets quality thresholds.
    Raises DataQualityException if any check fails.
    """
    try:
        errors = []

        # 1. Retain at least 60% of raw rows
        retention = len(df) / raw_row_count if raw_row_count else 0
        if retention < 0.60:
            errors.append(f"Row retention {retention:.1%} < 60% threshold")

        # 2. No column should have > 20% nulls
        bad_null = df.isnull().mean()
        bad_null = bad_null[bad_null > 0.20].to_dict()
        if bad_null:
            errors.append(f"High-null columns (>20%): {bad_null}")

        # 3. Price must be positive
        neg_price = (df["Price_INR"] <= 0).sum()
        if neg_price:
            errors.append(f"{neg_price} rows with Price_INR ≤ 0")

        # 4. Age_of_Car must be in valid range
        bad_age = (~df["Age_of_Car"].between(0, 40)).sum()
        if bad_age:
            errors.append(f"{bad_age} rows with Age_of_Car out of range 0–40")

        if errors:
            raise DataQualityException(
                "Data quality checks FAILED:\n" + "\n".join(f"  • {e}" for e in errors)
            )

        logger.info(
            f"Data quality checks PASSED ✓ | "
            f"Rows: {len(df):,} | Retention: {retention:.1%} | "
            f"Cols: {df.shape[1]}"
        )
    except DataQualityException:
        raise
    except Exception as e:
        logger.error(f"Error running quality checks: {str(e)}")
        raise DataQualityException(f"Error running quality checks: {str(e)}")


# ══════════════════════════════════════════════════════════════
# STEP 7 — Save to S3 Silver
# ══════════════════════════════════════════════════════════════

def save_to_silver(df: pd.DataFrame) -> None:
    """
    Upload cleaned DataFrame to S3 Silver layer as:
      • Parquet  (primary — used by downstream ML tasks)
      • CSV      (secondary — easy to inspect in S3 console)
    """
    try:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region,
        )
        bucket = settings.s3_bucket

        # ── Parquet ──────────────────────────────────────────
        parquet_buf = BytesIO()
        df.to_parquet(parquet_buf, index=False, engine="pyarrow")
        parquet_buf.seek(0)
        s3_client.put_object(
            Bucket=bucket,
            Key=settings.silver_parquet,
            Body=parquet_buf.getvalue(),
            ContentType="application/octet-stream",
        )
        logger.info(f"Saved Parquet → s3://{bucket}/{settings.silver_parquet}")

        # ── CSV ──────────────────────────────────────────────
        csv_buf = StringIO()
        df.to_csv(csv_buf, index=False)
        s3_client.put_object(
            Bucket=bucket,
            Key=settings.silver_csv,
            Body=csv_buf.getvalue(),
            ContentType="text/csv",
        )
        logger.info(f"Saved CSV     → s3://{bucket}/{settings.silver_csv}")

    except Exception as e:
        logger.error(f"Error saving to Silver: {str(e)}")
        raise S3UploadException(f"Error saving to Silver: {str(e)}")


# ══════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT — called by script & Airflow task
# ══════════════════════════════════════════════════════════════

def clean_and_save_silver(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts the raw combined DataFrame from data_loader_s3(),
    runs the full Bronze → Silver cleaning pipeline,
    saves to S3, and returns the clean DataFrame.

    Usage:
        from src.data.data_loader  import data_loader_s3
        from src.data.data_cleaner import clean_and_save_silver

        raw_df   = data_loader_s3()
        clean_df = clean_and_save_silver(raw_df)
    """
    try:
        logger.info("══ Bronze → Silver Cleaning START ══")
        raw_row_count = len(df)
        logger.info(f"Input shape: {df.shape}")

        # Parse nested JSON columns
        df = parse_nested_columns(df)
        df = extract_car_detail(df)
        df = extract_specs(df)
        df = drop_raw_nested_columns(df)

        # Clean individual columns
        df = convert_price(df)
        df = strip_units(df)
        df = compute_age_of_car(df)
        df = cast_numeric_columns(df)

        # Nulls & duplicates
        df = handle_nulls(df)
        df = remove_duplicates(df)

        # Final column selection
        df = select_silver_columns(df)

        # Quality gate
        run_quality_checks(df, raw_row_count)

        # Save
        save_to_silver(df)

        logger.info(f"══ Bronze → Silver Cleaning DONE  ══ | Final shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Bronze → Silver pipeline failed: {str(e)}")
        raise CustomException(f"Bronze → Silver pipeline failed: {str(e)}")
    
from src.data_ingestion.data_loader import data_loader_s3

if __name__ == "__main__":
    print("🚀 Starting Silver Pipeline...")

    # Step 1: Load data
    df = data_loader_s3()
    print("Loaded Data:", df.shape)

    # Step 2: Run cleaning pipeline
    df = clean_and_save_silver(df)

    print("✅ Final Data Shape:", df.shape)