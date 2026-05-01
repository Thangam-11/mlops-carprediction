# src/models/model_training.py
"""
Model Training Pipeline — config-driven + MLflow Tracking
All hyperparameters come from config/model_config.yaml.
All paths/credentials come from config/settings.py (.env).
Nothing is hardcoded here.

Run locally:
    python -m src.models.model_training
"""

import io
import json
import pickle

import boto3
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd

from sklearn.ensemble        import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model    import Ridge
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost                 import XGBRegressor

from config.settings         import get_model_config, get_settings
from utils.custom_exceptions import CustomException, S3UploadException
from utils.logger_exceptions import get_logger

logger   = get_logger(__name__)
settings = get_settings()
cfg      = get_model_config()          # full validated YAML config
train_cfg = cfg.training               # shorthand for training section
models_cfg = cfg.models                # shorthand for models section


# ══════════════════════════════════════════════════════════════
# S3 CLIENT
# ══════════════════════════════════════════════════════════════

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id     = settings.aws_access_key_id,
        aws_secret_access_key = settings.aws_secret_access_key,
        region_name           = settings.aws_region,
    )


# ══════════════════════════════════════════════════════════════
# STEP 1 — Load Gold Parquet from S3
# ══════════════════════════════════════════════════════════════

def load_gold(s3_client) -> pd.DataFrame:
    """Read ML-ready Parquet from S3 Gold (path from settings)."""
    key = settings.gold_parquet
    logger.info(f"Loading Gold data from s3://{settings.s3_bucket}/{key} ...")
    try:
        obj = s3_client.get_object(Bucket=settings.s3_bucket, Key=key)
        df  = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        logger.info(f"  Loaded shape : {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading Gold data: {e}")
        raise S3UploadException(f"Error loading Gold data: {e}")


# ══════════════════════════════════════════════════════════════
# STEP 2 — Prepare Features & Target
# ══════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Uses train_cfg.feature_cols and train_cfg.target_col
    — both come from model_config.yaml, not hardcoded.
    """
    logger.info("Preparing features and target...")

    missing = [c for c in train_cfg.feature_cols if c not in df.columns]
    if missing:
        raise CustomException(f"Missing feature columns in Gold data: {missing}")

    X = df[train_cfg.feature_cols].copy()
    y = df[train_cfg.target_col].copy()

    x_nulls = X.isnull().sum().sum()
    y_nulls = y.isnull().sum()
    if x_nulls or y_nulls:
        raise CustomException(
            f"Nulls detected — X: {x_nulls} cells | y: {y_nulls} rows. "
            "Re-run Gold pipeline before training."
        )

    logger.info(f"  X shape : {X.shape}")
    logger.info(f"  y stats : min={y.min():,.0f}  mean={y.mean():,.0f}  max={y.max():,.0f}")
    return X, y


# ══════════════════════════════════════════════════════════════
# STEP 3 — Train / Test Split
# ══════════════════════════════════════════════════════════════

def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    test_size and random_state come from train_cfg (model_config.yaml).
    Stratified by price quantile bucket for consistent distribution.
    """
    logger.info(
        f"Splitting — test_size={train_cfg.test_size} | "
        f"random_state={train_cfg.random_state}"
    )
    y_buckets = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = train_cfg.test_size,
        random_state = train_cfg.random_state,
        stratify     = y_buckets,
    )
    logger.info(f"  Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════
# STEP 4 — Build Models from YAML config
# ══════════════════════════════════════════════════════════════

def get_models() -> dict:
    """
    Instantiate all models using params from models_cfg (model_config.yaml).
    Returns dict of model_name → (model_instance, params_dict).
    params_dict is the raw dict for MLflow logging.
    """
    ridge_p = models_cfg.Ridge.model_dump()
    rf_p    = models_cfg.RandomForest.model_dump()
    gb_p    = models_cfg.GradientBoosting.model_dump()
    xgb_p   = models_cfg.XGBoost.model_dump()

    return {
        "Ridge":            (Ridge(**ridge_p),                    ridge_p),
        "RandomForest":     (RandomForestRegressor(**rf_p),       rf_p),
        "GradientBoosting": (GradientBoostingRegressor(**gb_p),   gb_p),
        "XGBoost":          (XGBRegressor(**xgb_p),               xgb_p),
    }


# ══════════════════════════════════════════════════════════════
# STEP 5 — Compute Metrics
# ══════════════════════════════════════════════════════════════

def _compute_metrics(model, X_test, y_test) -> dict:
    """Return MAE, RMSE, R², MAPE for a fitted model."""
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2   = r2_score(y_test, y_pred)
    mask = y_test != 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)

    return {
        "MAE"      : round(mae,  2),
        "RMSE"     : round(rmse, 2),
        "R2"       : round(r2,   6),
        "MAPE_pct" : round(mape, 4),
    }


# ══════════════════════════════════════════════════════════════
# STEP 6 — Train All Models, Log Each to MLflow
# ══════════════════════════════════════════════════════════════

def train_and_evaluate(
    X_train, X_test, y_train, y_test
) -> tuple[dict, dict]:
    """
    For every model:
      1. Open a child MLflow run
      2. Log hyperparams (from YAML via models_cfg)
      3. Fit
      4. Log metrics
      5. Log model artifact
    """
    models        = get_models()
    fitted_models = {}
    all_metrics   = {}

    logger.info("── Training Models ──────────────────────────────")

    for name, (model, params) in models.items():
        logger.info(f"  Training {name}...")

        with mlflow.start_run(run_name=name, nested=True) as run:

            # Log hyperparams from YAML
            mlflow.log_params(params)
            mlflow.log_param("model_name",   name)
            mlflow.log_param("train_rows",   len(X_train))
            mlflow.log_param("test_rows",    len(X_test))
            mlflow.log_param("n_features",   len(train_cfg.feature_cols))

            # Fit
            model.fit(X_train, y_train)

            # Metrics
            metrics = _compute_metrics(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # Log model artifact
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")

            metrics["run_id"] = run.info.run_id

            fitted_models[name] = model
            all_metrics[name]   = {"model_name": name, **metrics}

            logger.info(
                f"  [{name}] MAE={metrics['MAE']:,.0f}  RMSE={metrics['RMSE']:,.0f}  "
                f"R²={metrics['R2']:.4f}  MAPE={metrics['MAPE_pct']:.2f}%  "
                f"run_id={run.info.run_id}"
            )

    return fitted_models, all_metrics


# ══════════════════════════════════════════════════════════════
# STEP 7 — Select Best Model  (lowest RMSE)
# ══════════════════════════════════════════════════════════════

def select_best_model(
    fitted_models: dict, all_metrics: dict
) -> tuple[str, object, dict]:
    best_name = min(all_metrics, key=lambda n: all_metrics[n]["RMSE"])
    logger.info(
        f"Best model → {best_name} | RMSE={all_metrics[best_name]['RMSE']:,.0f}"
    )
    return best_name, fitted_models[best_name], all_metrics[best_name]


# ══════════════════════════════════════════════════════════════
# STEP 8 — Cross-Validation on Best Model
# ══════════════════════════════════════════════════════════════

def cross_validate_best(
    best_model, X: pd.DataFrame, y: pd.Series, best_name: str
) -> dict:
    """
    cv_folds comes from train_cfg (model_config.yaml).
    Results logged to parent MLflow run.
    """
    logger.info(
        f"Running {train_cfg.cv_folds}-fold CV on {best_name}..."
    )
    cv_r2   = cross_val_score(
        best_model, X, y,
        cv      = train_cfg.cv_folds,
        scoring = "r2",
        n_jobs  = -1,
    )
    cv_rmse = -cross_val_score(
        best_model, X, y,
        cv      = train_cfg.cv_folds,
        scoring = "neg_root_mean_squared_error",
        n_jobs  = -1,
    )
    cv_results = {
        "cv_r2_mean"   : round(float(cv_r2.mean()),   4),
        "cv_r2_std"    : round(float(cv_r2.std()),     4),
        "cv_rmse_mean" : round(float(cv_rmse.mean()),  2),
        "cv_rmse_std"  : round(float(cv_rmse.std()),   2),
    }
    logger.info(f"  CV R²  : {cv_results['cv_r2_mean']} ± {cv_results['cv_r2_std']}")
    logger.info(f"  CV RMSE: {cv_results['cv_rmse_mean']:,.0f} ± {cv_results['cv_rmse_std']:,.0f}")
    return cv_results


# ══════════════════════════════════════════════════════════════
# STEP 9 — Save to S3
# ══════════════════════════════════════════════════════════════

def _model_folder() -> str:
    parts = settings.model_output_path.split("/")
    return "/".join(parts[:-1]) + "/"    # e.g. "model_output/"


def save_model_to_s3(
    s3_client,
    best_model,
    best_name: str,
    all_metrics: dict,
    cv_results: dict,
) -> None:
    """
    Saves to S3 — all paths from settings, nothing hardcoded:
      model_output/model.pkl
      model_output/all_metrics.json
      model_output/model_metadata.json
    """
    bucket = settings.s3_bucket
    folder = _model_folder()

    logger.info(f"Saving to s3://{bucket}/{folder} ...")

    # Model pickle
    model_buf = io.BytesIO()
    pickle.dump(best_model, model_buf)
    model_buf.seek(0)
    s3_client.put_object(
        Bucket = bucket,
        Key    = settings.model_output_path,
        Body   = model_buf.getvalue(),
    )
    logger.info(f"  Saved model    -> s3://{bucket}/{settings.model_output_path}")

    # All metrics
    metrics_key = f"{folder}all_metrics.json"
    s3_client.put_object(
        Bucket      = bucket,
        Key         = metrics_key,
        Body        = json.dumps(all_metrics, indent=2),
        ContentType = "application/json",
    )
    logger.info(f"  Saved metrics  -> s3://{bucket}/{metrics_key}")

    # Metadata — includes feature list and CV results from YAML config
    metadata = {
        "best_model"    : best_name,
        "feature_cols"  : train_cfg.feature_cols,
        "target_col"    : train_cfg.target_col,
        "cv_results"    : cv_results,
        "test_size"     : train_cfg.test_size,
        "random_state"  : train_cfg.random_state,
        "cv_folds"      : train_cfg.cv_folds,
        "model_config"  : models_cfg.model_dump(),     # full YAML params snapshot
    }
    metadata_key = f"{folder}model_metadata.json"
    s3_client.put_object(
        Bucket      = bucket,
        Key         = metadata_key,
        Body        = json.dumps(metadata, indent=2),
        ContentType = "application/json",
    )
    logger.info(f"  Saved metadata -> s3://{bucket}/{metadata_key}")


# ══════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════

def run_model_training() -> dict:
    """
    Full config-driven pipeline:
      • All hyperparams    ← config/model_config.yaml
      • All paths/creds    ← .env via config/settings.py
      • Nothing hardcoded in this file

    MLflow run hierarchy:
      [Parent]  training_pipeline
          ├── [Child]  Ridge
          ├── [Child]  RandomForest
          ├── [Child]  GradientBoosting
          └── [Child]  XGBoost

    Usage:
        from src.models.model_training import run_model_training
        metrics = run_model_training()
    """
    try:
        logger.info("══ Model Training Pipeline START ══")
        logger.info(f"  S3 bucket      : {settings.s3_bucket}")
        logger.info(f"  Gold path      : {settings.gold_path}")
        logger.info(f"  Model output   : {settings.model_output_path}")
        logger.info(f"  MLflow URI     : {settings.mlflow_tracking_uri}")
        logger.info(f"  MLflow exp     : {train_cfg.experiment_name}")
        logger.info(f"  Config path    : {settings.model_config_path}")

        s3_client = get_s3_client()

        # Set MLflow — URI from settings, experiment name from YAML
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(train_cfg.experiment_name)

        with mlflow.start_run(run_name="training_pipeline") as parent_run:

            logger.info(f"  MLflow parent run_id: {parent_run.info.run_id}")

            # Log dataset + config-level params on the parent run
            mlflow.log_params({
                "gold_path"    : settings.gold_path,
                "s3_bucket"    : settings.s3_bucket,
                "test_size"    : train_cfg.test_size,
                "random_state" : train_cfg.random_state,
                "cv_folds"     : train_cfg.cv_folds,
                "n_features"   : len(train_cfg.feature_cols),
                "target_col"   : train_cfg.target_col,
            })

            # Log the YAML config file itself as an artifact
            mlflow.log_artifact(settings.model_config_path, artifact_path="config")

            # 1. Load
            df = load_gold(s3_client)
            mlflow.log_param("total_rows", len(df))

            # 2. Features & target
            X, y = prepare_features(df)

            # 3. Split
            X_train, X_test, y_train, y_test = split_data(X, y)

            # 4 & 5. Train all models
            fitted_models, all_metrics = train_and_evaluate(
                X_train, X_test, y_train, y_test
            )

            # Log comparison table
            logger.info("── Model Comparison ─────────────────────────────")
            for name, m in all_metrics.items():
                logger.info(
                    f"  {name:<20} MAE={m['MAE']:>12,.0f}  "
                    f"RMSE={m['RMSE']:>12,.0f}  R²={m['R2']:.4f}  "
                    f"MAPE={m['MAPE_pct']:.2f}%"
                )

            # 6. Select best
            best_name, best_model, best_metrics = select_best_model(
                fitted_models, all_metrics
            )

            # 7. Cross-validate — cv_folds from YAML
            cv_results = cross_validate_best(best_model, X, y, best_name)

            # Log CV + best summary on parent run
            mlflow.log_metrics({
                "best_cv_r2_mean"   : cv_results["cv_r2_mean"],
                "best_cv_r2_std"    : cv_results["cv_r2_std"],
                "best_cv_rmse_mean" : cv_results["cv_rmse_mean"],
                "best_cv_rmse_std"  : cv_results["cv_rmse_std"],
                "best_MAE"          : best_metrics["MAE"],
                "best_RMSE"         : best_metrics["RMSE"],
                "best_R2"           : best_metrics["R2"],
                "best_MAPE_pct"     : best_metrics["MAPE_pct"],
            })
            mlflow.log_param("best_model", best_name)
            mlflow.log_text(
                "\n".join(train_cfg.feature_cols),
                artifact_file="feature_cols.txt",
            )

            # 8. Register best model in MLflow Model Registry
            best_run_id = all_metrics[best_name]["run_id"]
            registered  = mlflow.register_model(
                model_uri = f"runs:/{best_run_id}/model",
                name      = f"used_car_price_{best_name}",
            )
            logger.info(
                f"  Registered MLflow model: {registered.name} v{registered.version}"
            )

            # 9. Save to S3
            save_model_to_s3(
                s3_client,
                best_model,
                best_name,
                all_metrics,
                cv_results,
            )

        logger.info("══ Model Training Pipeline DONE ══")
        return best_metrics

    except Exception as e:
        logger.error(f"Model training pipeline failed: {str(e)}")
        raise CustomException(f"Model training pipeline failed: {str(e)}")


# ══════════════════════════════════════════════════════════════
# Run as standalone script
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    metrics = run_model_training()
    print("\n✅ Best Model Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")