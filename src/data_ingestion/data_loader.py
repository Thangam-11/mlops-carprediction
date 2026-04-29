import boto3
import pandas as pd
from io import BytesIO
from config.settings import Settings
from utils.custom_exceptions import CustomException
from utils.logger_exceptions import get_logger

logger = get_logger(__name__)
settings = Settings()


def data_loader_s3() -> pd.DataFrame:
    """
    Loads all Excel files from S3 Bronze layer,
    combines them into one DataFrame.
    """

    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region_name=settings.aws_region
        )

        bucket = settings.s3_bucket
        prefix = settings.bronze_path

        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        dfs = []

        for obj in response.get('Contents', []):
            key = obj['Key']

            if key.endswith(".xlsx"):
                logger.info(f"Reading file: {key}")

                file_obj = s3_client.get_object(Bucket=bucket, Key=key)
                df_temp = pd.read_excel(BytesIO(file_obj['Body'].read()))

                logger.info(f"Loaded shape: {df_temp.shape}")

                dfs.append(df_temp)

        if not dfs:
            raise ValueError("No Excel files found in S3 path")

        # 🔥 Combine all files
        final_df = pd.concat(dfs, ignore_index=True)

        logger.info(f"Final combined shape: {final_df.shape}")

        return final_df

    except Exception as e:
        logger.error(f"Error loading data from S3: {str(e)}")
        raise CustomException(f"Error loading data from S3: {str(e)}")