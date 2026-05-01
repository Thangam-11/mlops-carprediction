# check_prices.py
import io, boto3, pandas as pd
from config.settings import get_settings

settings  = get_settings()
s3_client = boto3.client(
    "s3",
    aws_access_key_id     = settings.aws_access_key_id,
    aws_secret_access_key = settings.aws_secret_access_key,
    region_name           = settings.aws_region,
)

obj = s3_client.get_object(Bucket=settings.s3_bucket, Key=settings.silver_parquet)
df  = pd.read_parquet(io.BytesIO(obj["Body"].read()))

print("Price_INR stats:")
print(df["Price_INR"].describe())
print(f"\nTop 10 highest prices:")
print(df["Price_INR"].nlargest(10))
print(f"\nSample Price_Raw values:")
print(df["Price_Raw"].head(20))