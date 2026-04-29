from src.data_ingestion.data_loader import data_loader_s3

def main():
    df = data_loader_s3()

    print("\n✅ Data Loaded Successfully")
    print("Shape:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()