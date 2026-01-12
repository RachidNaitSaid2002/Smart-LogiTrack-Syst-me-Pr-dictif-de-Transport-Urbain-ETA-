import os
import gdown
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from dotenv import load_dotenv

load_dotenv()

# Initialize Spark with optimized configurations
spark = SparkSession.builder \
    .appName("LogiTrack-Optimization") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

# Use Constants for Paths
BASE_PATH = "/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/Test"
BRONZE_PATH = f"{BASE_PATH}/Bronze"
SILVER_PATH = f"{BASE_PATH}/Silver"

def load_and_save_bronze(file_id: str, filename: str = "data.parquet"):
    output_path = f"{BASE_PATH}/Data/{filename}"
    if not os.path.exists(output_path):
        gdown.download(id=file_id, output=output_path, quiet=False)
    
    df = spark.read.parquet(output_path)
    df.write.mode('overwrite').parquet(BRONZE_PATH)
    return df

def clean_data_optimized(df):
    # 1. Deduplication (Single Action)
    df = df.dropDuplicates()

    # 2. Time Engineering & Outlier Filtering (Combined to reduce passes)
    df = df.withColumn("Durée_minutes", 
             (F.col("tpep_dropoff_datetime").cast("long") - F.col("tpep_pickup_datetime").cast("long")) / 60) \
           .withColumn("pickup_hour", F.hour("tpep_pickup_datetime")) \
           .withColumn("pickup_day_week", F.dayofweek("tpep_pickup_datetime")) \
           .withColumn("pickup_month", F.month("tpep_pickup_datetime"))

    df = df.filter(
        (F.col("trip_distance").between(0.001, 200)) & 
        (F.col("passenger_count") > 0) & 
        (F.col("Durée_minutes").between(0.001, 240)) &
        (F.col("fare_amount") > 0) &
        (F.col("total_amount") > 0) &
        (F.col("RatecodeID") != 99)
    )

    # 3. Intelligent Null Handling (Batch processing stats)
    # Get null counts for all columns in ONE pass instead of a loop
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
    total_rows = df.count()

    for col_name, null_count in null_counts.items():
        if null_count > 0:
            percentage = (null_count / total_rows) * 100
            if percentage < 5:
                df = df.na.drop(subset=[col_name])
            else:
                # Handle Imputation
                dtype = dict(df.dtypes)[col_name]
                if dtype in ['int', 'double', 'bigint']:
                    mean_val = df.select(F.mean(col_name)).collect()[0][0]
                    df = df.fillna({col_name: mean_val})
                else:
                    mode_val = df.groupBy(col_name).count().orderBy("count", ascending=False).first()[0]
                    df = df.fillna({col_name: mode_val})
    
    return df.drop("tpep_dropoff_datetime", "tpep_pickup_datetime")

def save_to_postgres(df):
    jdbc_url = f"jdbc:postgresql://{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}"
    df.write.format("jdbc").options(
        url=jdbc_url,
        driver="org.postgresql.Driver",
        dbtable="silver_data_test",
        user=os.getenv('user'),
        password=os.getenv('password')
    ).mode("overwrite").save()

# Main Execution Flow
if __name__ == "__main__":
    # Load
    raw_df = load_and_save_bronze("1zVL3AIQFKE_lwEYdX2CJJnJRQLRKmUAa")
    
    # Transform
    silver_df = clean_data_optimized(raw_df)
    
    # Save Local and Postgres
    silver_df.cache() # Cache because we are writing to two destinations
    silver_df.write.mode('overwrite').parquet(SILVER_PATH)
    save_to_postgres(silver_df)
    silver_df.unpersist()