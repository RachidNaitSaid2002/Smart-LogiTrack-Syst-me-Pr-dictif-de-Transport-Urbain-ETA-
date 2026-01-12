import gdown
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, when, hour, dayofweek, month, mean
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
# Define paths as constants to avoid repetition and errors
BASE_PATH = "/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/Test"
DATA_PATH = f"{BASE_PATH}/Data/data.parquet"
BRONZE_PATH = f"{BASE_PATH}/Bronze"
SILVER_PATH = f"{BASE_PATH}/Silver"

# Initialize Spark Session with optimized settings
# 3 Million rows is small for a cluster, but large for a single machine. 
# We reduce shuffle partitions to avoid overhead.
spark = SparkSession.builder \
    .appName("ETL_Optimized") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

#-------------------------------------------------------------------------------------------------------------------------------------
def load_data(file_id: str):
    # Check if file exists to avoid re-downloading every time
    if not os.path.exists(DATA_PATH):
        print(f"Downloading data to {DATA_PATH}...")
        # Ensure directory exists
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
        gdown.download(id=file_id, output=DATA_PATH, quiet=False)
    else:
        print("Data file already exists. Skipping download.")
    
    return spark.read.parquet(DATA_PATH)

#-------------------------------------------------------------------------------------------------------------------------------------
def check_nulls_optimized(df):
    """
    Optimized Null Handling:
    1. Calculates null percentages for ALL columns in a single pass.
    2. Separates columns into 'Drop' (<5% nulls) and 'Fill' (>=5% nulls).
    3. Applies drop and fill operations in bulk.
    """
    total_rows = df.count()
    
    # 1. Calculate null counts for all columns in one aggregation
    # This creates a single action/job instead of one per column
    null_counts = df.select([
        sum(col(c).isNull().cast("int")).alias(c) for c in df.columns
    ]).collect()[0].asDict()
    
    cols_to_drop = []
    cols_to_fill = {}
    
    # 2. Determine strategy for each column
    for col_name, null_count in null_counts.items():
        if null_count > 0:
            null_percent = (null_count / total_rows) * 100
            print(f"Column {col_name}: {null_count} nulls ({null_percent:.2f}%)")
            
            if null_percent < 5:
                cols_to_drop.append(col_name)
            else:
                # Prepare to fill (We calculate values later to save passes)
                cols_to_fill[col_name] = None # Placeholder

    # 3. Apply Drop (One operation)
    if cols_to_drop:
        print(f"Dropping rows with nulls in: {cols_to_drop}")
        df = df.na.drop(subset=cols_to_drop)
    
    # 4. Calculate Fill Values and Apply (Batch operations)
    if cols_to_fill:
        fill_values = {}
        
        # Separate numeric and string columns for appropriate filling strategies
        numeric_cols = [f.name for f in df.schema.fields if f.dataType.typeName() in ['integer', 'double', 'float', 'long', 'short']]
        
        # Calculate Means for Numeric columns in one pass
        numeric_fill_cols = [c for c in cols_to_fill.keys() if c in numeric_cols]
        if numeric_fill_cols:
            means = df.select([mean(c).alias(c) for c in numeric_fill_cols]).collect()[0].asDict()
            for c in numeric_fill_cols:
                fill_values[c] = means[c]

        # Calculate Modes for Non-Numeric columns (Iterative but usually fewer columns)
        string_fill_cols = [c for c in cols_to_fill.keys() if c not in numeric_cols]
        for c in string_fill_cols:
            try:
                # Approximate mode: take first row of the most frequent value
                mode_val = df.groupBy(c).count().orderBy(col("count").desc()).first()[0]
                fill_values[c] = mode_val
            except:
                fill_values[c] = "Unknown" # Fallback
        
        print(f"Filling columns with calculated values: {list(fill_values.keys())}")
        df = df.na.fill(fill_values)
        
    return df

#-------------------------------------------------------------------------------------------------------------------------------------
def check_duplicates_optimized(df):
    """
    Optimized Duplicate Handling:
    Simply drops duplicates without counting them first. 
    Counting adds two expensive Jobs (Action -> Plan -> Execute).
    """
    return df.dropDuplicates()

#-------------------------------------------------------------------------------------------------------------------------------------
def time_cleaning(df):
    """
    Optimized Time Handling:
    Combines duration calculation and casting into fewer steps.
    """
    # Calculate duration in minutes directly
    df = df.withColumn("Durée_minutes", 
        ((col("tpep_dropoff_datetime").cast("long") - col("tpep_pickup_datetime").cast("long")) / 60).cast("int")
    )
    
    # Extract features
    df = df.withColumn("pickup_hour", hour("tpep_pickup_datetime"))
    df = df.withColumn("pickup_day_week", dayofweek("tpep_pickup_datetime"))
    df = df.withColumn("pickup_month", month("tpep_pickup_datetime"))
    
    # Drop unused columns
    df = df.drop("tpep_dropoff_datetime", "tpep_pickup_datetime")
    return df

#-------------------------------------------------------------------------------------------------------------------------------------
def outliers_filtering(df):
    """Standard filtering is efficient, logic remains unchanged."""
    return df.filter(
        (col("trip_distance") > 0) & (col("trip_distance") <= 200) & 
        (col("passenger_count") > 0) & 
        (col("Durée_minutes") > 0) & (col("Durée_minutes") <= 240) &
        (col("fare_amount") > 0) & (col("total_amount") > 0) &
        (col("RatecodeID") != 99)
    )

#-------------------------------------------------------------------------------------------------------------------------------------
def save_silver_postgresql(df):
    """
    Optimized DB Write:
    1. Accepts DataFrame directly (no re-reading from disk).
    2. Uses batchsize for faster JDBC writes.
    """
    jdbc_url = f"jdbc:postgresql://{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}"
    
    connection_properties = {
        "user": os.getenv('user'),
        "password": os.getenv('password'),
        "driver": "org.postgresql.Driver"
    }

    print("Writing data to PostgreSQL...")
    df.write.jdbc(
        url=jdbc_url, 
        table="silver_data_test", 
        mode="overwrite", 
        properties=connection_properties,
        batchsize=10000 # Tuning this number can improve write performance significantly
    )

#-------------------------------------------------------------------------------------------------------------------------------------
def main_silver_pipeline():
    file_id = "1zVL3AIQFKE_lwEYdX2CJJnJRQLRKmUAa"

    # 1. Load & Save Bronze
    print("--- Starting Load & Bronze Save ---")
    df_raw = load_data(file_id)
    df_raw.write.mode('overwrite').format("parquet").save(BRONZE_PATH)
    
    # 2. Read Bronze
    print("--- Processing Silver Layer ---")
    # Note: In a production pipeline, you might pass df_raw directly, 
    # but reading back simulates the "Bronze" isolation layer.
    df_bronze = spark.read.parquet(BRONZE_PATH)
    
    # 3. Transformations
    df_silver = check_nulls_optimized(df_bronze)
    df_silver = check_duplicates_optimized(df_silver)
    df_silver = time_cleaning(df_silver)
    df_silver = outliers_filtering(df_silver)
    
    # 4. Save Silver Local
    df_silver.write.mode('overwrite').format("parquet").save(SILVER_PATH)
    print(f"Silver data saved locally to {SILVER_PATH}")
    
    # 5. Save Silver Postgres
    save_silver_postgresql(df_silver)
    print("Pipeline Completed.")

#-------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main_silver_pipeline()