import requests
import os
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def download_dataset(source_url: str, local_destination: str) -> None:
    """
    Tache 1: Téléchargement du dataset
    Downloads a dataset from a given URL to a local destination.
    
    Args:
        source_url (str): The URL of the dataset.
        local_destination (str): The local file path to save the dataset.
    """
    try:
        response = requests.get(source_url, stream=True)
        response.raise_for_status()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(local_destination), exist_ok=True)
        
        with open(local_destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Dataset downloaded successfully to {local_destination}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def store_bronze(df: DataFrame, bronze_path: str) -> None:
    """
    Tache 2: Stockage en Bronze
    Saves the DataFrame to the Bronze layer in Parquet format.
    
    Args:
        df (DataFrame): The raw Spark DataFrame.
        bronze_path (str): The destination path for the Bronze layer.
    """
    try:
        df.write.mode('overwrite').parquet(bronze_path)
        print(f"Data successfully stored in Bronze layer at {bronze_path}")
    except Exception as e:
        print(f"Error storing data in Bronze layer: {e}")
        raise

def store_silver_postgres(df: DataFrame, jdbc_url: str, table_name: str, properties: dict) -> None:
    """
    Tache 3: Stockage Silver dans PostgreSQL
    Saves the DataFrame to a PostgreSQL database table.
    
    Args:
        df (DataFrame): The cleaned/processed Spark DataFrame (Silver data).
        jdbc_url (str): The JDBC URL for PostgreSQL (e.g., jdbc:postgresql://host:port/dbname).
        table_name (str): The target table name.
        properties (dict): Connection properties including 'user' and 'password'.
    """
    try:
        df.write \
            .format("jdbc") \
            .option("url", jdbc_url) \
            .option("dbtable", table_name) \
            .option("user", properties.get("user")) \
            .option("password", properties.get("password")) \
            .option("driver", "org.postgresql.Driver") \
            .mode("overwrite") \
            .save()
        print(f"Data successfully stored in PostgreSQL table '{table_name}'")
    except Exception as e:
        print(f"Error storing data in PostgreSQL: {e}")
        raise

def train_model(train_df: DataFrame, label_col: str = "Durée_minutes"):
    """
    Tache 4: Entrainement modèle
    Trains a GBTRegressor model using the provided training data.
    
    Args:
        train_df (DataFrame): The training data from Silver layer.
        label_col (str): The name of the target column.
        
    Returns:
        model: The trained GBTRegressor model.
        metrics (dict): Evaluation metrics (RMSE, R2).
    """
    # 1. Feature Engineering: Assemble features
    # Note: Adjust inputCols based on actual available columns in train_df
    feature_cols = [
        'passenger_count', 'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 
        'PULocationID', 'DOLocationID', 'payment_type', 'fare_amount', 'extra', 
        'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 
        'congestion_surcharge', 'Airport_fee', 'cbd_congestion_fee', 
        'pickup_hour', 'pickup_day_week', 'pickup_month'
    ]
    
    # Check if columns exist to avoid errors
    existing_cols = [c for c in feature_cols if c in train_df.columns]
    
    assembler = VectorAssembler(inputCols=existing_cols, outputCol="features_vec")
    
    # 2. Scaling
    scaler = StandardScaler(inputCol="features_vec", outputCol="features", withStd=True, withMean=False)
    
    # 3. Model
    gbt = GBTRegressor(featuresCol="features", labelCol=label_col, maxIter=50) # Using parameters from ml.ipynb
    
    # Pipeline
    pipeline = Pipeline(stages=[assembler, scaler, gbt])
    
    # Train
    print("Training GBTRegressor model...")
    model = pipeline.fit(train_df)
    print("Training complete.")
    
    # Evaluate (Self-evaluation on training data for demonstration, ideally use a test set provided)
    predictions = model.transform(train_df)
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    r2 = evaluator.setMetricName("r2").evaluate(predictions)
    
    print(f"Model Evaluation -> RMSE: {rmse}, R2: {r2}")
    
    return model, {"rmse": rmse, "r2": r2}
