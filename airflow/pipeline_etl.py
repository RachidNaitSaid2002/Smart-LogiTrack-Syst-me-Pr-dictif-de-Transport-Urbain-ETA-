import gdown
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, hour, dayofweek, month, mean, log1p, expm1
from pyspark.ml.feature import VectorAssembler
from dotenv import load_dotenv
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os

load_dotenv()

spark = (
    SparkSession.builder
    .appName("App")
    .config(
        "spark.jars",
        "/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/postgresql-42.7.3.jar"
    )
    .getOrCreate()
)

#-------------------------------------------------------------------------------------------------------------------------------------
def load_data(file_id: str, filename: str = "data.parquet"):
    output_path = f"/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/Test/Data/{filename}"
    gdown.download(id=file_id, output=output_path, quiet=False)
    df = spark.read.parquet(output_path)
    return df

file_id = "1zVL3AIQFKE_lwEYdX2CJJnJRQLRKmUAa"
df = load_data(file_id)
df.show(5)

#-------------------------------------------------------------------------------------------------------------------------------------
def Save_Bronze_Local(Data_i):
    Data_i.write.mode('overwrite').format("parquet").save("/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/Test/Bronze")
Save_Bronze_Local(df)

#-------------------------------------------------------------------------------------------------------------------------------------
def ReadDataset(file_path):
    df = spark.read.parquet(file_path)
    if df is None:
        raise Exception("Failed to read dataset")
    else :
        return df

#-------------------------------------------------------------------------------------------------------------------------------------
def CheckNull(Data):
    num_rows = Data.count()       
    Columns_list = Data.columns

    for c in Columns_list:
        num_null = Data.filter(col(c).isNull()).count()
        if num_null > 0:
            null_percent = (num_null / num_rows) * 100
            print(f"Column {c} has {num_null} null values ({null_percent:.2f}%)")
            
            if null_percent < 5:
                Data = Data.na.drop(subset=[c])
            else:
                try:
                    mean_value = Data.select(mean(c)).collect()[0][0]
                    Data = Data.fillna({c: mean_value})
                except:
                    mode_value = Data.groupBy(c).count().orderBy(col("count").desc()).first()[0]
                    Data = Data.fillna({c: mode_value})
    return Data

#-------------------------------------------------------------------------------------------------------------------------------------
def CheckDuplicated(Data):
    num_rows = Data.count()
    num_rows_no_duplicate = Data.distinct().count()
    num_duplicate_values = num_rows - num_rows_no_duplicate
    if num_duplicate_values == 0:
        print("you don't have any duplicated values !!")
    else:
        Data = Data.distinct()
        return Data
    return Data

#-------------------------------------------------------------------------------------------------------------------------------------
def Time_Cleaning(Data):
    Data = Data.withColumn("Durée", col("tpep_dropoff_datetime") - col("tpep_pickup_datetime"))
    Data = Data.withColumn("Durée_minutes", (col("Durée").cast("long")/60).cast("int"))

    Data = Data.withColumn("pickup_hour", hour(col("tpep_pickup_datetime")))
    Data = Data.withColumn("pickup_day_week", dayofweek(col("tpep_pickup_datetime")))
    Data = Data.withColumn("pickup_month", month(col("tpep_pickup_datetime")))

    Data = Data.drop("Durée", "tpep_dropoff_datetime", "tpep_pickup_datetime")
    return Data

#-------------------------------------------------------------------------------------------------------------------------------------    
def Outliers_filtering(Data):
    Data = Data.filter(
        (col("trip_distance") > 0) & 
        (col("trip_distance") <= 200) & 
        (col("passenger_count") > 0) & 
        (col("Durée_minutes") > 0) &
        (col("fare_amount") > 0) &
        (col("total_amount") > 0) &
        (col("Durée_minutes") <= 240) &
        (col("RatecodeID") != 99)
    )
    return Data

#-------------------------------------------------------------------------------------------------------------------------------------
def Save_Silver_postgresql(Silver_Local_Path):
    jdbc_url = f"jdbc:postgresql://{os.getenv('host')}:{os.getenv('port')}/{os.getenv('database')}"

    connection_properties = {
        "user": os.getenv('user'),
        "password": os.getenv('password'),
        "driver": "org.postgresql.Driver"
    }

    df_load = spark.read.parquet(Silver_Local_Path)

    df_load.write.jdbc(url=jdbc_url, table="silver_data_test", mode="overwrite", properties=connection_properties)

#-------------------------------------------------------------------------------------------------------------------------------------
def Silver_Local():
    Df = ReadDataset("/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/Test/Bronze")
    DF = CheckNull(Df)
    DF = CheckDuplicated(DF)
    DF = Time_Cleaning(DF)
    DF = Outliers_filtering(DF)
    DF.write.mode('overwrite').format("parquet").save("/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/Test/Silver")
    Save_Silver_postgresql("/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/Test/Silver")

#-------------------------------------------------------------------------------------------------------------------------------------
Silver_Local()

#-------------------------------------------------------------------------------------------------------------------------------------
def Prepare_Training(Data):
    Data = Data.withColumn("store_and_fwd_flag", when(col("store_and_fwd_flag") == "N", 0).when(col("store_and_fwd_flag") == "Y", 1))
    Best = ["trip_distance", "pickup_hour", "pickup_month", "Airport_fee", "pickup_day_week", "RateCodeID","Durée_minutes","fare_amount"]
    df_drp = Data.select(Best)
    vec_assmebler = VectorAssembler(inputCols=["trip_distance", "pickup_hour", "pickup_month", "Airport_fee", "pickup_day_week", "RateCodeID","fare_amount"], outputCol='Features')
    features_df = vec_assmebler.transform(df_drp)
    features_df = features_df.withColumn(
        "Durée_minutes_log",
        log1p(col("Durée_minutes"))
    )
    model_df = features_df.select("Features","Durée_minutes_log","Durée_minutes")
    Train_df , Test_df = model_df.randomSplit([0.7,0.3])
    model2 = GBTRegressor(
        featuresCol="Features",
        labelCol="Durée_minutes_log",
        seed=42
    )
    lr_model2 = model2.fit(Train_df)
    test_result2 = lr_model2.transform(Test_df)

    test_result2 = test_result2.withColumn(
        "prediction_original",
        expm1(col("prediction"))
    )

    evaluator_r2 = RegressionEvaluator(
        labelCol="Durée_minutes_log",
        predictionCol="prediction",
        metricName="r2"
    )

    evaluator_mae = RegressionEvaluator(
        labelCol="Durée_minutes",
        predictionCol="prediction_original",
        metricName="mae"
    )

    r2 = evaluator_r2.evaluate(test_result2)
    mae = evaluator_mae.evaluate(test_result2)

    print(f"R2 : {r2}")
    print(f"MAE : {mae}")

    model_path = "../Model_test/gbt_duration_minutes_v1"
    lr_model2.write().overwrite().save(model_path)


Data_S = ReadDataset("/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/Test/Silver")
Prepare_Training(Data_S)