import pytest
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.sql import SparkSession

@pytest.fixture
def Data():
    spark = SparkSession.builder.appName("App").getOrCreate()
    fake_data = [2.85, 17.0, 1.0, 0.0, 5.0, 1.0, 15.6]
    row = Row(Features=Vectors.dense([
        fake_data[0],
        fake_data[1],
        fake_data[2],
        fake_data[3],
        fake_data[4],
        fake_data[5],
        fake_data[6]
    ]))
    df = spark.createDataFrame([row])
    return df

def test_predict(Data):
    model_path = "/media/rachid/d70e3dc6-74e7-4c87-96bc-e4c3689c979a/lmobrmij/Projects/Smart-LogiTrack-Syst-me-Pr-dictif-de-Transport-Urbain-ETA-/ml/Model/gbt_duration_minutes_v1"
    lr_model_loaded = GBTRegressionModel.load(model_path)
    prediction_fake_data = lr_model_loaded.transform(Data)
    value = prediction_fake_data.first()["prediction"]
    assert value == 2.601984904827173
