import pytest
from ml.pipeline_etl import CheckNull, CheckDuplicated
from pyspark.sql import SparkSession
                                                                                                                                                                                                                                   
@pytest.fixture
def Data():
    spark = SparkSession.builder.appName("App").getOrCreate()
    fake_data = [
        (1, 2.0, None, 4.0),
        (2, None, 3.0, 4.0),
        (3, 2.0, 3.0, None),
        (4, None, None, None),
        (5, 2.0, 3.0, 4.0),
        (6, 2.0, 3.0, 5.0),
        (6, 2.0, 3.0, 5.0),
        (6, 2.0, 3.0, 5.0)
    ]
    columns = ["id", "feature1", "feature2", "feature3"]
    df = spark.createDataFrame(fake_data, columns)
    return df

def test_CheckDuplicate(Data):
    Df = CheckDuplicated(Data)
    assert Df.count() < Data.count()

def test_CheckNull(Data):
    Df = CheckNull(Data)
    for c in Df.columns:
        num_null = Df.filter(Df[c].isNull()).count()
        assert num_null == 0