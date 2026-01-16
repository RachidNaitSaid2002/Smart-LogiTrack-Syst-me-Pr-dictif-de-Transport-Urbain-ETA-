from math import expm1
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.Database.db import engine,SessionLocal,Base
from fastapi import FastAPI, HTTPException, Response, status, Depends, Cookie
from backend.Models.users import User
from backend.Models.predections import Prediction
from backend.Models.Silver import SilverData
from backend.Schemas.users import UserCreate, UserLogin
from backend.Schemas.predictions import PredictionCreate, PredictionOut
from fastapi.security import HTTPBearer, HTTPBasicCredentials
from dotenv import load_dotenv
import os
from backend.Auth.auth import get_password_hash, get_uer, verify_jwt, create_jwt, verify_password
from pyspark.ml.regression import GBTRegressionModel
from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from sqlalchemy import text


load_dotenv()

SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM')
bearer_scheme = HTTPBearer()

Base.metadata.create_all(engine)
app = FastAPI()

def get_spark():
    return SparkSession.builder.getOrCreate()


db = SessionLocal()
  
# SignUp --------------------------------------------------------------------- 
@app.post('/signup')
def Signup(user: UserCreate):
    db_user = get_uer(db, user.email)
    if db_user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail='Email Already exist !!')
    pass_hached = get_password_hash(user.hashed_password)
    db_user = User(username=user.username, email=user.email, full_name=user.full_name, hashed_password=pass_hached)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {'message':'User Register Successfully'}


# Login ----------------------------------------------------------------------- 
@app.post('/login')
async def login(user: UserLogin):
    db_user = get_uer(db, user.email)
    if db_user:
        verify_pass = verify_password(user.hashed_password, db_user.hashed_password)
        if verify_pass:
            token = create_jwt(user.email)
            return { 'access_token':token, 'token_type': "bearer", "user_id": db_user.id }
        else:
            return  {'message':'Password incorrect !!'}
    return {'message':'Username ou Password incorrect !!'}


# Create Prediction ---------------------------------------------------------- 
@app.post('/predictions/')
def create_prediction(prediction: PredictionCreate, credentials: HTTPBasicCredentials = Depends(bearer_scheme)):
    email = verify_jwt(credentials.credentials)
    db_user = get_uer(db, email)

    if db_user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    
    model_path = "../ml/Model/gbt_duration_minutes_v1"

    features = Row(Features=Vectors.dense([
        prediction.trip_distance,
        prediction.pickup_hour,
        prediction.pickup_month,
        prediction.Airport_fee,
        prediction.pickup_day_week,
        prediction.RateCodeID,
        prediction.fare_amount,
    ]))

    spark = get_spark()
    df = spark.createDataFrame([features])

    lr_model_loaded = GBTRegressionModel.load(model_path)
    predicted_duration = lr_model_loaded.transform(df)
    predicted_duration = predicted_duration.collect()[0]["prediction"]

    db_prediction = Prediction(
        user_id=db_user.id,
        trip_distance=prediction.trip_distance,
        pickup_hour=prediction.pickup_hour,
        pickup_month=prediction.pickup_month,
        Airport_fee=prediction.Airport_fee,
        pickup_day_week=prediction.pickup_day_week,
        RateCodeID=prediction.RateCodeID,
        fare_amount = prediction.fare_amount,
        predicted_duration=expm1(predicted_duration),
        model_version=model_path.split("/")[-1]
    )
    
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction


#avg-duration-by-hour use cte --------------------------------------------------------------
@app.get("/analytics/avg-duration-by-hour/{pickup_hour}")
def get_avg_duration_by_hour(pickup_hour: int, credentials: HTTPBasicCredentials = Depends(bearer_scheme)):
    email = verify_jwt(credentials.credentials)
    db_user = get_uer(db, email)

    if db_user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    if pickup_hour < 0 or pickup_hour > 23:
        result = db.execute(
            text("""
                WITH avg_hour AS (
                    SELECT
                        pickup_hour,
                        AVG("Durée_minutes") AS avgduration
                    FROM public.silver_data
                    GROUP BY pickup_hour
                )
                SELECT *
                FROM avg_hour ;
            """)
        ).fetchall()
        if not result:
            raise HTTPException(status_code=404, detail="No data found")

        return_query = []
        for row in result: 
            return_query.append({"pickup_hour": row.pickup_hour, "avgduration": row.avgduration})
        return return_query
    else:
        result = db.execute(
            text("""
                WITH avg_hour AS (
                    SELECT
                        pickup_hour,
                        AVG("Durée_minutes") AS avgduration
                    FROM public.silver_data
                    GROUP BY pickup_hour
                )
                SELECT *
                FROM avg_hour where pickup_hour = :pickup_hour;
            """),
            {"pickup_hour": pickup_hour}
        ).fetchone()

        if not result:
            raise HTTPException(status_code=404, detail="No data found")

        return {"pickup_hour": result.pickup_hour, "avgduration": result.avgduration}


#analytics/payment-analysis ------------------------------------------------------------------------
@app.get("/analytics/payment-analysis")
def payment_analysis(credentials: HTTPBasicCredentials = Depends(bearer_scheme)):
    email = verify_jwt(credentials.credentials)
    db_user = get_uer(db, email)

    if db_user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")

    result = db.execute(text("""
        with payment_analysis as (
            select payment_type, count(*) as total_trips, avg("Durée_minutes") as avg_duration
            from public.silver_data
            group by payment_type
        )
        select * from payment_analysis
    """)).fetchall()

    if not result:
        raise HTTPException(status_code=404, detail="No data found")
    
    return_query = []

    for row in result: 
        return_query.append({"payment_type": row.payment_type, "total_trips": row.total_trips, "avg_duration":row.avg_duration})
        
    return return_query