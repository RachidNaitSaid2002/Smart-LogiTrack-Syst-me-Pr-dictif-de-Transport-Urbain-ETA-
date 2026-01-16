
from sqlalchemy import  Column, ForeignKey, Integer, String, Boolean, Float
from backend.Database.db import Base
from sqlalchemy.orm import relationship
from sqlalchemy.types import DateTime
from datetime import datetime


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    #Best = ["trip_distance", "pickup_hour", "pickup_month", "Airport_fee", "pickup_day_week", "RateCodeID","Durée_minutes"]
    trip_distance = Column(Float, nullable=False)
    pickup_hour = Column(Integer, nullable=False)
    pickup_month = Column(Integer, nullable=False)
    Airport_fee = Column(Float, nullable=False)
    pickup_day_week = Column(Integer, nullable=False)
    RateCodeID = Column(Integer, nullable=False)
    fare_amount = Column(Float, nullable=False)
    timeStamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String, nullable=False, default="v1")
    # Target
    predicted_duration = Column(Float, nullable=False)

    # foreign key
    user_id = Column(Integer, ForeignKey("users.id"))

    # Relationship
    user = relationship("User", back_populates="predictions")
    
