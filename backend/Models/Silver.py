from sqlalchemy import Column,Integer,String, DateTime, SmallInteger, Numeric
from backend.Database.db import Base


class SilverData(Base):
    __tablename__ = "silver_data"

    id = Column(Integer, primary_key=True, autoincrement=True)

    VendorID = Column(SmallInteger, nullable=False)

    passenger_count = Column(SmallInteger)
    trip_distance = Column(Numeric(6, 2))

    RatecodeID = Column(SmallInteger)

    store_and_fwd_flag = Column(String(1))

    PULocationID = Column(Integer)
    DOLocationID = Column(Integer)

    payment_type = Column(SmallInteger)

    fare_amount = Column(Numeric(10, 2))
    extra = Column(Numeric(10, 2))
    mta_tax = Column(Numeric(10, 2))
    tip_amount = Column(Numeric(10, 2))
    tolls_amount = Column(Numeric(10, 2))
    improvement_surcharge = Column(Numeric(10, 2))
    total_amount = Column(Numeric(10, 2))

    congestion_surcharge = Column(Numeric(10, 2))
    Airport_fee = Column(Numeric(10, 2))
    cbd_congestion_fee = Column(Numeric(10, 2))

    Durée_minutes = Column(Numeric(10, 2))
    pickup_hour = Column(Integer)
    pickup_month = Column(Integer)
    pickup_day_week = Column(Integer)


