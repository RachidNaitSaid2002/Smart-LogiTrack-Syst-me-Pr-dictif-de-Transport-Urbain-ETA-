from pydantic import BaseModel
from traitlets import Integer

class PredictionBase(BaseModel):
    trip_distance: float
    pickup_hour: int
    pickup_month: int
    Airport_fee: float
    pickup_day_week: int
    RateCodeID: int
    fare_amount: float

    class Config:
        orm_mode = True

class PredictionCreate(PredictionBase):
    pass

    class Config:
        orm_mode = True


class PredictionOut(PredictionBase):
    id: int
    user_id: int
    predicted_duration: float

    class Config:
        orm_mode = True