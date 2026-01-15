from sqlalchemy.orm import relationship
from sqlalchemy import  Column, Integer, String, Boolean
from backend.Database.db import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(120), unique=True, index=True, nullable=False)
    full_name = Column(String(100))
    hashed_password = Column(String(255), nullable=False)
    active = Column(Boolean, default=True)

    # Relationship
    predictions = relationship("Prediction", back_populates="user")