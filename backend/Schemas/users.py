from typing import Optional
from pydantic import BaseModel

class UserBase(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

class UserCreate(UserBase):
    hashed_password: str

    class Config:
        orm_mode = True

class UserOut(UserBase):
    id: int
    active: bool

    class Config:
        orm_mode = True

    
class UserLogin(BaseModel):
    email: str
    hashed_password:str

    class Config:
        orm_mode = True