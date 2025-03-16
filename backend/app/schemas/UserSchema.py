from pydantic import BaseModel
from typing import Optional

class UserCreate(BaseModel):
    username: str
    data_url: str
    # email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str
    
class RefreshToken(BaseModel):
    refresh_token: str

class ClientSessionStatusSchema(BaseModel):
    session_id: int
    curr_round: int
    max_round: int
    session_price: Optional[float]
    training_status: int
    client_status: Optional[int]
