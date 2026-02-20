from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException
from jose import jwt
from pydantic import BaseModel

from app.config import settings

router = APIRouter(prefix="/auth", tags=["auth"])

SECRET_KEY = "trader-bot-jwt-secret"
ALGORITHM = "HS256"


class LoginRequest(BaseModel):
    password: str


class TokenResponse(BaseModel):
    token: str


@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest):
    if not settings.auth_password:
        raise HTTPException(status_code=400, detail="Auth not configured")
    if body.password != settings.auth_password:
        raise HTTPException(status_code=401, detail="Wrong password")
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.auth_token_expire_minutes)
    token = jwt.encode({"exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
    return TokenResponse(token=token)
