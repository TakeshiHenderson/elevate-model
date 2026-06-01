from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from config import get_users, settings

_bearer = HTTPBearer()


def authenticate_user(email: str, password: str) -> Optional[dict]:
    for u in get_users():
        if u["email"] == email and u["password"] == password:
            return u
    return None


def create_access_token(user: dict) -> str:
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(
        {"sub": user["email"], "name": user["name"], "role": user["role"], "exp": expire},
        settings.SECRET_KEY,
        algorithm="HS256",
    )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    try:
        payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=["HS256"])
        if not payload.get("sub"):
            raise HTTPException(status_code=401, detail="Invalid token")
        return {"email": payload["sub"], "name": payload.get("name"), "role": payload.get("role")}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
