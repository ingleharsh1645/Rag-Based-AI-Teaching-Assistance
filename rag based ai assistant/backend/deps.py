import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from backend.models import UserRole
from backend.services import db

security = HTTPBearer()


def create_access_token(user_id: str, role: str, full_name: str) -> str:
    expire_minutes = int(os.environ.get("JWT_EXPIRE_MINUTES", "1440"))
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    payload = {
        "sub": user_id,
        "role": role,
        "full_name": full_name,
        "exp": expire,
    }
    return jwt.encode(
        payload,
        os.environ["JWT_SECRET"],
        algorithm=os.environ.get("JWT_ALGORITHM", "HS256"),
    )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            os.environ["JWT_SECRET"],
            algorithms=[os.environ.get("JWT_ALGORITHM", "HS256")],
        )
        user_id: str | None = payload.get("sub")
        if not user_id:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    user = await db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    user["id"] = str(user["_id"])
    return user


def require_role(user: dict, role: UserRole) -> None:
    if user.get("role") != role.value:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{role.value} access required",
        )
