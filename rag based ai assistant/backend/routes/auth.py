from fastapi import APIRouter, HTTPException, status

from backend.deps import create_access_token
from backend.models import TokenResponse, UserLogin, UserRegister, UserRole
from backend.services import db

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(body: UserRegister):
    existing = await db.get_user_by_email(body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    user = await db.create_user(
        email=body.email,
        password=body.password,
        full_name=body.full_name,
        role=body.role.value,
    )
    user_id = str(user["_id"])
    token = create_access_token(user_id, body.role.value, body.full_name)
    return TokenResponse(
        access_token=token,
        role=body.role,
        full_name=body.full_name,
        user_id=user_id,
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: UserLogin):
    user = await db.get_user_by_email(body.email)
    if not user or not db.verify_password(body.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user_id = str(user["_id"])
    role = UserRole(user["role"])
    token = create_access_token(user_id, user["role"], user["full_name"])
    return TokenResponse(
        access_token=token,
        role=role,
        full_name=user["full_name"],
        user_id=user_id,
    )
