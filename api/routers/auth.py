from fastapi import APIRouter, HTTPException, status

from api.schemas.auth import LoginRequest, RegisterRequest, RegisterResponse, TokenResponse
from api.services.auth import authenticate_user, create_access_token, create_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest):
    try:
        user = create_user(body.email, body.password)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    return RegisterResponse(message="User registered successfully", user_id=user["user_id"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest):
    user = authenticate_user(body.email, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    token = create_access_token({"user_id": user["user_id"], "email": user["email"]})
    return TokenResponse(access_token=token, user_id=user["user_id"])
