"""
Auth service — Stage 1 uses an in-memory user store.
Stage 2 replaces _users_db with real MongoDB reads/writes.
"""
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from api.config import settings

# ── In-memory store (Stage 1 only) ────────────────────────────────────────────
# { email: {"user_id": str, "password_hash": bytes} }
_users_db: dict[str, dict] = {}


# ── Password helpers ───────────────────────────────────────────────────────────

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())


def verify_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), hashed)


# ── Token ──────────────────────────────────────────────────────────────────────

def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expiration_hours)
    payload["exp"] = expire
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


# ── User operations (in-memory, Stage 1) ──────────────────────────────────────

def create_user(email: str, password: str) -> dict:
    if email in _users_db:
        raise ValueError("Email already registered")
    user_id = str(uuid.uuid4())
    _users_db[email] = {
        "user_id": user_id,
        "email": email,
        "password_hash": hash_password(password),
        "feedback_count": 0,
        "is_admin": False,
        "created_at": datetime.now(timezone.utc),
    }
    return {"user_id": user_id, "email": email}


def authenticate_user(email: str, password: str) -> dict | None:
    user = _users_db.get(email)
    if not user:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return {"user_id": user["user_id"], "email": user["email"], "is_admin": user["is_admin"]}


def get_user_by_id(user_id: str) -> dict | None:
    for user in _users_db.values():
        if user["user_id"] == user_id:
            return user
    return None
