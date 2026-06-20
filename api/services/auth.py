"""
Auth service — Stage 2: MongoDB persistence.
Falls back to in-memory if Mongo is unavailable (graceful degradation).
"""
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt

from api.config import settings

# ── Password helpers ───────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── Token ──────────────────────────────────────────────────────────────────────

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=settings.jwt_expiration_hours)
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


# ── User operations ────────────────────────────────────────────────────────────

def create_user(email: str, password: str) -> dict:
    from api.db.mongodb import get_database
    from api.models.user import User

    db = get_database()
    user_id = str(uuid.uuid4())

    if db is not None:
        col = User.get_collection(db)
        if col.find_one({"email": email}):
            raise ValueError("Email already registered")
        user = User(email=email, password_hash=hash_password(password), user_id=user_id)
        col.insert_one(user.to_dict())
    else:
        # Fallback: in-memory (graceful degradation when Mongo is down)
        if email in _fallback_store:
            raise ValueError("Email already registered")
        _fallback_store[email] = {
            "user_id": user_id,
            "email": email,
            "password_hash": hash_password(password),
            "feedback_count": 0,
            "is_admin": False,
        }

    return {"user_id": user_id, "email": email}


def authenticate_user(email: str, password: str) -> dict | None:
    from api.db.mongodb import get_database
    from api.models.user import User

    db = get_database()

    if db is not None:
        doc = User.get_collection(db).find_one({"email": email})
        if not doc or not verify_password(password, doc["password_hash"]):
            return None
        return {"user_id": doc["user_id"], "email": doc["email"], "is_admin": doc.get("is_admin", False)}
    else:
        user = _fallback_store.get(email)
        if not user or not verify_password(password, user["password_hash"]):
            return None
        return {"user_id": user["user_id"], "email": user["email"], "is_admin": user.get("is_admin", False)}


def get_user_by_id(user_id: str) -> dict | None:
    from api.db.mongodb import get_database
    from api.models.user import User

    db = get_database()

    if db is not None:
        doc = User.get_collection(db).find_one({"user_id": user_id})
        return doc if doc else None
    else:
        for user in _fallback_store.values():
            if user["user_id"] == user_id:
                return user
        return None


# ── In-memory fallback (Mongo unavailable) ────────────────────────────────────
_fallback_store: dict[str, dict] = {}
