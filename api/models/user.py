from datetime import datetime, timezone

from pymongo.collection import Collection
from pymongo.database import Database


class User:
    def __init__(
        self,
        email: str,
        password_hash: str,
        user_id: str | None = None,
        feedback_count: int = 0,
        is_admin: bool = False,
        created_at: datetime | None = None,
    ):
        self.user_id = user_id
        self.email = email
        self.password_hash = password_hash
        self.feedback_count = feedback_count
        self.is_admin = is_admin
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "password_hash": self.password_hash,
            "feedback_count": self.feedback_count,
            "is_admin": self.is_admin,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            user_id=data.get("user_id"),
            email=data["email"],
            password_hash=data["password_hash"],
            feedback_count=data.get("feedback_count", 0),
            is_admin=data.get("is_admin", False),
            created_at=data.get("created_at"),
        )

    @staticmethod
    def get_collection(db: Database) -> Collection:
        return db["users"]

    @staticmethod
    def increment_feedback_count(db: Database, user_id: str) -> int:
        result = db["users"].find_one_and_update(
            {"user_id": user_id},
            {"$inc": {"feedback_count": 1}},
            return_document=True,
        )
        return result["feedback_count"] if result else 0
