from datetime import datetime, timezone

from pymongo.collection import Collection
from pymongo.database import Database


class SpeakerProfile:
    def __init__(
        self,
        user_id: str,
        owner_embedding: list[float] | None = None,
        enrolled: bool = False,
        enrollment_state: str = "not_started",
        pending_embeddings: list[list[float]] | None = None,
        sample_count: int = 0,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
    ):
        self.user_id = user_id
        self.owner_embedding = owner_embedding
        self.enrolled = enrolled
        self.enrollment_state = enrollment_state
        self.pending_embeddings = pending_embeddings or []
        self.sample_count = sample_count
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "owner_embedding": self.owner_embedding,
            "enrolled": self.enrolled,
            "enrollment_state": self.enrollment_state,
            "pending_embeddings": self.pending_embeddings,
            "sample_count": self.sample_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerProfile":
        return cls(
            user_id=data["user_id"],
            owner_embedding=data.get("owner_embedding"),
            enrolled=data.get("enrolled", False),
            enrollment_state=data.get("enrollment_state", "not_started"),
            pending_embeddings=data.get("pending_embeddings", []),
            sample_count=data.get("sample_count", 0),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    @staticmethod
    def get_collection(db: Database) -> Collection:
        return db["speaker_profiles"]
