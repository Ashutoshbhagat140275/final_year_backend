from datetime import datetime, timezone

from pymongo.collection import Collection
from pymongo.database import Database


class UserFeedback:
    def __init__(
        self,
        user_id: str,
        session_id: str,
        embedding: list[float],
        predicted_emotion: str,
        corrected_emotion: str,
        timestamp: datetime | None = None,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.embedding = embedding
        self.predicted_emotion = predicted_emotion
        self.corrected_emotion = corrected_emotion
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "embedding": self.embedding,
            "predicted_emotion": self.predicted_emotion,
            "corrected_emotion": self.corrected_emotion,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def get_collection(db: Database) -> Collection:
        return db["user_feedback"]
