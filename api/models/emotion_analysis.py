from datetime import datetime, timezone

from pymongo.collection import Collection
from pymongo.database import Database


class EmotionAnalysis:
    """
    One emotion prediction per audio session.

    NOTE: `mfcc_features` is a historical field name — it stores the **768-dim
    Wav2Vec2 embedding**, not MFCCs. Reused as training input for the user head
    so feedback training needs no audio re-decoding.
    """

    def __init__(
        self,
        user_id: str,
        session_id: str,
        emotion_label: str,
        confidence: float,
        mfcc_features: list[float],
        timestamp: datetime | None = None,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.emotion_label = emotion_label
        self.confidence = confidence
        self.mfcc_features = mfcc_features
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "emotion_label": self.emotion_label,
            "confidence": self.confidence,
            "mfcc_features": self.mfcc_features,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EmotionAnalysis":
        return cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            emotion_label=data["emotion_label"],
            confidence=data["confidence"],
            mfcc_features=data.get("mfcc_features", []),
            timestamp=data.get("timestamp"),
        )

    @staticmethod
    def get_collection(db: Database) -> Collection:
        return db["emotion_analyses"]
