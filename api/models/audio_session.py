from datetime import datetime, timezone

from pymongo.collection import Collection
from pymongo.database import Database


class AudioSession:
    def __init__(
        self,
        user_id: str,
        audio_file_path: str,
        emotion_data: dict | None = None,
        transcription_text: str = "",
        qdrant_collection_id: str | None = None,
        speaker_timeline: list | None = None,
        owner_detection_status: str | None = None,
        owner_speech_ratio: float | None = None,
        owner_segments_count: int | None = None,
        other_segments_count: int | None = None,
        personalization_trainable: bool = True,
        session_id: str | None = None,
        timestamp: datetime | None = None,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.audio_file_path = audio_file_path
        self.emotion_data = emotion_data or {}
        self.transcription_text = transcription_text
        self.qdrant_collection_id = qdrant_collection_id
        self.speaker_timeline = speaker_timeline or []
        self.owner_detection_status = owner_detection_status
        self.owner_speech_ratio = owner_speech_ratio
        self.owner_segments_count = owner_segments_count
        self.other_segments_count = other_segments_count
        self.personalization_trainable = personalization_trainable
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "audio_file_path": self.audio_file_path,
            "emotion_data": self.emotion_data,
            "transcription_text": self.transcription_text,
            "qdrant_collection_id": self.qdrant_collection_id,
            "speaker_timeline": self.speaker_timeline,
            "owner_detection_status": self.owner_detection_status,
            "owner_speech_ratio": self.owner_speech_ratio,
            "owner_segments_count": self.owner_segments_count,
            "other_segments_count": self.other_segments_count,
            "personalization_trainable": self.personalization_trainable,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AudioSession":
        return cls(
            session_id=str(data.get("_id", data.get("session_id", ""))),
            user_id=data["user_id"],
            audio_file_path=data.get("audio_file_path", ""),
            emotion_data=data.get("emotion_data", {}),
            transcription_text=data.get("transcription_text", ""),
            qdrant_collection_id=data.get("qdrant_collection_id"),
            speaker_timeline=data.get("speaker_timeline", []),
            owner_detection_status=data.get("owner_detection_status"),
            owner_speech_ratio=data.get("owner_speech_ratio"),
            owner_segments_count=data.get("owner_segments_count"),
            other_segments_count=data.get("other_segments_count"),
            personalization_trainable=data.get("personalization_trainable", True),
            timestamp=data.get("timestamp"),
        )

    @staticmethod
    def get_collection(db: Database) -> Collection:
        return db["audio_sessions"]
