from pydantic import BaseModel


class EmotionRecord(BaseModel):
    session_id: str | None = None
    emotion_label: str
    confidence: float
    timestamp: str | None = None


class EmotionHistoryResponse(BaseModel):
    emotions: list[EmotionRecord] = []
    total: int = 0


class StatsResponse(BaseModel):
    total_sessions: int
    emotion_distribution: dict
    avg_confidence: float
