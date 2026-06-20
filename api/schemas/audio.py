from datetime import datetime

from pydantic import BaseModel


class SpeakerSegment(BaseModel):
    speaker_label: str
    start: float
    end: float
    owner_confidence: float | None = None


class UploadResponse(BaseModel):
    session_id: str
    emotion: str
    confidence: float
    global_emotion: str | None = None
    global_confidence: float | None = None
    user_emotion: str | None = None
    user_confidence: float | None = None
    blend_weight: float | None = None
    alpha_data: float | None = None
    alpha_conf: float | None = None
    alpha_formula: str | None = None
    transcription: str = ""
    timestamp: datetime
    owner_speech_ratio: float | None = None
    owner_segments_count: int | None = None
    other_segments_count: int | None = None
    owner_detection_status: str | None = None
    speaker_timeline: list[SpeakerSegment] = []
