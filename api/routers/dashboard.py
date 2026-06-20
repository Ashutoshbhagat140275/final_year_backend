from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from api.db.mongodb import get_database
from api.middleware.auth import get_current_user
from api.models.audio_session import AudioSession
from api.schemas.dashboard import EmotionHistoryResponse, EmotionRecord, StatsResponse

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


def _guard(user_id: str, current: dict):
    if current["user_id"] != user_id and not current.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db


@router.get("/emotions/{user_id}", response_model=EmotionHistoryResponse)
def emotions(user_id: str, start_date: str | None = None, end_date: str | None = None,
             limit: int = 100, current: dict = Depends(get_current_user)):
    db = _guard(user_id, current)
    q: dict = {"user_id": user_id}
    ts: dict = {}
    for key, val in (("$gte", start_date), ("$lte", end_date)):
        if val:
            try:
                ts[key] = datetime.fromisoformat(val)
            except ValueError:
                pass
    if ts:
        q["timestamp"] = ts

    cursor = AudioSession.get_collection(db).find(q).sort("timestamp", -1).limit(limit)
    records = []
    for doc in cursor:
        ed = doc.get("emotion_data") or {}
        t = doc.get("timestamp")
        records.append(EmotionRecord(
            session_id=doc.get("session_id"),
            emotion_label=ed.get("emotion", ed.get("label", "neutral")),
            confidence=float(ed.get("confidence", 0.0)),
            timestamp=t.isoformat() if isinstance(t, datetime) else (str(t) if t else None),
        ))
    return EmotionHistoryResponse(emotions=records, total=len(records))


@router.get("/stats/{user_id}", response_model=StatsResponse)
def stats(user_id: str, current: dict = Depends(get_current_user)):
    db = _guard(user_id, current)
    docs = list(AudioSession.get_collection(db).find({"user_id": user_id}))
    dist: dict[str, int] = {}
    conf_sum = 0.0
    for doc in docs:
        ed = doc.get("emotion_data") or {}
        label = ed.get("emotion", ed.get("label", "neutral"))
        dist[label] = dist.get(label, 0) + 1
        conf_sum += float(ed.get("confidence", 0.0))
    n = len(docs)
    return StatsResponse(
        total_sessions=n,
        emotion_distribution=dist,
        avg_confidence=round(conf_sum / n, 4) if n else 0.0,
    )
