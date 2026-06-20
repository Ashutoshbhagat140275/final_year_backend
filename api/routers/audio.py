from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from api.middleware.auth import get_current_user_id
from api.schemas.audio import UploadResponse
from api.services.audio_processor import process_audio

router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.post("/upload", response_model=UploadResponse)
def upload_audio(
    file: UploadFile,
    user_id: str = Depends(get_current_user_id),
):
    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file provided")
    return process_audio(user_id, file)


# POST /api/audio/feedback is served by the feedback router (Stage 7) — the real
# personalization endpoint the mobile app calls.
