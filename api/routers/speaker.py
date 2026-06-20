from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from api.db.mongodb import get_database
from api.middleware.auth import get_current_user_id
from api.schemas.speaker import EnrollmentStatus, MessageResponse
from api.services import speaker_service
from api.services.audio_processor import save_enrollment_clip

router = APIRouter(prefix="/api/speaker", tags=["speaker"])


def _db_or_503():
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db


@router.post("/enroll/start", response_model=MessageResponse)
def enroll_start(user_id: str = Depends(get_current_user_id)):
    return speaker_service.start_enrollment(_db_or_503(), user_id)


@router.post("/enroll/upload", response_model=MessageResponse)
def enroll_upload(file: UploadFile, user_id: str = Depends(get_current_user_id)):
    path = save_enrollment_clip(user_id, file)
    return speaker_service.add_enrollment_sample(_db_or_503(), user_id, path)


@router.post("/enroll/complete", response_model=MessageResponse)
def enroll_complete(user_id: str = Depends(get_current_user_id)):
    try:
        return speaker_service.complete_enrollment(_db_or_503(), user_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/enroll/status", response_model=EnrollmentStatus)
def enroll_status(user_id: str = Depends(get_current_user_id)):
    return speaker_service.get_enrollment_status(_db_or_503(), user_id)
