from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.db.mongodb import get_database
from api.middleware.auth import get_current_admin
from api.services import admin_service

router = APIRouter(prefix="/admin", tags=["admin"])


class CleanupRequest(BaseModel):
    min_days_inactive: int | None = None
    max_feedback_count: int | None = None
    user_ids: list[str] | None = None


def _db():
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db


@router.get("/user-models")
def list_user_models(_admin=Depends(get_current_admin)):
    return admin_service.list_user_models(_db())


@router.delete("/user-models/cleanup")
def cleanup_user_models(body: CleanupRequest, _admin=Depends(get_current_admin)):
    return admin_service.cleanup_user_models(
        _db(), body.min_days_inactive, body.max_feedback_count, body.user_ids
    )
