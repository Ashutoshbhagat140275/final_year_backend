from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from api.db.mongodb import get_database
from api.middleware.auth import get_current_admin, get_current_user, get_current_user_id
from api.schemas.feedback import FeedbackRequest, FeedbackResponse, TrainingStatusResponse
from api.services import feedback_service
from api.services.task_queue import FastAPITaskQueue

router = APIRouter(tags=["feedback"])


def _db():
    db = get_database()
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return db


def _submit(body: FeedbackRequest, user_id: str, background_tasks: BackgroundTasks):
    db = _db()
    try:
        return feedback_service.submit_feedback(
            db, user_id, body.session_id, body.corrected_emotion,
            task_queue=FastAPITaskQueue(background_tasks),
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest, background_tasks: BackgroundTasks,
                    user_id: str = Depends(get_current_user_id)):
    return _submit(body, user_id, background_tasks)


@router.post("/api/audio/feedback", response_model=FeedbackResponse)
def submit_feedback_legacy(body: FeedbackRequest, background_tasks: BackgroundTasks,
                           user_id: str = Depends(get_current_user_id)):
    """Legacy alias used by the mobile app — same logic as /api/feedback."""
    return _submit(body, user_id, background_tasks)


@router.get("/api/training-status/{user_id}", response_model=TrainingStatusResponse)
def training_status(user_id: str, current=Depends(get_current_user)):
    if current["user_id"] != user_id and not current.get("is_admin"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    from api.services.training_job_tracker import get_latest_job

    job = get_latest_job(_db(), user_id)
    if not job:
        return TrainingStatusResponse(status=None)

    def iso(v):
        return v.isoformat() if v else None

    return TrainingStatusResponse(
        job_id=job.get("job_id"), status=job.get("status"),
        created_at=iso(job.get("created_at")), started_at=iso(job.get("started_at")),
        completed_at=iso(job.get("completed_at")), error_message=job.get("error_message"),
        metrics=job.get("metrics"),
    )


@router.post("/api/trigger-training/{user_id}")
def trigger_training(user_id: str, background_tasks: BackgroundTasks,
                     _admin=Depends(get_current_admin)):
    from api.services.training_job_tracker import create_training_job
    from training.train_user_head import train_user_head_async

    db = _db()
    job_id = create_training_job(db, user_id)
    FastAPITaskQueue(background_tasks).enqueue(train_user_head_async, user_id, job_id=job_id)
    return {"status": "queued", "training_job_id": job_id,
            "message": f"Training queued for user {user_id}"}
