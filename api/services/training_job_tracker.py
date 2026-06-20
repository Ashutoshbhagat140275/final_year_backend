"""Track per-user training jobs in MongoDB."""
import uuid
from datetime import datetime, timezone

from api.models.training_job import VALID_STATUSES, TrainingJob


def create_training_job(db, user_id: str) -> str:
    job_id = str(uuid.uuid4())
    TrainingJob.get_collection(db).insert_one(TrainingJob(user_id=user_id, job_id=job_id).to_dict())
    return job_id


def update_job_status(db, job_id: str, status: str, error_message: str | None = None,
                      metrics: dict | None = None) -> None:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    now = datetime.now(timezone.utc)
    update = {"status": status, "updated_at": now}
    if status == "running":
        update["started_at"] = now
    if status in ("completed", "failed"):
        update["completed_at"] = now
    if error_message is not None:
        update["error_message"] = error_message
    if metrics is not None:
        update["metrics"] = metrics
    TrainingJob.get_collection(db).update_one({"job_id": job_id}, {"$set": update})


def get_latest_job(db, user_id: str) -> dict | None:
    return TrainingJob.get_collection(db).find_one(
        {"user_id": user_id}, sort=[("created_at", -1)]
    )


def get_job_by_id(db, job_id: str) -> dict | None:
    return TrainingJob.get_collection(db).find_one({"job_id": job_id})
