from datetime import datetime, timezone

from pymongo.collection import Collection
from pymongo.database import Database

VALID_STATUSES = {"queued", "running", "completed", "failed"}


class TrainingJob:
    def __init__(
        self,
        user_id: str,
        job_id: str,
        status: str = "queued",
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        error_message: str | None = None,
        metrics: dict | None = None,
    ):
        self.user_id = user_id
        self.job_id = job_id
        self.status = status
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.started_at = started_at
        self.completed_at = completed_at
        self.error_message = error_message
        self.metrics = metrics

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }

    @staticmethod
    def get_collection(db: Database) -> Collection:
        return db["training_jobs"]
