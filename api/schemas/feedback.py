from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    session_id: str
    corrected_emotion: str


class FeedbackResponse(BaseModel):
    status: str
    feedback_count: int
    training_triggered: bool
    training_job_id: str | None = None
    message: str


class TrainingStatusResponse(BaseModel):
    job_id: str | None = None
    status: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    metrics: dict | None = None
