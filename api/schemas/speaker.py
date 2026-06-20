from pydantic import BaseModel


class EnrollmentStatus(BaseModel):
    enrolled: bool
    enrollment_state: str
    samples_collected: int
    required_samples: int
    max_samples: int
    updated_at: str | None = None


class MessageResponse(BaseModel):
    message: str
    samples_collected: int | None = None
    enrolled: bool | None = None
    sample_count: int | None = None
