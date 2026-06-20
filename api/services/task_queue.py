"""
Background task queue abstraction. FastAPITaskQueue wraps FastAPI BackgroundTasks
(in-process, non-persistent — swap Celery/RQ for production durability).
"""
import uuid
from abc import ABC, abstractmethod


class TaskQueue(ABC):
    @abstractmethod
    def enqueue(self, func, *args, **kwargs) -> str:
        ...


class FastAPITaskQueue(TaskQueue):
    def __init__(self, background_tasks):
        self._bg = background_tasks

    def enqueue(self, func, *args, **kwargs) -> str:
        job_id = kwargs.pop("job_id", None) or str(uuid.uuid4())
        self._bg.add_task(func, job_id, *args, **kwargs)
        return job_id
