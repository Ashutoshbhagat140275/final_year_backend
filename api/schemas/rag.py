from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class Source(BaseModel):
    text: str
    session_id: str | None = None
    timestamp: str | None = None
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    query: str
