from fastapi import APIRouter, Depends

from api.middleware.auth import get_current_user_id
from api.schemas.rag import QueryRequest, QueryResponse
from api.services.rag_service import query_rag

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
def rag_query(body: QueryRequest, user_id: str = Depends(get_current_user_id)):
    result = query_rag(user_id, body.query, top_k=body.top_k)
    return QueryResponse(**result)
