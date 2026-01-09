from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.api.schemas import SearchRequest
from src.api.controllers import handle_search_stream

router = APIRouter()

@router.post("/search")
async def search_games(request: SearchRequest):
    """
    Execute the agentic search workflow.
    Returns a stream of JSON events (NDJSON).
    """
    return StreamingResponse(
        handle_search_stream(request), 
        media_type="application/x-ndjson"
    )
