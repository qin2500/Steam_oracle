from fastapi import HTTPException
from typing import Dict, Any, AsyncGenerator
from src.agent.graph import get_graph
from src.api.schemas import SearchRequest
import json
import asyncio

# Initialize graph once at module level
graph_app = get_graph()

def format_game_response(game: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the internal game result for the API response.
    Filters out deal_breakers from assessments.
    """
    assessments = game.get('criteria_assessments', [])
    # If they are Pydantic objects, dump them; if dicts, use directly
    assessments_data = [
        asm.model_dump() if hasattr(asm, 'model_dump') else asm 
        for asm in assessments
    ]
    
    # Filter out deal breakers for the UI
    filtered_assessments = [
        asm for asm in assessments_data
        if asm.get('category') != 'deal_breaker'
    ]
    
    return {
        "id": game.get('game_id'),
        "name": game.get('name'),
        "score": game.get('match_score'),
        "reasoning": game.get('reasoning'),
        "assessments": filtered_assessments,
        "summary": game.get('summary')
    }

async def handle_search_stream(request: SearchRequest) -> AsyncGenerator[str, None]:
    """
    Streams the search process as Server-Sent Events (SSE).
    """
    try:
        initial_state = {
            "query": request.query,
            "interpreted_needs": None,
            "candidates": [],
            "final_results": []
        }
        
        # Stream the graph execution
        # We need to stream updates to the state
        async for chunk in graph_app.astream(initial_state):
            # Inspect the chunk to see which node finished
            
            # 1. Interpreter Finished
            if "interpret" in chunk:
                criteria = chunk["interpret"]["interpreted_needs"]
                data = criteria.model_dump()
                # Yield an event
                yield json.dumps({"type": "optimization", "data": data}) + "\n"
                
            # 2. Retriever Finished (Optional, can show "Found 25 candidates")
            elif "retrieve" in chunk:
                yield json.dumps({"type": "progress", "message": "Analyzing candidates..."}) + "\n"
                
            # 3. Grader Finished (Final Result)
            elif "grade" in chunk:
                graded_results = chunk["grade"]["final_results"]
                formatted = [format_game_response(g) for g in graded_results]
                
                final_payload = {
                    "type": "result",
                    "data": formatted
                }
                yield json.dumps(final_payload) + "\n"
                
    except Exception as e:
        print(f"Streaming Error: {str(e)}")
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"
