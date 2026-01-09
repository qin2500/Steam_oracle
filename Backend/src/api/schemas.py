from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
# Import internal models to reuse their definitions if valid for API response
from src.models.state import GradedResult, GameCriteria

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="The natural language query from the user")

class ErrorResponse(BaseModel):
    error: str

class SearchResponse(BaseModel):
    query: str
    interpretation: Dict[str, Any] # Or use GameCriteria.model_dump()
    results: List[Dict[str, Any]] # Or define a strict ResponseGame model
