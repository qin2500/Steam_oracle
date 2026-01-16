from typing import List, Dict, Optional, TypedDict, Any
from pydantic import BaseModel, Field

class GameCriteria(BaseModel):
    """
    The structured interpretation of the user's game request.
    This is the 'search plan' for the agent.
    """
    primary_mechanics: List[str] = Field(
        default_factory=list,
        description="List of core gameplay loops or mechanics (e.g., 'deck building', 'worker placement', 'FPS')"
    )
    atmosphere: List[str] = Field(
        default_factory=list,
        description="Vibe, mood, or artistic style (e.g., 'cozy', 'oppressive', 'cyberpunk', 'pixel art')"
    )
    deal_breakers: List[str] = Field(
        default_factory=list,
        description="Features or elements to explicitly avoid (e.g., 'microtransactions', 'permadeath', 'anime')"
    )
    search_terms: str = Field(
        ...,
        description="A highly optimized natural language string to use for the vector similarity search."
    )
    
    # New filters for Hybrid Search
    min_reviews: Optional[int] = Field(None, description="Minimum number of total reviews (for popularity)")
    min_rating: Optional[float] = Field(None, description="Minimum rating from 0.0 to 1.0 (positive/total)")
    price_max: Optional[int] = Field(None, description="Maximum price in cents")
    is_free: Optional[bool] = Field(None, description="If true, only show free games")
    release_year: Optional[int] = Field(None, description="Specific release year to filter by")
    genres: Optional[List[str]] = Field(None, description="List of genres to strictly filter by")

class GameCandidate(TypedDict):
    """
    Represents a game retrieved from the vector store.
    """
    game_id: str
    name: str
    summary: str
    similarity_score: float
    metadata: Dict[str, Any]

class CriterionAssessment(BaseModel):
    criterion: str = Field(..., description="The specific requirement being checked")
    category: str = Field(..., description="'mechanic', 'atmosphere', or 'deal_breaker'")
    status: str = Field(..., description="'met' (green), 'partial' (yellow), 'missing' (red), or 'violated' (fatal)")
    confidence: Optional[int] = Field(default=0, description="0-100 confidence level")
    evidence: Optional[str] = Field(default="", description="Quote or reasoning from the summary")

class GradingOutput(BaseModel):
    """
    The strict output schema for the Grader LLM.
    """
    match_score: int = Field(..., description="Score 0-100")
    criteria_assessments: List[CriterionAssessment] = Field(..., description="Assessment of each criterion")
    reasoning: str = Field(..., description="Short summary of the grading decision")

class GradedResult(GameCandidate):
    """
    A game candidate that has been graded/scored by the agent.
    """
    match_score: int # 0-100
    criteria_assessments: List[CriterionAssessment]
    reasoning: str # Overall summary reasoning

class AgentState(TypedDict):
    """
    The main state object passed through the LangGraph workflow.
    """
    query: str
    interpreted_needs: Optional[GameCriteria]
    candidates: List[GameCandidate]
    final_results: List[GradedResult]
