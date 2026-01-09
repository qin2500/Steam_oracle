from typing import List, Dict, Any
from langchain_core.output_parsers import JsonOutputParser

from ..models.state import AgentState, GameCriteria, GradedResult, GameCandidate
from ..services.llm import get_llm
from ..services.vector_store import get_vector_store
from .prompts import interpreter_prompt, grader_prompt

# ------------------------------------------------------------------------------
# NODE: INTERPRETER
# ------------------------------------------------------------------------------
def interpret(state: AgentState) -> Dict[str, Any]:
    """
    Analyzes the user's query to extract structured criteria.
    """
    query = state['query']
    
    # Use a structured LLM
    llm = get_llm(temperature=0).with_structured_output(GameCriteria)
    chain = interpreter_prompt | llm
    
    criteria = chain.invoke({"query": query})
    
    return {"interpreted_needs": criteria}

# ------------------------------------------------------------------------------
# NODE: RETRIEVER
# ------------------------------------------------------------------------------
def retrieve(state: AgentState) -> Dict[str, Any]:
    """
    Uses the optimized search terms to fetch candidates from vector store.
    """
    criteria = state['interpreted_needs']
    search_query = criteria.search_terms
    
    vector_store = get_vector_store()
    
    # Get top 25 candidates to allow for filtering
    # PGVector returns 'Document' objects
    docs = vector_store.similarity_search(search_query, k=25)
    
    candidates: List[GameCandidate] = []
    
    for doc in docs:
        # Check if we have parsing issues (sometimes metadata is missing)
        metadata = doc.metadata or {}
        candidate: GameCandidate = {
            "game_id": metadata.get("game_id", "unknown"),
            "name": metadata.get("game_name", "Unknown Game"),
            "summary": doc.page_content,
            "similarity_score": 0.0, # Not strictly available in basic similarity_search, use similarity_search_with_score if needed
            "metadata": metadata
        }
        candidates.append(candidate)
        
    return {"candidates": candidates}

# ------------------------------------------------------------------------------
# NODE: GRADER
# ------------------------------------------------------------------------------
def grade(state: AgentState) -> Dict[str, Any]:
    """
    Scores each candidate against the criteria using a lighter LLM.
    """
    candidates = state['candidates']
    criteria = state['interpreted_needs']
    
    # Use a faster model for the loop
    # In a real app, you might want to async batch this
    llm = get_llm(model_name="anthropic/claude-3-haiku", temperature=0) # Or same model if unavailable
    chain = grader_prompt | llm | JsonOutputParser()
    
    final_results: List[GradedResult] = []
    
    # Format criteria for the prompt
    mechanics_str = ", ".join(criteria.primary_mechanics)
    atmosphere_str = ", ".join(criteria.atmosphere)
    deal_breakers_str = ", ".join(criteria.deal_breakers)
    
    # Limit to top 5 for speed in this demo, typically you'd do more
    # Or use map-reduce
    for candidate in candidates[:5]: 
        try:
            grade_data = chain.invoke({
                "summary": candidate['summary'],
                "primary_mechanics": mechanics_str,
                "atmosphere": atmosphere_str,
                "deal_breakers": deal_breakers_str
            })
            
            result: GradedResult = {
                **candidate,
                "match_score": grade_data.get("match_score", 0),
                "matched_criteria": grade_data.get("matched_criteria", []),
                "missing_criteria": grade_data.get("missing_criteria", []),
                "reasoning": grade_data.get("reasoning", "")
            }
            final_results.append(result)
        except Exception as e:
            print(f"Error grading {candidate['name']}: {e}")
            continue

    # Sort by score
    final_results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return {"final_results": final_results}
