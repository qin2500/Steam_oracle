from typing import List, Dict, Any
from langchain_core.output_parsers import JsonOutputParser

from ..models.state import AgentState, GameCriteria, GradedResult, GameCandidate, GradingOutput
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
    
    # Returns List[Tuple[Document, float]]
    results = vector_store.similarity_search_with_score(search_query, k=25)
    
    candidates: List[GameCandidate] = []
    
    for doc, score in results:
        # Check if we have parsing issues (sometimes metadata is missing)
        metadata = doc.metadata or {}
        candidate: GameCandidate = {
            "game_id": metadata.get("game_id", "unknown"),
            "name": metadata.get("game_name", "Unknown Game"),
            "summary": doc.page_content,
            "similarity_score": score,
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
    
    llm = get_llm(model_name="anthropic/claude-3-haiku", temperature=0)
    # Use Structured Output for robustness
    structured_llm = llm.with_structured_output(GradingOutput)
    chain = grader_prompt | structured_llm
    
    final_results: List[GradedResult] = []
    
    # Format criteria for the prompt
    mechanics_str = ", ".join(criteria.primary_mechanics)
    atmosphere_str = ", ".join(criteria.atmosphere)
    deal_breakers_str = ", ".join(criteria.deal_breakers)
    
    # Prepare inputs for all candidates
    batch_inputs = []
    
    for candidate in candidates: 
        batch_inputs.append({
            "summary": candidate['summary'],
            "primary_mechanics": mechanics_str,
            "atmosphere": atmosphere_str,
            "deal_breakers": deal_breakers_str
        })
    
    #Run all batches in parallel
    print(f"   ... Parallel grading of {len(batch_inputs)} candidates ...")
    # return_exceptions=True ensures one failure doesn't crash the whole batch
    batch_results = chain.batch(batch_inputs, config={"max_concurrency": 5, "return_exceptions": True})

    #Combine original candidate data with new grading scores
    for candidate, grade_output in zip(candidates, batch_results):
        try:
            # Check for batch errors
            if isinstance(grade_output, Exception):
                print(f"Error grading {candidate['name']}: {grade_output}")
                continue
                
            # grade_output is chosen by tool calling, allowing safer parsing
            # It will be an instance of GradingOutput
            
            result: GradedResult = {
                **candidate,
                "match_score": grade_output.match_score,
                "criteria_assessments": grade_output.criteria_assessments,
                "reasoning": grade_output.reasoning
            }
            final_results.append(result)
        except Exception as e:
            print(f"Error processing result for {candidate['name']}: {e}")
            continue

    # Sort by score
    final_results.sort(key=lambda x: x['match_score'], reverse=True)
    
    return {"final_results": final_results}
