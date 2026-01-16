from typing import List, Dict, Any
from langchain_core.output_parsers import JsonOutputParser

from ..models.state import AgentState, GameCriteria, GradedResult, GameCandidate, GradingOutput
from ..services.llm import get_llm
from ..services.vector_store import get_vector_store
from ..services.sql_tool import get_filtered_game_ids
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
    
    # FILTERING: Check if we need to apply SQL filters
    filtered_ids = get_filtered_game_ids(criteria)
    
    # Construct filter args for PGVector if ids are returned
    # PGVector uses a specific syntax for metadata/id filtering depending on the adapter
    # For langchain-postgres PGVector, it usually accepts a 'filter' dict.
    # But filtering by ID is often special.
    # Note: PGVector's similarity_search accepts a `filter` argument which is passed to the store.
    # For Postgres, it is usually a JSON match on metadata.
    # However, since we want to filter by the primary ID (which might be in metadata or DB primary key),
    # The safest way given standard usage is to filter WHERE metadata->>'game_id' IN (...)
    
    search_kwargs = {"k": 25}
    if filtered_ids is not None:
        if len(filtered_ids) == 0:
            # Filters matched nothing! Return empty result immediately
            return {"candidates": []}
            
        print(f"Applying SQL filter: restricted to {len(filtered_ids)} games")
        # Construct metadata filter
        # Syntax depends on the underlying driver but normally it's:
        # filter = {"game_id": {"$in": ["id1", "id2"]}}
        search_kwargs["filter"] = {"game_id": {"$in": filtered_ids}}
    
    results = vector_store.similarity_search_with_score(search_query, **search_kwargs)
    
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
