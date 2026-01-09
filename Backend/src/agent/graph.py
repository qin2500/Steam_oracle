from langgraph.graph import StateGraph, END
from ..models.state import AgentState
from .nodes import interpret, retrieve, grade

def get_graph():
    """
    Constructs and compiles the Agentic Search Graph.
    """
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("interpret", interpret)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    
    # Add Edges
    # Flow: Interpret -> Retrieve -> Grade -> End
    workflow.set_entry_point("interpret")
    
    workflow.add_edge("interpret", "retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", END)
    
    # Compile
    app = workflow.compile()
    
    return app
