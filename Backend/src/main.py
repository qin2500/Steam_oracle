import sys
import os
import json

# Add current directory to path so we can import src modules if running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.graph import get_graph
from src.config.settings import settings

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <query>")
        print('Example: python main.py "Soulslike for casuals"')
        # query = "I want a chill sci-fi game with base building that isn't too expensive"
    else:
        query = sys.argv[1]
        
    print(f"\nAgentic Search: '{query}'")
    print("--------------------------------------------------")
    
    app = get_graph()
    
    # Initialize state
    initial_state = {
        "query": query,
        "interpreted_needs": None,
        "candidates": [],
        "final_results": []
    }
    
    # Run the graph
    try:
        result = app.invoke(initial_state)
        
        # Process results for output
        final_output = {
            "query": result['query'],
            "interpretation": result['interpreted_needs'].model_dump(),
            "results": []
        }
        
        for game in result['final_results']:
            # Filter out deal_breakers from assessments for the user view
            filtered_assessments = [
                asm for asm in game['criteria_assessments'] 
                if asm.category != 'deal_breaker'
            ]
            
            game_data = {
                "id": game['game_id'],
                "name": game['name'],
                "score": game['match_score'],
                "reasoning": game['reasoning'],
                "assessments": [asm.model_dump() for asm in filtered_assessments],
                "summary": game['summary']
            }
            final_output["results"].append(game_data)
            
        # Output ONLY raw JSON
        print(json.dumps(final_output, indent=2))
            
    except Exception as e:
        # Output error as JSON too so frontend can handle it
        error_response = {"error": str(e)}
        print(json.dumps(error_response))
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    main()
