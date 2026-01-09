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
        # Default query for testing
        query = "I want a chill sci-fi game with base building that isn't too expensive"
    else:
        query = sys.argv[1]
        
    print(f"\n🚀 Agentic Search: '{query}'")
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
    # We use .invoke for simple execution
    try:
        result = app.invoke(initial_state)
        
        # Display Results
        
        # 1. Interpretation
        criteria = result['interpreted_needs']
        print(f"\n🧠 Interpretation:")
        print(f"   • Mechanics: {', '.join(criteria.primary_mechanics)}")
        print(f"   • Atmosphere: {', '.join(criteria.atmosphere)}")
        print(f"   • Avoid: {', '.join(criteria.deal_breakers)}")
        print(f"   • Search Terms: '{criteria.search_terms}'")
        
        # 2. Results
        graded = result['final_results']
        print(f"\n🏆 Top Matches ({len(graded)} found):")
        
        for i, game in enumerate(graded[:3], 1): # Show top 3
            print(f"\n   {i}. {game['name']} (Score: {game['match_score']})")
            print(f"      Matched: {', '.join(game['matched_criteria'])}")
            if game['missing_criteria']:
                print(f"      Missing: {', '.join(game['missing_criteria'])}")
            print(f"      Reasoning: {game['reasoning']}")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
