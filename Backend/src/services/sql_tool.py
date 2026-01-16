from typing import List, Optional
import psycopg
from ..config.settings import settings
from ..models.state import GameCriteria

def get_filtered_game_ids(criteria: GameCriteria) -> Optional[List[str]]:
    """
    Constructs and executes a SQL query based on the provided criteria.
    Returns a list of matching game_ids.
    Returns None if no filters are applied (meaning "all games").
    """
    
    # Check if we have any filters that require SQL
    # We only care if there are constraints stronger than default
    has_filters = (
        criteria.min_reviews is not None or
        criteria.min_rating is not None or
        criteria.price_max is not None or
        criteria.is_free is not None or
        (criteria.genres and len(criteria.genres) > 0) or
        criteria.release_year is not None
    )
    
    if not has_filters:
        return None
        
    query_parts = ["SELECT game_id FROM games WHERE 1=1"]
    params = []
    
    # 1. Popularity (Review Count)
    if criteria.min_reviews is not None:
        query_parts.append("AND (positive_reviews + negative_reviews) >= %s")
        params.append(criteria.min_reviews)
        
    # 2. Rating (Positive / Total)
    if criteria.min_rating is not None:
        # Avoid division by zero by ensuring at least 1 review if we filter by rating
        query_parts.append("""
            AND (positive_reviews + negative_reviews) > 0 
            AND (CAST(positive_reviews AS FLOAT) / (positive_reviews + negative_reviews)) >= %s
        """)
        params.append(criteria.min_rating)
        
    # 3. Price
    if criteria.price_max is not None:
        query_parts.append("AND price <= %s")
        params.append(criteria.price_max)
        
    # 4. Free to Play
    if criteria.is_free is not None:
        query_parts.append("AND is_free = %s")
        params.append(criteria.is_free)

    # 5. Release Year
    if criteria.release_year is not None:
        query_parts.append("AND EXTRACT(YEAR FROM release_date) = %s")
        params.append(criteria.release_year)

    # 6. Genres 
    # The genres column is type 'text', likely comma-separated or JSON string.
    # Looking at schema.txt it says 'text'. Assuming comma separated or simple string for now.
    # We will use ILIKE for basic matching.
    if criteria.genres:
        for genre in criteria.genres:
            query_parts.append("AND genres ILIKE %s")
            params.append(f"%{genre}%")
            
    # Limit to prevent returning 100k IDs if the filter is too loose
    query_parts.append("LIMIT 10000")
    
    full_query = "\n".join(query_parts)
    
    # Execute
    conn = None
    try:
        conn = psycopg2.connect(settings.DB_URI)
        cur = conn.cursor()
        cur.execute(full_query, tuple(params))
        rows = cur.fetchall()
        
        return [row[0] for row in rows]
        
    except Exception as e:
        print(f"SQL Filtering Error: {e}")
        # On error, fallback to vector search only (return None)
        return None
    finally:
        if conn:
            conn.close()
