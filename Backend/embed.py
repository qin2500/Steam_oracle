import os
import json
import psycopg2
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.messages import HumanMessage, SystemMessage
from langchain_postgres import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Database connection
DB_URI = "postgresql://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db"
# PGVector requires psycopg format (with + sign)
PGVECTOR_CONNECTION = "postgresql+psycopg://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db"

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Initialize embeddings model (will be reused)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)


def setup_database():
    """Enable pgvector extension. PGVector will handle table creation."""
    print("Setting up database...")
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    try:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print(" pgvector extension enabled")
        conn.commit()
        print("Database setup complete!\n")
    except Exception as e:
        conn.rollback()
        print(f"Error setting up database: {e}")
        raise
    finally:
        cur.close()
        conn.close()


def get_vector_store():
    """
    Initialize and return a PGVector store instance.
    This handles embedding generation and storage automatically.
    """
    return PGVector(
        embeddings=embeddings,
        collection_name="game_summaries",
        connection=PGVECTOR_CONNECTION,
        use_jsonb=True
    )


def get_top_games(limit: int = 1000) -> List[Dict]:
    """
    Query the top N games by total review count.

    Args:
        limit: Number of games to fetch (default: 1000)

    Returns:
        List of game dictionaries with game_id, name, and review counts
    """
    print(f"Fetching top {limit} games...")
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT game_id, name, positive_reviews, negative_reviews
            FROM games
            WHERE reviews_processed = true
            ORDER BY (positive_reviews + negative_reviews) DESC
            LIMIT %s;
        """, (limit,))

        games = []
        for row in cur.fetchall():
            games.append({
                'game_id': row[0],
                'name': row[1],
                'positive_reviews': row[2],
                'negative_reviews': row[3],
                'total_reviews': row[2] + row[3]
            })

        print(f" Found {len(games)} games\n")
        return games
    finally:
        cur.close()
        conn.close()


def get_reviews_for_game(game_id: str) -> Dict[str, List[str]]:
    """
    Get reviews for a game using the specified strategy:
    - Top 50 positive reviews by weighted_vote_score
    - Top 50 negative reviews by weighted_vote_score
    - 10 random positive reviews
    - 10 random negative reviews
    """
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    reviews = {
        'top_positive': [],
        'top_negative': [],
        'random_positive': [],
        'random_negative': []
    }

    try:
        # Top 50 positive reviews
        cur.execute("""
            SELECT text
            FROM reviews
            WHERE game_id = %s
                AND is_positive = true
                AND text IS NOT NULL
            ORDER BY weighted_vote_score DESC NULLS LAST
            LIMIT 50;
        """, (game_id,))
        reviews['top_positive'] = [row[0] for row in cur.fetchall()]

        # Top 50 negative reviews
        cur.execute("""
            SELECT text
            FROM reviews
            WHERE game_id = %s
                AND is_positive = false
                AND text IS NOT NULL
            ORDER BY weighted_vote_score DESC NULLS LAST
            LIMIT 50;
        """, (game_id,))
        reviews['top_negative'] = [row[0] for row in cur.fetchall()]

        # 10 random positive reviews
        cur.execute("""
            SELECT text
            FROM reviews
            WHERE game_id = %s
                AND is_positive = true
                AND text IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 10;
        """, (game_id,))
        reviews['random_positive'] = [row[0] for row in cur.fetchall()]

        # 10 random negative reviews
        cur.execute("""
            SELECT text
            FROM reviews
            WHERE game_id = %s
                AND is_positive = false
                AND text IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 10;
        """, (game_id,))
        reviews['random_negative'] = [row[0] for row in cur.fetchall()]

        return reviews
    finally:
        cur.close()
        conn.close()


def truncate_review(text: str, max_chars: int = 500) -> str:
    """Truncate review text to max_chars."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def format_reviews_for_prompt(reviews: Dict[str, List[str]]) -> str:
    """Format reviews into a prompt-friendly string."""
    formatted = []

    for review in reviews['top_positive']:
        formatted.append(f"[TOP POSITIVE] {truncate_review(review)}")

    for review in reviews['top_negative']:
        formatted.append(f"[TOP NEGATIVE] {truncate_review(review)}")

    for review in reviews['random_positive']:
        formatted.append(f"[RANDOM POSITIVE] {truncate_review(review)}")

    for review in reviews['random_negative']:
        formatted.append(f"[RANDOM NEGATIVE] {truncate_review(review)}")

    return "\n\n".join(formatted)


def generate_summary_with_langchain(game_name: str, reviews: Dict[str, List[str]]) -> Optional[Dict]:
    """
    Use LangChain with Claude via OpenRouter to generate a game summary.
    Returns a dict with 'summary', 'positive_themes', and 'negative_themes'.
    """
    # Initialize LangChain ChatOpenAI with OpenRouter
    llm = ChatOpenAI(
        model="anthropic/claude-3.5-sonnet",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.3
    )

    # Format reviews
    review_text = format_reviews_for_prompt(reviews)

    # Create prompt
    system_prompt = """You are an expert game analyst. Analyze Steam reviews and extract key insights.
Return your response as valid JSON only, with no additional text or markdown formatting."""

    user_prompt = f"""Analyze these Steam reviews for the game "{game_name}" and provide:
1. A detailed summary of what players think about the game
2. 5-11 positive themes (short phrases like "balanced gameplay", "great graphics")
3. 5-11 negative themes (short phrases like "poor optimization", "steep learning curve")

Reviews:
{review_text}

Return ONLY a JSON object in this exact format:
{{
  "summary": "Your summary here",
  "positive_themes": ["theme1", "theme2", "theme3", "theme4", "theme5"],
  "negative_themes": ["theme1", "theme2", "theme3", "theme4", "theme5"]
}}"""

    try:
        # Make API call
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages)
        response_text = response.content.strip()

        # Parse JSON response
        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        result = json.loads(response_text)
        return result
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None




def process_game(game: Dict, index: int, total: int, vector_store: PGVector) -> bool:
    """
    Process a single game: get reviews, generate summary, create embedding, store results.
    Uses LangChain's PGVector to automatically handle embedding and storage.
    Returns True if successful, False otherwise.
    """
    game_id = game['game_id']
    game_name = game['name']

    print(f"[{index}/{total}] Processing: {game_name} ({game_id})")

    try:
        # Get reviews
        reviews = get_reviews_for_game(game_id)
        review_count = (len(reviews['top_positive']) + len(reviews['top_negative']) +
                       len(reviews['random_positive']) + len(reviews['random_negative']))

        if review_count == 0:
            print(f" No reviews found, skipping")
            return False

        print(f"  → Found {review_count} reviews")

        # Generate summary using Claude
        summary_data = generate_summary_with_langchain(game_name, reviews)
        if not summary_data:
            return False

        print(f"  → Generated summary")

        # Create a Document object with the summary and metadata
        doc = Document(
            page_content=summary_data['summary'],
            metadata={
                'game_id': game_id,
                'game_name': game_name,
                'positive_themes': summary_data['positive_themes'],
                'negative_themes': summary_data['negative_themes'],
                'review_count_used': review_count
            }
        )

        # Delete existing document if it exists (upsert behavior)
        try:
            vector_store.delete(ids=[game_id])
            print(f"  → Deleted existing entry")
        except Exception:
            # No existing entry to delete, which is fine
            pass

        # Add to vector store
        vector_store.add_documents(
            documents=[doc],
            ids=[game_id]
        )

        print(f"  ✓ Stored with embedding")

        return True
    except Exception as e:
        print(f"Error processing game: {e}")
        return False


def main():
    """
    Main execution function.

    Args:
        limit: Number of top games to process (default: 1000)
    """
    print("=" * 60)
    print("Steam Game Review Embedding Pipeline")
    print("=" * 60)
    print()

    setup_database()

    # Initialize vector store
    print("Initializing PGVector store...")
    vector_store = get_vector_store()
    print("✓ Vector store initialized\n")

    # Get top N games
    games = get_top_games(100)

    # Process each game
    print(f"Processing {len(games)} games...")
    print("-" * 60)

    success_count = 0
    failure_count = 0

    for i, game in enumerate(games, 1):
        if process_game(game, i, len(games), vector_store):
            success_count += 1
        else:
            failure_count += 1
        print()

    # Print summary
    print("=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"✓ Successful: {success_count}")
    print(f"✗ Failed: {failure_count}")
    print(f"Total: {len(games)}")
    print()

    # Validation
    print("Validating results...")
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    # PGVector creates a table named: langchain_pg_collection and langchain_pg_embedding
    cur.execute("""
        SELECT COUNT(*)
        FROM langchain_pg_embedding lpe
        JOIN langchain_pg_collection lpc ON lpe.collection_id = lpc.uuid
        WHERE lpc.name = 'game_summaries';
    """)
    total_rows = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*)
        FROM langchain_pg_embedding lpe
        JOIN langchain_pg_collection lpc ON lpe.collection_id = lpc.uuid
        WHERE lpc.name = 'game_summaries'
        AND lpe.embedding IS NOT NULL;
    """)
    with_embeddings = cur.fetchone()[0]

    print(f"✓ Total documents in collection: {total_rows}")
    print(f"✓ Documents with embeddings: {with_embeddings}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
