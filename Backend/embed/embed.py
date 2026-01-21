import os
import json
import psycopg2
import concurrent.futures
import argparse
import time
import warnings
import logging
import sys
from contextlib import contextmanager
from typing import List, Dict, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.messages import HumanMessage, SystemMessage
from langchain_postgres import PGVector
from langchain_core.documents import Document
from dotenv import load_dotenv

# explicitly load .env from parent directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
env_path = os.path.join(parent_dir, '.env')
load_dotenv(env_path)

# Suppress warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger('sqlalchemy.engine').setLevel(logging.ERROR)

# Database connection
DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise ValueError("DB_URI environment variable not set")

# PGVector requires psycopg format (with + sign)
# Append options to silence warnings
if DB_URI.startswith("postgresql://"):
    PGVECTOR_CONNECTION = DB_URI.replace("postgresql://", "postgresql+psycopg://", 1)
else:
    PGVECTOR_CONNECTION = DB_URI

if "?" not in PGVECTOR_CONNECTION:
    PGVECTOR_CONNECTION += "?options=-c%20client_min_messages=ERROR"
else:
    PGVECTOR_CONNECTION += "&options=-c%20client_min_messages=ERROR"

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable not set")

# OpenRouter base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Initialize embeddings model (stateless/thread-safe enough for our usage)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr (including C-level output) using os.dup2."""
    try:
        # Open devnull
        with open(os.devnull, 'w') as fnull:
            # Save original stderr fd
            try:
                old_stderr_fd = os.dup(sys.stderr.fileno())
            except Exception:
                # If we can't dup (e.g. no console), just yield
                yield
                return

            try:
                # Redirect stderr to devnull
                os.dup2(fnull.fileno(), sys.stderr.fileno())
                yield
            finally:
                # Restore stderr
                os.dup2(old_stderr_fd, sys.stderr.fileno())
                os.close(old_stderr_fd)
    except Exception:
         # Fallback if anything goes wrong with file descriptors
         yield


def get_db_connection():
    """Create a database connection."""
    # Pass options to silence server-side warnings (best effort)
    conn = psycopg2.connect(DB_URI, options="-c client_min_messages=ERROR")
    return conn


def setup_database():
    """Enable pgvector extension. PGVector will handle table creation."""
    print("Setting up database...")
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error setting up database: {e}")
        raise


def get_vector_store():
    """
    Initialize and return a PGVector store instance.
    This handles embedding generation and storage automatically.
    """
    return PGVector(
        embeddings=embeddings,
        collection_name="game_summaries",
        connection=PGVECTOR_CONNECTION,
        use_jsonb=True,
    )


def get_top_games(limit: Optional[int] = None, skip_existing: bool = False) -> List[Dict]:
    """
    Query the top N games by total review count.
    If limit is None, fetch all games.

    Args:
        limit: Number of games to fetch (default: None for all)
        skip_existing: If True, exclude games that already have an embedding.

    Returns:
        List of game dictionaries with game_id, name, and review counts
    """
    limit_str = f"{limit}" if limit else "ALL"
    print(f"Fetching top games (Limit: {limit_str}, Skip Existing: {skip_existing})...")
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Base query
        query = """
            SELECT game_id, name, positive_reviews, negative_reviews
            FROM games
            WHERE reviews_processed = true
        """
        
        # Add exclusion filter if requested
        if skip_existing:
            query += """
                AND game_id NOT IN (
                    SELECT lpe.cmetadata->>'game_id'
                    FROM langchain_pg_embedding lpe
                    JOIN langchain_pg_collection lpc ON lpe.collection_id = lpc.uuid
                    WHERE lpc.name = 'game_summaries'
                )
            """
            
        query += " ORDER BY (positive_reviews + negative_reviews) DESC"
        
        params = ()
        if limit:
            query += " LIMIT %s"
            params = (limit,)
            
        cur.execute(query, params)

        games = []
        for row in cur.fetchall():
            games.append({
                'game_id': row[0],
                'name': row[1],
                'positive_reviews': row[2],
                'negative_reviews': row[3],
                'total_reviews': row[2] + row[3]
            })
        
        cur.close()
        conn.close()

        print(f" Found {len(games)} games to process\n")
        return games
    except Exception as e:
        print(f"Error fetching top games: {e}")
        raise


def get_reviews_for_game(game_id: str) -> Dict[str, List[str]]:
    """
    Get reviews for a game using the specified strategy:
    - Top 50 positive reviews by weighted_vote_score
    - Top 50 negative reviews by weighted_vote_score
    - 10 random positive reviews
    - 10 random negative reviews
    """
    conn = get_db_connection()
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

        # Suppress stderr here just in case LangChain logs something
        with suppress_stderr():
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
        print(f"Error generating summary for {game_name}: {e}")
        return None


def process_game(game: Dict) -> bool:
    """
    Process a single game: get reviews, generate summary, create embedding, store results.
    Each worker initializes its own vector store to ensure DB connection safety.
    Returns True if successful, False otherwise.
    """
    try:
        # Initialize thread-local vector store
        vector_store = get_vector_store()
        
        game_id = game['game_id']
        game_name = game['name']
        
        # print(f"Processing: {game_name} ({game_id})")

        # Get reviews
        reviews = get_reviews_for_game(game_id)
        review_count = (len(reviews['top_positive']) + len(reviews['top_negative']) +
                       len(reviews['random_positive']) + len(reviews['random_negative']))

        if review_count == 0:
            print(f"Skipping {game_name} - No reviews")
            return False

        # Generate summary using Claude
        summary_data = generate_summary_with_langchain(game_name, reviews)
        if not summary_data:
            print(f"Failed to generate summary for {game_name}")
            return False

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
        # Note: In a highly concurrent environment with duplicate game IDs this could be racy,
        # but since we iterate distinct games from the DB, it should be fine.
        try:
            vector_store.delete(ids=[game_id])
        except Exception:
            pass

        # Add to vector store
        vector_store.add_documents(
            documents=[doc],
            ids=[game_id]
        )

        print(f"SUCCESS: {game_name}")
        return True

    except Exception as e:
        print(f"ERROR processing {game.get('name', 'Unknown')}: {e}")
        return False


def main():
    """
    Main execution function.
    """
    try:
        parser = argparse.ArgumentParser(description='Steam Game Review Embedding Pipeline')
        parser.add_argument('--limit', type=int, default=None, help='Number of top games to process (default: All)')
        parser.add_argument('--workers', type=int, default=5, help='Number of parallel workers')
        parser.add_argument('--skip-existing', action='store_true', help='Skip games that already have an embedding')
        args = parser.parse_args()

        print("=" * 60)
        print("Steam Game Review Embedding Pipeline (Parallel)")
        print("=" * 60)
        print(f"Limit: {args.limit if args.limit else 'ALL'}")
        print(f"Workers: {args.workers}")
        print(f"Skip Existing: {args.skip_existing}")
        print()

        # Suppress stderr for setup and validation as well
        with suppress_stderr():
            setup_database()
            # Get top N games
            games = get_top_games(args.limit, args.skip_existing)

        # Process games in parallel
        print(f"Processing {len(games)} games with {args.workers} workers...")
        print("-" * 60)

        success_count = 0
        failure_count = 0
        
        start_time = time.time()

        # Suppress stderr GLOBAL for parallel execution to avoid C-level thread output leaking
        with suppress_stderr():
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
                # Submit all tasks
                future_to_game = {executor.submit(process_game, game): game for game in games}
                
                # Process results as they complete
                for i, future in enumerate(concurrent.futures.as_completed(future_to_game), 1):
                    game = future_to_game[future]
                    try:
                        result = future.result()
                        if result:
                            success_count += 1
                        else:
                            failure_count += 1
                    except Exception as exc:
                        # This prints to stdout, so it's visible even with stderr suppressed
                        print(f'{game["name"]} generated an exception: {exc}')
                        failure_count += 1
                    
                    # We can print to stdout
                    if i % 1 == 0:
                        print(f"Progress: {i}/{len(games)} | Success: {success_count} | Fail: {failure_count}")

        duration = time.time() - start_time
        
        # Print summary
        print("=" * 60)
        print("Processing Complete!")
        print("=" * 60)
        print(f"Time taken: {duration:.2f} seconds")
        print(f"✓ Successful: {success_count}")
        print(f"✗ Failed: {failure_count}")
        print(f"Total: {len(games)}")
        print()

        # Validation
        print("Validating results...")
        
        with suppress_stderr():
            conn = get_db_connection()
            cur = conn.cursor()

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

            cur.close()
            conn.close()

        print(f"✓ Total documents in collection: {total_rows}")
        print(f"✓ Documents with embeddings: {with_embeddings}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Fatal error: {e}")


if __name__ == "__main__":
    main()
