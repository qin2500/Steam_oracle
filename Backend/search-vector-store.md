# Searching the Vector Store with LangChain and OpenRouter

This guide explains how to perform semantic search on your Steam game summaries using the vector embeddings stored in PostgreSQL with pgvector.

## Overview

Your vector store uses LangChain's `PGVector` integration, which stores data in two tables:
- `langchain_pg_collection`: Collection metadata (collection name: "game_summaries")
- `langchain_pg_embedding`: Document texts, embeddings (1536-dimensional vectors), and metadata

Each document contains:
- `document`: The game summary text
- `embedding`: 1536-dimensional vector (from `text-embedding-3-small`)
- `cmetadata`: JSON metadata with:
  - `game_id`: Unique game identifier
  - `game_name`: Name of the game
  - `positive_themes`: List of positive review themes
  - `negative_themes`: List of negative review themes
  - `review_count_used`: Number of reviews processed

You can search this vector store to find games similar to a user's query based on semantic meaning.

---

## Method 1: Direct SQL Query with Cosine Similarity

This approach uses raw SQL queries with pgvector's cosine similarity operator on LangChain's PGVector tables.

### Example Code

```python
import os
import psycopg2
from langchain_openai import OpenAIEmbeddings

# Configuration
DB_URI = "postgresql://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def search_games(query: str, top_k: int = 10):
    """
    Search for games similar to the query using semantic search.

    Args:
        query: User's search query (e.g., "fast-paced shooter with good graphics")
        top_k: Number of results to return

    Returns:
        List of matching games with similarity scores
    """
    # Step 1: Generate embedding for the query
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL
    )

    query_embedding = embeddings.embed_query(query)

    # Step 2: Query the database using cosine similarity
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    try:
        # Query PGVector's langchain tables
        # Use <=> operator for cosine distance (lower is more similar)
        cur.execute("""
            SELECT
                lpe.cmetadata->>'game_id' AS game_id,
                lpe.cmetadata->>'game_name' AS game_name,
                lpe.document AS summary,
                lpe.cmetadata->'positive_themes' AS positive_themes,
                lpe.cmetadata->'negative_themes' AS negative_themes,
                lpe.cmetadata->>'review_count_used' AS review_count,
                1 - (lpe.embedding <=> %s::vector) AS similarity_score
            FROM langchain_pg_embedding lpe
            JOIN langchain_pg_collection lpc ON lpe.collection_id = lpc.uuid
            WHERE lpc.name = 'game_summaries'
            ORDER BY lpe.embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, top_k))

        results = []
        for row in cur.fetchall():
            results.append({
                'game_id': row[0],
                'name': row[1],
                'summary': row[2],
                'positive_themes': row[3],
                'negative_themes': row[4],
                'review_count': row[5],
                'similarity_score': float(row[6])
            })

        return results
    finally:
        cur.close()
        conn.close()


# Example usage
if __name__ == "__main__":
    # Search for games
    query = "relaxing puzzle game with beautiful art"
    results = search_games(query, top_k=5)

    print(f"Search results for: '{query}'\\n")
    print("=" * 80)

    for i, game in enumerate(results, 1):
        print(f"\\n{i}. {game['name']} (Similarity: {game['similarity_score']:.3f})")
        print(f"   Game ID: {game['game_id']}")
        print(f"   Summary: {game['summary'][:200]}...")
        if game['positive_themes']:
            print(f"   Positive Themes: {', '.join(game['positive_themes'][:5])}")
        if game['negative_themes']:
            print(f"   Negative Themes: {', '.join(game['negative_themes'][:3])}")
```

### Key Points

- **PGVector Tables**: Queries `langchain_pg_embedding` joined with `langchain_pg_collection`
- **Metadata Access**: Use `cmetadata->>'field'` for text and `cmetadata->'field'` for JSON arrays
- **Cosine Distance Operator** (`<=>`): Returns distance (0 = identical, 2 = opposite)
- **Cosine Similarity**: Use `1 - (embedding <=> query)` to get similarity (0 to 1)
- The query is automatically ordered by distance (closest matches first)

---

## Method 2: Using LangChain's PGVector Integration (Recommended)

LangChain provides a `PGVector` class that wraps pgvector functionality with a cleaner interface. This is the same method used in `embed.py`.

### Setup

```bash
pip install langchain-postgres
```

### Example Code

```python
import os
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings

# Configuration
PGVECTOR_CONNECTION = "postgresql+psycopg://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

# Connect to existing vector store
vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="game_summaries",
    connection=PGVECTOR_CONNECTION,
    use_jsonb=True
)

def search_with_langchain(query: str, top_k: int = 10):
    """
    Search using LangChain's PGVector integration.

    Args:
        query: User's search query
        top_k: Number of results to return

    Returns:
        List of Document objects with content and metadata
    """
    # Similarity search
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs

def search_with_scores(query: str, top_k: int = 10):
    """
    Search and return similarity scores.
    """
    # Similarity search with relevance scores
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=top_k)
    return docs_with_scores


# Example usage
if __name__ == "__main__":
    query = "competitive multiplayer with strategic depth"

    # Basic search
    results = search_with_langchain(query, top_k=5)

    print(f"Search results for: '{query}'\\n")
    print("=" * 80)

    for i, doc in enumerate(results, 1):
        print(f"\\n{i}. Document:")
        print(f"   Content: {doc.page_content[:200]}...")
        print(f"   Metadata: {doc.metadata}")

    print("\\n\\n" + "=" * 80)
    print("With similarity scores:")
    print("=" * 80)

    # Search with scores
    results_with_scores = search_with_scores(query, top_k=5)

    for i, (doc, score) in enumerate(results_with_scores, 1):
        print(f"\\n{i}. Score: {score:.3f}")
        print(f"   Content: {doc.page_content[:200]}...")
```

---

## Method 3: Advanced Search with Filters

You can combine semantic search with metadata filters for more precise results using JSONB operators.

### Example Code

```python
import os
import psycopg2
from langchain_openai import OpenAIEmbeddings
import json

DB_URI = "postgresql://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

def advanced_search(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.7,
    required_positive_themes: list = None,
    excluded_negative_themes: list = None
):
    """
    Advanced search with filters on themes and similarity threshold.

    Args:
        query: User's search query
        top_k: Number of results to return
        min_similarity: Minimum similarity score (0-1)
        required_positive_themes: List of themes that must be present
        excluded_negative_themes: List of negative themes to avoid
    """
    # Generate query embedding
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL
    )

    query_embedding = embeddings.embed_query(query)

    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    try:
        # Build dynamic WHERE clauses for JSONB filters
        where_clauses = ["lpc.name = 'game_summaries'"]
        params = [query_embedding, query_embedding]

        # Add similarity threshold
        where_clauses.append(f"1 - (lpe.embedding <=> %s::vector) >= %s")
        params.append(query_embedding)
        params.append(min_similarity)

        # Filter by required positive themes
        if required_positive_themes:
            for theme in required_positive_themes:
                where_clauses.append("lpe.cmetadata->'positive_themes' @> %s::jsonb")
                params.append(json.dumps([theme]))

        # Exclude games with certain negative themes
        if excluded_negative_themes:
            for theme in excluded_negative_themes:
                where_clauses.append("NOT (lpe.cmetadata->'negative_themes' @> %s::jsonb)")
                params.append(json.dumps([theme]))

        where_clause = " AND ".join(where_clauses)

        query_sql = f"""
            SELECT
                lpe.cmetadata->>'game_id' AS game_id,
                lpe.cmetadata->>'game_name' AS game_name,
                lpe.document AS summary,
                lpe.cmetadata->'positive_themes' AS positive_themes,
                lpe.cmetadata->'negative_themes' AS negative_themes,
                lpe.cmetadata->>'review_count_used' AS review_count,
                1 - (lpe.embedding <=> %s::vector) AS similarity_score
            FROM langchain_pg_embedding lpe
            JOIN langchain_pg_collection lpc ON lpe.collection_id = lpc.uuid
            WHERE {where_clause}
            ORDER BY lpe.embedding <=> %s::vector
            LIMIT %s;
        """

        params.append(top_k)
        cur.execute(query_sql, params)

        results = []
        for row in cur.fetchall():
            results.append({
                'game_id': row[0],
                'name': row[1],
                'summary': row[2],
                'positive_themes': row[3],
                'negative_themes': row[4],
                'review_count': row[5],
                'similarity_score': float(row[6])
            })

        return results
    finally:
        cur.close()
        conn.close()


# Example usage
if __name__ == "__main__":
    # Find games that are:
    # - Similar to "strategic gameplay with base building"
    # - Must have "great graphics" as a positive theme
    # - Should NOT have "poor optimization" as a negative theme
    # - Minimum 70% similarity

    results = advanced_search(
        query="strategic gameplay with base building",
        top_k=10,
        min_similarity=0.7,
        required_positive_themes=["great graphics"],
        excluded_negative_themes=["poor optimization"]
    )

    print(f"Found {len(results)} games matching criteria\\n")
    print("=" * 80)

    for i, game in enumerate(results, 1):
        print(f"\\n{i}. {game['name']}")
        print(f"   Similarity: {game['similarity_score']:.3f}")
        print(f"   Reviews: {game['positive_reviews']} positive, {game['negative_reviews']} negative")
        print(f"   Summary: {game['summary'][:150]}...")
```

---

## Method 4: Using LangChain with RAG (Retrieval-Augmented Generation)

Combine vector search with LLM to provide conversational game recommendations.

### Example Code

```python
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_postgres import PGVector
from langchain.prompts import PromptTemplate

# Configuration
PGVECTOR_CONNECTION = "postgresql+psycopg://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Initialize components
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

llm = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.7
)

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name="game_summaries",
    connection=PGVECTOR_CONNECTION,
    use_jsonb=True
)

# Create custom prompt
prompt_template = """You are a helpful Steam game recommendation assistant. Use the following game information to answer the user's question.

Context from game reviews:
{context}

User Question: {question}

Provide a helpful response that:
1. Recommends games that match their criteria
2. Explains why each game is a good fit based on player reviews
3. Mentions both positive aspects and potential concerns

Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create retrieval chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

def get_game_recommendations(query: str):
    """
    Get conversational game recommendations using RAG.
    """
    result = qa_chain.invoke({"query": query})
    return result


# Example usage
if __name__ == "__main__":
    query = "I want a relaxing game with beautiful visuals that I can play after work to unwind"

    result = get_game_recommendations(query)

    print("User Query:")
    print(query)
    print("\\n" + "=" * 80)
    print("\\nAssistant Response:")
    print(result['result'])
    print("\\n" + "=" * 80)
    print("\\nSource Games:")
    for i, doc in enumerate(result['source_documents'], 1):
        print(f"{i}. {doc.metadata.get('name', 'Unknown')}")
```

---

## Understanding pgvector Operators

| Operator | Description | Use Case |
|----------|-------------|----------|
| `<=>` | Cosine distance (0 to 2) | Most common for semantic search |
| `<->` | L2 distance (Euclidean) | When magnitude matters |
| `<#>` | Inner product | When vectors are normalized |

For semantic search, **cosine distance** (`<=>`) is recommended as it measures the angle between vectors, not magnitude.

---

## Performance Tips

1. **Index**: The IVFFlat index speeds up searches (already created in `embed.py`)
2. **Batch queries**: If searching multiple queries, reuse the embeddings model instance
3. **Connection pooling**: Use `psycopg2.pool` for production applications
4. **Limit results**: Use `LIMIT` to return only what you need
5. **Pre-filter**: Apply filters before vector search when possible

---

## Example: Building a Simple Search API

```python
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
import os
from langchain_openai import OpenAIEmbeddings
import psycopg2

app = FastAPI(title="Steam Game Search API")

DB_URI = "postgresql://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Initialize embeddings globally
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

class GameResult(BaseModel):
    game_id: str
    name: str
    summary: str
    positive_themes: List[str]
    negative_themes: List[str]
    similarity_score: float

@app.get("/search", response_model=List[GameResult])
def search_games(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=50, description="Number of results")
):
    """Search for games using semantic search."""

    # Generate embedding
    query_embedding = embeddings.embed_query(q)

    # Query database
    conn = psycopg2.connect(DB_URI)
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT
                gs.game_id,
                g.name,
                gs.summary_text,
                gs.positive_themes,
                gs.negative_themes,
                1 - (gs.summary_embedding <=> %s::vector) AS similarity_score
            FROM game_summaries gs
            JOIN games g ON gs.game_id = g.game_id
            ORDER BY gs.summary_embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, query_embedding, limit))

        results = []
        for row in cur.fetchall():
            results.append(GameResult(
                game_id=row[0],
                name=row[1],
                summary=row[2],
                positive_themes=row[3],
                negative_themes=row[4],
                similarity_score=float(row[5])
            ))

        return results
    finally:
        cur.close()
        conn.close()

# Run with: uvicorn search_api:app --reload
```

Access at: `http://localhost:8000/search?q=fast-paced%20action%20game&limit=5`

---

## Next Steps

1. **Add caching**: Cache embeddings for common queries
2. **Add filters**: Allow users to filter by genres, tags, review scores
3. **Hybrid search**: Combine vector search with keyword search
4. **Re-ranking**: Use a more powerful model to re-rank results
5. **User feedback**: Track clicks to improve search quality

---

## Resources

- [pgvector documentation](https://github.com/pgvector/pgvector)
- [LangChain PGVector](https://python.langchain.com/docs/integrations/vectorstores/pgvector)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
