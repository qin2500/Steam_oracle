# Game Review Embedding Setup Instructions

## Project Overview
Build a pipeline to create vector embeddings for the top 1000 games in our PostgreSQL database. These embeddings will power semantic search based on what players say in reviews.

## Database Connection
```
postgresql://vec_search:sPvDZbcMAR5yH8pyG76Xa7uq@138.197.137.22:5234/steam_db
```

## Tech Stack Requirements
- **Python 3.9+**
- **LangChain** for LLM orchestration
- **OpenRouter** for all LLM and embedding API calls
- **pgvector** for PostgreSQL vector support
- **psycopg2** for database connection

## Task Breakdown

### Step 1: Database Setup
Create a new table called `game_summaries` with the following schema:

```sql
CREATE TABLE game_summaries (
    game_id VARCHAR PRIMARY KEY REFERENCES games(game_id),
    summary_text TEXT NOT NULL,
    summary_embedding vector(1536),
    positive_themes JSONB,
    negative_themes JSONB,
    review_count_used INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Requirements:**
- Enable the pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector`
- Create a vector similarity index on `summary_embedding` using IVFFlat with cosine similarity
- Use 1536 dimensions (for OpenAI text-embedding-3-small model)

### Step 2: Identify Top 1000 Games
Query the `games` table to get the top 1000 games by total review count:
- Filter: `reviews_processed = true`
- Sort by: `(positive_reviews + negative_reviews) DESC`
- Limit: 1000

### Step 3: Review Selection Strategy
For each game, gather reviews using this strategy:

**Top Reviews (by weighted_vote_score):**
- Top 50 positive reviews (`is_positive = true`)
- Top 50 negative reviews (`is_positive = false`)

**Random Sample (for variety):**
- 10 random positive reviews
- 10 random negative reviews

**Filters:**
- Only reviews where `text IS NOT NULL`
- Order top reviews by `weighted_vote_score DESC NULLS LAST`
- Use `ORDER BY RANDOM()` for random selection

### Step 4: Generate Summaries with LangChain
Use Claude (via OpenRouter) to analyze the reviews and extract:

1. **Summary**: 3-4 sentence overview of what players think about the game
2. **Positive Themes**: 5-7 short phrases (e.g., "balanced gameplay", "great graphics")
3. **Negative Themes**: 5-7 short phrases (e.g., "poor optimization", "steep learning curve")

**Prompt Guidelines:**
- Truncate each review to max 500 characters to stay within context limits
- Label reviews as [TOP POSITIVE], [TOP NEGATIVE], [RANDOM POSITIVE], [RANDOM NEGATIVE]
- Request JSON output format:
  ```json
  {
    "summary": "...",
    "positive_themes": ["theme1", "theme2", ...],
    "negative_themes": ["theme1", "theme2", ...]
  }
  ```

**LangChain Configuration:**
- Model: `anthropic/claude-3.5-sonnet`
- OpenRouter API base: `https://openrouter.ai/api/v1`
- Temperature: 0.3

### Step 5: Create Embeddings
Use OpenAI's text-embedding-3-small model (via OpenRouter) to create embeddings:
- Input: The `summary` text from step 4
- Model: `text-embedding-3-small`
- Output: 1536-dimensional vector

### Step 6: Store Results
Insert/update the `game_summaries` table with:
- `game_id`: The game's ID
- `summary_text`: The generated summary
- `summary_embedding`: The embedding vector
- `positive_themes`: JSON array of positive themes
- `negative_themes`: JSON array of negative themes
- `review_count_used`: Total number of reviews processed

Use `ON CONFLICT (game_id) DO UPDATE` to handle re-runs.

## Implementation Requirements

### Error Handling
- Wrap each game's processing in try-except
- Log failures but continue processing remaining games
- Print progress: `[X/1000] Processing: Game Name (game_id)`

### Batch Processing
- Process all 1000 games sequentially
- Print summary at end: success count vs. failures

### Performance Considerations
- This will take time (1000 games × API calls)
- Consider adding a progress bar (optional)
- Save progress periodically in case of interruption (optional enhancement)

## Expected Output
After running the script:
- `game_summaries` table populated with 1000 entries (or as many as successfully processed)
- Each entry has a summary, embedding, and extracted themes
- Console output showing progress and final statistics

## Validation Steps
After completion, verify:
1. Row count: `SELECT COUNT(*) FROM game_summaries;`
2. Sample data: `SELECT game_id, summary_text, positive_themes FROM game_summaries LIMIT 5;`
3. Embeddings exist: `SELECT COUNT(*) FROM game_summaries WHERE summary_embedding IS NOT NULL;`

## Environment Variables Needed
- `OPENROUTER_API_KEY`: Your OpenRouter API key

## Notes
- The existing database schema has tables: `games`, `reviews`, `users`, `ownerships`
- Games table has: `game_id`, `name`, `positive_reviews`, `negative_reviews`, `genres`, etc.
- Reviews table has: `review_id`, `game_id`, `text`, `is_positive`, `weighted_vote_score`, etc.
