from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from ..config.settings import settings

# Global instance of embeddings to avoid re-initializing
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL
)

def get_vector_store():
    """
    Returns the PGVector store connected to the 'game_summaries' collection.
    """
    return PGVector(
        embeddings=embeddings,
        collection_name="game_summaries",
        connection=settings.LANGCHAIN_DB_URI,
        use_jsonb=True
    )
