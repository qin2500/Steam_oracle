from langchain_openai import ChatOpenAI
from ..config.settings import settings

def get_llm(model_name: str = None, temperature: float = 0):
    """
    Returns a configured ChatOpenAI instance using OpenRouter.
    """
    model = model_name or settings.DEFAULT_MODEL
    
    return ChatOpenAI(
        model=model,
        api_key=settings.OPENROUTER_API_KEY,
        base_url=settings.OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=2048 # Ensure enough space for structured JSON
    )
