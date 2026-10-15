"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter."""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("API key is not set — please set GROQ_API_KEY or OPENROUTER_API_KEY in .env")
    
    default_base_url = "https://api.groq.com/openai/v1" if os.environ.get("GROQ_API_KEY") or (api_key and api_key.startswith("gsk_")) else "https://openrouter.ai/api/v1"
    default_model = "llama-3.3-70b-versatile" if default_base_url.startswith("https://api.groq.com") else "openai/gpt-4o-mini"
    
    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", default_model),
        base_url=os.environ.get("LLM_BASE_URL", default_base_url),
        api_key=api_key,
        temperature=temperature,
    )
