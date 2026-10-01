"""Build the LLM and embedding model for the chosen provider.

    gemini  (default) - Google AI Studio free tier, no credit card. Needs GOOGLE_API_KEY.
    ollama            - runs a model locally on your computer, completely free, no key.
    openai            - paid API. Needs OPENAI_API_KEY.
"""
from __future__ import annotations

from .config import Settings

PROVIDERS = ("gemini", "ollama", "openai")


def build_llm(s: Settings):
    p = s.llm_provider
    if p == "gemini":
        from llama_index.llms.google_genai import GoogleGenAI
        return GoogleGenAI(model=s.gemini_model, temperature=0.0, max_retries=5)
    if p == "ollama":
        from llama_index.llms.ollama import Ollama
        return Ollama(model=s.ollama_model, temperature=0.0, request_timeout=180.0)
    if p == "openai":
        from llama_index.llms.openai import OpenAI
        return OpenAI(model=s.openai_model, temperature=0.0)
    raise ValueError(f"Unknown LLM_PROVIDER '{p}'. Use one of: {', '.join(PROVIDERS)}")


def build_embed_model(s: Settings):
    p = s.llm_provider
    if p == "gemini":
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        return GoogleGenAIEmbedding(model_name=s.gemini_embed_model)
    if p == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        return OllamaEmbedding(model_name=s.ollama_embed_model)
    if p == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(model=s.embed_model)
    raise ValueError(f"Unknown LLM_PROVIDER '{p}'")


def required_key(provider: str) -> str | None:
    """Name of the environment variable the provider needs, or None."""
    return {"gemini": "GOOGLE_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
