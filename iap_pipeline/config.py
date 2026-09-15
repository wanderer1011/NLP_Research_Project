"""
Configuration for the Intent-Aware Prompting (IAP) pipeline.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- LLM Provider ---
    # "openai", "anthropic", or "groq"
    provider: str = "groq"

    # Model to use for inference
    # OpenAI:    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", etc.
    # Anthropic: "claude-sonnet-4-20250514", "claude-3-haiku-20240307", etc.
    # Groq:      "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", etc.
    model: str = "llama-3.3-70b-versatile"

    # --- API Keys (read from environment) ---
    openai_api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))

    # --- Dataset ---
    # Path to the MentalManip CSV file
    dataset_path: str = "mentalmanip_maj.csv"

    # --- Pipeline ---
    # Max dialogues to process (0 = all)
    max_dialogues: int = 0

    # Temperature for LLM calls
    temperature: float = 0.2

    # Max tokens for intent summaries
    intent_max_tokens: int = 512

    # Max tokens for final classification
    classify_max_tokens: int = 256

    # Concurrent API calls (be mindful of rate limits)
    concurrency: int = 5

    # --- Output ---
    output_dir: str = "results"

    def api_key(self) -> str:
        if self.provider == "openai":
            return self.openai_api_key
        elif self.provider == "anthropic":
            return self.anthropic_api_key
        elif self.provider == "groq":
            return self.groq_api_key
        raise ValueError(f"Unknown provider: {self.provider}")
