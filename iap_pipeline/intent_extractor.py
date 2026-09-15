"""
Step 2 — Intent Extraction.

For each dialogue, makes two independent LLM calls to summarize the hidden,
latent intentions of Person A and Person B separately.  This operationalizes
the "Theory of Mind" modelling described in the IAP framework.
"""

from __future__ import annotations

from config import Config
from llm_client import LLMResponse, call_llm
from prompts import INTENT_SYSTEM_PROMPT, INTENT_USER_TEMPLATE


async def extract_intent(
    config: Config,
    dialogue: str,
    speaker: str,
) -> LLMResponse:
    """
    Generate an intent vector (structured summary) for one speaker.

    Parameters
    ----------
    dialogue : str
        The full multi-turn conversation text.
    speaker : str
        "Person A" or "Person B".

    Returns
    -------
    LLMResponse with the intent summary in .text
    """
    user_prompt = INTENT_USER_TEMPLATE.format(dialogue=dialogue, speaker=speaker)
    return await call_llm(
        config,
        system=INTENT_SYSTEM_PROMPT,
        user=user_prompt,
        max_tokens=config.intent_max_tokens,
    )
