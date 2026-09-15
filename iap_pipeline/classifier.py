"""
Step 3 — Final Classification.

Combines the original dialogue with the two intent vectors (i_A, i_B)
and a specialized manipulation-detection prompt to produce the final
binary/multi-label classification:

    r = LLM(D, i_A, i_B, P_MD)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import Config
from llm_client import LLMResponse, call_llm
from prompts import CLASSIFY_USER_TEMPLATE, build_classify_system_prompt


@dataclass
class ClassificationResult:
    manipulative: int  # 0 or 1
    confidence: float
    technique: str
    explanation: str
    base_assistant_response: str = ""
    corrected_assistant_response: str = ""
    corrected_response: str = ""
    raw_response: str = ""


def _normalize_label_key(label: str) -> str:
    """Normalize label text so minor phrasing/punctuation differences match."""
    tokens = re.findall(r"[a-z0-9]+", label.casefold())
    if not tokens:
        return ""
    tokens = [t for t in tokens if t != "the"]
    return " ".join(tokens)


def _canonicalize_techniques(technique_text: str, allowed_techniques: list[str] | None) -> str:
    """Map free-form LLM technique text to canonical dataset labels."""
    text = (technique_text or "").strip()
    if not text:
        return ""

    if not allowed_techniques:
        return text

    lookup: dict[str, str] = {}
    for label in allowed_techniques:
        cleaned = str(label).strip()
        if not cleaned:
            continue
        key = _normalize_label_key(cleaned)
        if key and key not in lookup:
            lookup[key] = cleaned

    selected: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,;|/]+", text):
        cleaned = part.strip().strip('"\'')
        if not cleaned:
            continue
        key = _normalize_label_key(cleaned)
        canonical = lookup.get(key)
        if canonical and canonical.casefold() not in seen:
            seen.add(canonical.casefold())
            selected.append(canonical)

    return ", ".join(selected)


def _parse_classification(raw: str, allowed_techniques: list[str] | None = None) -> ClassificationResult:
    """
    Parse the JSON output from the classification LLM call.
    Handles minor formatting issues (markdown fences, trailing commas).
    """
    text = raw.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Last-resort: try to extract a JSON object from the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return ClassificationResult(
                manipulative=0,
                confidence=0.0,
                technique="",
                explanation=f"PARSE_ERROR: {raw[:200]}",
                corrected_response="",
                raw_response=raw,
            )

    manipulative = int(data.get("manipulative", 0))
    confidence = float(data.get("confidence", 0.0))
    explanation = str(data.get("explanation", ""))

    technique_raw = str(data.get("technique", ""))
    technique = "" if manipulative == 0 else _canonicalize_techniques(technique_raw, allowed_techniques)

    normalized = {
        "manipulative": manipulative,
        "confidence": confidence,
        "technique": technique,
        "explanation": explanation,
    }

    return ClassificationResult(
        manipulative=normalized["manipulative"],
        confidence=normalized["confidence"],
        technique=normalized["technique"],
        explanation=normalized["explanation"],
        corrected_response=json.dumps(normalized),
        raw_response=raw,
    )


async def classify_dialogue(
    config: Config,
    dialogue: str,
    intent_a: str,
    intent_b: str,
    allowed_techniques: list[str] | None = None,
) -> ClassificationResult:
    """
    Run the final IAP classification: r = LLM(D, i_A, i_B, P_MD).

    Parameters
    ----------
    dialogue : str
        Original conversation text.
    intent_a : str
        Intent summary for Person A (from Step 2).
    intent_b : str
        Intent summary for Person B (from Step 2).

    Returns
    -------
    ClassificationResult with the predicted label and metadata.
    """
    user_prompt = CLASSIFY_USER_TEMPLATE.format(
        dialogue=dialogue,
        intent_a=intent_a,
        intent_b=intent_b,
    )
    system_prompt = build_classify_system_prompt(allowed_techniques)
    resp: LLMResponse = await call_llm(
        config,
        system=system_prompt,
        user=user_prompt,
        max_tokens=config.classify_max_tokens,
    )
    return _parse_classification(resp.text, allowed_techniques=allowed_techniques)
