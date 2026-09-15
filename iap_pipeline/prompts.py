"""
Prompt templates for the IAP pipeline.

These prompts operationalize the "Theory of Mind" approach:
  Step 2 — Extract latent intent vectors for each speaker independently.
  Step 3 — Combine intents with the original dialogue for final classification.
"""

DATASET_TECHNIQUES_DEFAULT = [
    "Accusation",
    "Brandishing Anger",
    "Denial",
    "Evasion",
    "Feigning Innocence",
    "Intimidation",
    "Persuasion or Seduction",
    "Playing Servant Role",
    "Playing Victim Role",
    "Rationalization",
    "Shaming or Belittlement",
]

# ---------------------------------------------------------------------------
# Step 2: Intent Extraction
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """\
You are an expert psycholinguist specializing in pragmatics and conversational implicature. \
Your task is to perform a deep analysis of one speaker's hidden, latent intentions in a conversation.

IMPORTANT RULES:
- Focus ONLY on the speaker you are asked about.
- Go beyond the literal, surface-level meaning of their words.
- Identify the psychological and relational goals driving their statements.
- Consider what they might be trying to make the other person feel, believe, or do.
- Look for patterns such as: guilt-tripping, shifting blame, seeking validation, \
establishing dominance, creating dependency, dismissing concerns, gaslighting, \
rationalizing harmful behavior, or appealing to emotion over logic.
- Be concise but thorough. Output a structured summary of 3-8 bullet points.
- If no hidden intent is apparent, state that the speaker's communication appears direct and transparent."""

INTENT_USER_TEMPLATE = """\
### Conversation
{dialogue}

### Task
Analyze the latent, hidden intentions of **{speaker}** in the conversation above.

For each intention you identify, provide:
1. **Intent**: A concise label (e.g., "Guilt induction", "Blame deflection").
2. **Evidence**: The specific statement(s) that reveal this intent.
3. **Psychological goal**: What {speaker} is trying to achieve emotionally or relationally.

Return your analysis as a structured list."""


# ---------------------------------------------------------------------------
# Step 3: Final Classification
# ---------------------------------------------------------------------------

def build_classify_system_prompt(allowed_techniques: list[str] | None = None) -> str:
    """Build the classification prompt with the active dataset taxonomy."""
    techniques = []
    seen: set[str] = set()
    for label in (allowed_techniques or DATASET_TECHNIQUES_DEFAULT):
        cleaned = str(label).strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        techniques.append(cleaned)

    techniques_text = ", ".join(techniques)

    return f"""\
You are an expert in detecting psychological manipulation in interpersonal conversations. \
You will be given:
  (1) A multi-turn conversation between two people.
  (2) An independent analysis of Person A's hidden intentions.
  (3) An independent analysis of Person B's hidden intentions.

Your task is to evaluate how the two speakers' intentions interact and determine \
whether psychological manipulation is occurring.

CLASSIFICATION CRITERIA:
- Manipulation is present when one speaker systematically exploits the other's \
emotions, cognitive biases, or vulnerabilities to serve their own goals at the \
other's expense.
- Key markers include: asymmetric power dynamics, emotional exploitation, \
distortion of reality, undermining autonomy, and covert agenda advancement.
- The mere presence of persuasion, disagreement, or emotional expression is \
NOT sufficient to classify a dialogue as manipulative.

OUTPUT FORMAT — respond with EXACTLY this JSON structure and nothing else:
{{
  "manipulative": 0 or 1,
  "confidence": <float 0.0-1.0>,
  "technique": "<comma-separated list of techniques, or empty string>",
  "explanation": "<1-2 sentence justification>"
}}

Technique labels MUST use only this dataset taxonomy (if applicable):
{techniques_text}"""


CLASSIFY_SYSTEM_PROMPT = build_classify_system_prompt()

CLASSIFY_USER_TEMPLATE = """\
### Original Conversation
{dialogue}

### Intent Analysis — Person A
{intent_a}

### Intent Analysis — Person B
{intent_b}

### Task
Based on the conversation and the two intent analyses above, determine whether \
psychological manipulation is occurring. Respond with the JSON structure specified \
in your instructions."""
