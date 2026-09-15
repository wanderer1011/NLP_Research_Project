"""
IAP Pipeline — full orchestration.

Processes a batch of dialogues through the three-step IAP pipeline:
  1. Load data
  2. Extract intent vectors (Person A & B) in parallel
  3. Classify using dialogue + intent vectors

Supports concurrent API calls with a configurable semaphore.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
import time
from pathlib import Path

from classifier import ClassificationResult, classify_dialogue
from config import Config
from data_loader import Dialogue, load_dataset
from evaluation import Metrics, evaluate
from intent_extractor import extract_intent
from llm_client import call_llm, call_llm_with_intervention


RESPONSE_SYSTEM_PROMPT = (
    "You are a calm, safety-first assistant. "
    "Reply to the final speaker in 1-2 concise sentences."
)

RESPONSE_USER_TEMPLATE = (
    "Conversation:\n"
    "{dialogue}\n\n"
    "Task: Provide a short assistant reply to the final speaker in this conversation."
)


def _extract_dataset_techniques(dialogues: list[Dialogue]) -> list[str]:
    """Collect canonical technique labels from dataset annotations."""
    seen: set[str] = set()
    labels: list[str] = []

    for d in dialogues:
        for raw in d.techniques:
            for part in re.split(r"[|,]", raw or ""):
                cleaned = part.strip()
                if not cleaned:
                    continue
                key = cleaned.casefold()
                if key in seen:
                    continue
                seen.add(key)
                labels.append(cleaned)

    labels.sort(key=str.casefold)
    return labels


async def _process_one(
    config: Config,
    dialogue: Dialogue,
    allowed_techniques: list[str],
    semaphore: asyncio.Semaphore,
) -> ClassificationResult:
    """Run the full IAP pipeline for a single dialogue."""
    async with semaphore:
        # Step 2: Extract intents in parallel (two independent calls)
        task_a = extract_intent(config, dialogue.text, "Person A")
        task_b = extract_intent(config, dialogue.text, "Person B")
        intent_a, intent_b = await asyncio.gather(task_a, task_b)

        # Step 3: Classify
        result = await classify_dialogue(
            config,
            dialogue.text,
            intent_a.text,
            intent_b.text,
            allowed_techniques=allowed_techniques,
        )

        response_prompt = RESPONSE_USER_TEMPLATE.format(dialogue=dialogue.text)
        base_response = await call_llm(
            config,
            system=RESPONSE_SYSTEM_PROMPT,
            user=response_prompt,
            max_tokens=180,
        )
        corrected_response = await call_llm_with_intervention(
            config=config,
            system=RESPONSE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": response_prompt}],
            classification=result,
            max_tokens=180,
        )

        result.base_assistant_response = base_response.text.strip()
        result.corrected_assistant_response = corrected_response.text.strip()
        return result


def _save_results(
    output_dir: Path,
    dialogues: list[Dialogue],
    predictions: list[ClassificationResult],
    metrics: Metrics,
) -> None:
    """Persist predictions and metrics to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-dialogue predictions ---
    pred_path = output_dir / "predictions.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "ground_truth", "predicted", "confidence",
            "technique", "explanation", "base_assistant_response",
            "corrected_assistant_response", "corrected_response", "raw_response",
        ])
        for d, p in zip(dialogues, predictions):
            writer.writerow([
                d.id, d.label, p.manipulative, f"{p.confidence:.3f}",
                p.technique,
                p.explanation,
                p.base_assistant_response,
                p.corrected_assistant_response,
                p.corrected_response,
                p.raw_response,
            ])

    # --- Aggregate metrics ---
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": metrics.total,
                "accuracy": round(metrics.accuracy, 4),
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1": round(metrics.f1, 4),
                "false_negative_rate": round(metrics.false_negative_rate, 4),
                "tp": metrics.tp,
                "tn": metrics.tn,
                "fp": metrics.fp,
                "fn": metrics.fn,
            },
            f,
            indent=2,
        )

    print(f"\n✓ Predictions saved to {pred_path}")
    print(f"✓ Metrics saved to {metrics_path}")


async def run_pipeline(config: Config) -> Metrics:
    """
    Main entry point — runs the full IAP pipeline end-to-end.

    Returns the computed Metrics object.
    """
    # Step 1: Load dataset
    print(f"Loading dataset from {config.dataset_path} ...")
    dialogues = load_dataset(config.dataset_path)
    allowed_techniques = _extract_dataset_techniques(dialogues)
    if config.max_dialogues > 0:
        dialogues = dialogues[: config.max_dialogues]
    print(f"  → {len(dialogues)} dialogues loaded.\n")
    if allowed_techniques:
        print(f"  → Technique taxonomy ({len(allowed_techniques)} labels): {', '.join(allowed_techniques)}\n")

    # Steps 2+3: Process all dialogues with bounded concurrency
    semaphore = asyncio.Semaphore(config.concurrency)
    total = len(dialogues)
    completed = 0

    async def _tracked(d: Dialogue) -> ClassificationResult:
        nonlocal completed
        result = await _process_one(config, d, allowed_techniques, semaphore)
        completed += 1
        if completed % 10 == 0 or completed == total:
            print(f"  [{completed}/{total}] dialogues processed")
        return result

    start = time.time()
    predictions = await asyncio.gather(*[_tracked(d) for d in dialogues])
    elapsed = time.time() - start
    print(f"\n  Pipeline finished in {elapsed:.1f}s")

    # Evaluate
    metrics = evaluate(dialogues, list(predictions))

    # Print summary
    print("\n" + "=" * 50)
    print("  IAP Pipeline — Evaluation Results")
    print("=" * 50)
    print(f"  Total dialogues : {metrics.total}")
    print(f"  Accuracy        : {metrics.accuracy:.4f}")
    print(f"  Precision       : {metrics.precision:.4f}")
    print(f"  Recall          : {metrics.recall:.4f}")
    print(f"  F1 Score        : {metrics.f1:.4f}")
    print(f"  FN Rate         : {metrics.false_negative_rate:.4f}")
    print(f"  TP={metrics.tp}  TN={metrics.tn}  FP={metrics.fp}  FN={metrics.fn}")
    print("=" * 50)

    # Save
    _save_results(Path(config.output_dir), dialogues, list(predictions), metrics)

    return metrics
