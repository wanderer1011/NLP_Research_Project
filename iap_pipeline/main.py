#!/usr/bin/env python3
"""
main.py — CLI entry point for the Intent-Aware Prompting (IAP) pipeline.

Usage examples:
  # Run on 20 dialogues with OpenAI gpt-4o-mini (default)
  python main.py --dataset data/mentalmanip_maj.csv --max 20

  # Run with Anthropic Claude
  python main.py --provider anthropic --model claude-sonnet-4-20250514 --max 50

  # Run on the full dataset
  python main.py --dataset data/mentalmanip_maj.csv
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from config import Config
from pipeline import run_pipeline


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Intent-Aware Prompting (IAP) pipeline for manipulation detection",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "groq"],
        default="groq",
        help="LLM provider (default: groq)",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Model name (default: llama-3.3-70b-versatile)",
    )
    parser.add_argument(
        "--dataset",
        default="mentalmanip_maj.csv",
        help="Path to MentalManip CSV file",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        dest="max_dialogues",
        help="Max dialogues to process (0 = all, default: 0)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent API calls (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="LLM temperature (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        default="results",
        help="Output directory (default: results)",
    )

    args = parser.parse_args()

    config = Config(
        provider=args.provider,
        model=args.model,
        dataset_path=args.dataset,
        max_dialogues=args.max_dialogues,
        concurrency=args.concurrency,
        temperature=args.temperature,
        output_dir=args.output,
    )

    # Validate API key is set
    key = config.api_key()
    if not key:
        env_vars = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY"}
        env_var = env_vars[config.provider]
        print(f"ERROR: {env_var} environment variable is not set.", file=sys.stderr)
        print(f"  Set it with:  $env:{env_var} = 'your-key-here'", file=sys.stderr)
        sys.exit(1)

    return config


def main() -> None:
    config = parse_args()

    print("=" * 50)
    print("  Intent-Aware Prompting (IAP) Pipeline")
    print("=" * 50)
    print(f"  Provider    : {config.provider}")
    print(f"  Model       : {config.model}")
    print(f"  Dataset     : {config.dataset_path}")
    print(f"  Max items   : {'all' if config.max_dialogues == 0 else config.max_dialogues}")
    print(f"  Concurrency : {config.concurrency}")
    print(f"  Temperature : {config.temperature}")
    print(f"  Output dir  : {config.output_dir}")
    print("=" * 50 + "\n")

    asyncio.run(run_pipeline(config))


if __name__ == "__main__":
    main()
