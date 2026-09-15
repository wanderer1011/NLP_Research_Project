#!/usr/bin/env python3
"""
main.py - CLI entry point for the Similarity/Retrieval-Based Detection pipeline.

Usage examples:
  python main.py --dataset ../mentalmanip_maj.csv --k 5
"""

import argparse
from config import Config
from pipeline import run_pipeline

def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Similarity/Retrieval-Based Detection for manipulation detection",
    )
    parser.add_argument(
        "--dataset",
        default="../mentalmanip_maj.csv",
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
        "--k",
        type=int,
        default=5,
        dest="k_neighbors",
        help="Number of nearest neighbors to retrieve (default: 5)",
    )
    parser.add_argument(
        "--k-labels",
        type=int,
        default=None,
        dest="k_neighbors_labels",
        help="Optional nearest neighbors used for technique/vulnerability labels",
    )
    parser.add_argument(
        "--retrieval-mode",
        choices=["auto", "global", "per_class", "linear"],
        default="auto",
        help="Scoring strategy: auto, global, per_class, or linear (default: auto)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio from train data for threshold tuning (default: 0.15)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        dest="decision_threshold",
        help="Fixed decision threshold in [0,1]. If omitted, tuned on validation split.",
    )
    parser.add_argument(
        "--threshold-metric",
        choices=["accuracy", "precision", "recall", "f1"],
        default="f1",
        help="Metric used to tune threshold (default: f1)",
    )
    parser.add_argument(
        "--min-recall-target",
        type=float,
        default=0.90,
        help="Minimum recall target when tuning threshold (default: 0.90)",
    )
    parser.add_argument(
        "--max-fpr-target",
        type=float,
        default=None,
        help="Optional max false positive rate target when tuning threshold",
    )
    parser.add_argument(
        "--fpr-penalty",
        type=float,
        default=0.35,
        help="Penalty for FPR during threshold/mode selection (default: 0.35)",
    )
    parser.add_argument(
        "--hard-fpr-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When constraints fail, fallback to minimum-FPR threshold first (default: enabled)",
    )
    parser.add_argument(
        "--threshold-steps",
        type=int,
        default=99,
        help="Number of candidate thresholds in [0.01, 0.99] (default: 99)",
    )
    parser.add_argument(
        "--similarity-power",
        type=float,
        default=2.0,
        help="Exponent applied to similarity for weighted voting (default: 2.0)",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        dest="calibration_method",
        help="Optional score calibration method (default: none)",
    )
    parser.add_argument(
        "--uncertainty-margin",
        type=float,
        default=0.0,
        help="Mark prediction uncertain when |score-threshold| <= margin (default: 0.0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        dest="embedding_batch_size",
        help="Embedding batch size (default: 64)",
    )
    parser.add_argument(
        "--top-matches",
        type=int,
        default=3,
        dest="top_matches_to_save",
        help="Number of nearest exemplars saved per prediction (default: 3)",
    )
    parser.add_argument(
        "--technique-prediction-mode",
        choices=["neighbors", "linear", "hybrid"],
        default="hybrid",
        dest="technique_prediction_mode",
        help="Technique scoring mode: neighbors, linear, or hybrid (default: hybrid)",
    )
    parser.add_argument(
        "--technique-min-score",
        type=float,
        default=0.45,
        dest="technique_min_score",
        help="Minimum normalized score to include a predicted technique label (default: 0.45)",
    )
    parser.add_argument(
        "--technique-max-labels",
        type=int,
        default=3,
        dest="technique_max_labels",
        help="Maximum number of predicted technique labels per dialogue (default: 3)",
    )
    parser.add_argument(
        "--technique-min-support",
        type=int,
        default=5,
        dest="technique_min_support",
        help="Minimum positive support for a technique label in linear mode (default: 5)",
    )
    parser.add_argument(
        "--technique-linear-weight",
        type=float,
        default=0.70,
        dest="technique_linear_weight",
        help="Linear score weight in hybrid technique mode, in [0,1] (default: 0.70)",
    )
    parser.add_argument(
        "--technique-force-one",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force at least one predicted technique when manipulative (default: disabled)",
    )
    parser.add_argument(
        "--exclude-unlabeled-positive-from-label-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="exclude_unlabeled_positive_from_label_metrics",
        help="Exclude positives without technique annotation from label metrics (default: enabled)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of repeated runs with different seeds (default: 1)",
    )
    parser.add_argument(
        "--seed-step",
        type=int,
        default=101,
        help="Seed increment between repeated runs (default: 101)",
    )
    parser.add_argument(
        "--save-threshold-curve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save threshold sweep points to JSON (default: enabled)",
    )
    parser.add_argument(
        "--cache-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache embeddings for faster repeated runs (default: enabled)",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache",
        help="Directory for cached artifacts (default: .cache)",
    )
    parser.add_argument(
        "--include-text-in-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include raw dialogue text in predictions.csv (default: enabled)",
    )
    parser.add_argument(
        "--output",
        default="similarity_results",
        help="Output directory (default: similarity_results)",
    )
    parser.add_argument(
        "--model",
        default="j-hartmann/emotion-english-distilroberta-base",
        dest="embedding_model",
        help="Sentence Transformer model name (default: j-hartmann/emotion-english-distilroberta-base)",
    )

    args = parser.parse_args()

    config = Config(
        dataset_path=args.dataset,
        max_dialogues=args.max_dialogues,
        k_neighbors=args.k_neighbors,
        k_neighbors_labels=args.k_neighbors_labels,
        retrieval_mode=args.retrieval_mode,
        val_ratio=args.val_ratio,
        decision_threshold=args.decision_threshold,
        threshold_metric=args.threshold_metric,
        min_recall_target=args.min_recall_target,
        max_fpr_target=args.max_fpr_target,
        fpr_penalty=args.fpr_penalty,
        hard_fpr_fallback=args.hard_fpr_fallback,
        threshold_steps=args.threshold_steps,
        similarity_power=args.similarity_power,
        calibration_method=args.calibration_method,
        uncertainty_margin=args.uncertainty_margin,
        embedding_batch_size=args.embedding_batch_size,
        top_matches_to_save=args.top_matches_to_save,
        technique_prediction_mode=args.technique_prediction_mode,
        technique_min_score=args.technique_min_score,
        technique_max_labels=args.technique_max_labels,
        technique_min_support=args.technique_min_support,
        technique_linear_weight=args.technique_linear_weight,
        technique_force_one=args.technique_force_one,
        exclude_unlabeled_positive_from_label_metrics=args.exclude_unlabeled_positive_from_label_metrics,
        num_runs=args.num_runs,
        seed_step=args.seed_step,
        save_threshold_curve=args.save_threshold_curve,
        cache_embeddings=args.cache_embeddings,
        cache_dir=args.cache_dir,
        include_text_in_output=args.include_text_in_output,
        output_dir=args.output,
        embedding_model=args.embedding_model,
    )
    return config

def main() -> None:
    config = parse_args()

    print("=" * 50)
    print("  Similarity/Retrieval-Based Pipeline")
    print("=" * 50)
    print(f"  Dataset       : {config.dataset_path}")
    print(f"  Max items     : {'all' if config.max_dialogues == 0 else config.max_dialogues}")
    print(f"  KNN Neighbors : {config.k_neighbors}")
    print(f"  K Labels      : {config.k_neighbors_labels if config.k_neighbors_labels is not None else 'auto'}")
    print(f"  Retrieval mode: {config.retrieval_mode}")
    print(f"  Val ratio     : {config.val_ratio}")
    print(f"  Threshold     : {'auto' if config.decision_threshold is None else config.decision_threshold}")
    print(f"  Thr metric    : {config.threshold_metric}")
    print(f"  Min recall    : {config.min_recall_target}")
    print(f"  Max FPR       : {'none' if config.max_fpr_target is None else config.max_fpr_target}")
    print(f"  FPR penalty   : {config.fpr_penalty}")
    print(f"  Hard FPR fallback: {config.hard_fpr_fallback}")
    print(f"  Calibration   : {config.calibration_method}")
    print(f"  Uncertainty m.: {config.uncertainty_margin}")
    print(f"  Embed Model   : {config.embedding_model}")
    print(f"  Batch size    : {config.embedding_batch_size}")
    print(f"  Tech mode     : {config.technique_prediction_mode}")
    print(f"  Tech min score: {config.technique_min_score}")
    print(f"  Tech max labels: {config.technique_max_labels}")
    print(f"  Tech min support: {config.technique_min_support}")
    print(f"  Tech linear wt: {config.technique_linear_weight}")
    print(f"  Force one tech: {config.technique_force_one}")
    print(f"  Excl unlabeled+: {config.exclude_unlabeled_positive_from_label_metrics}")
    print(f"  Num runs      : {config.num_runs}")
    print(f"  Seed step     : {config.seed_step}")
    print(f"  Save thr curve: {config.save_threshold_curve}")
    print(f"  Cache embeds  : {config.cache_embeddings} ({config.cache_dir})")
    print(f"  Include text  : {config.include_text_in_output}")
    print(f"  Output dir    : {config.output_dir}")
    print("=" * 50 + "\n")

    run_pipeline(config)

if __name__ == "__main__":
    main()
